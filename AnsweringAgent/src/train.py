import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
from utils.logger import setup_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from config import Config
from models.answering_agent import AnsweringAgent
from data.dataset import AnsweringDataset
import traceback
import time
import gc
import signal
import sys
import datetime
import logging
import tempfile
import numpy as np
import threading
import math
from transformers import T5Tokenizer
import torch.nn.functional as F
from models.contrastive_loss import ContrastiveLoss

# Global flag to track if training should continue
TRAINING_FAILED = False
CHECKPOINT_LOCK = threading.Lock()

# Create a temporary file for error logging (minimal)
temp_error_file = tempfile.NamedTemporaryFile(prefix="training_error_", suffix=".log", delete=False)

# Exponential Moving Average Implementation
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Handle newly unfrozen parameters by adding them to shadow
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                    continue
                    
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply the EMA weights to the model for evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Handle newly unfrozen parameters by adding them to shadow
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                    
                self.backup[name] = param.data.cpu()  # Store backup on CPU to save GPU memory
                param.data = self.shadow[name].clone()
    
    def restore(self):
        """Restore the original weights for training"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Handle newly unfrozen parameters that might not have backup
                if name not in self.backup:
                    continue  # Skip if no backup exists (newly unfrozen parameter)
                    
                param.data = self.backup[name].to(param.device)  # Move from CPU back to GPU
        self.backup = {}
    
    def state_dict(self):
        return {
            'decay': self.decay,
            'shadow': self.shadow,
            'backup': self.backup
        }
    
    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
        self.backup = state_dict['backup']

def setup_minimal_environment():
    """Setup minimal environment for training without excessive logging."""
    # Disable excessive PyTorch distributed logging
    os.environ["NCCL_DEBUG"] = "WARN"  # Only warnings, not INFO
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # Minimal distributed info
    
    # Memory optimizations
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        # Remove blocking for performance
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Only for debugging
        
        # Optimize NCCL for better multi-GPU communication (minimal)
        os.environ['NCCL_NSOCKS_PERTHREAD'] = '2'
        os.environ['NCCL_SOCKET_NTHREADS'] = '2'
    
    # Enable cuDNN benchmarking for performance (will be disabled later if using mixed precision)
    torch.backends.cudnn.benchmark = True
    
    # Enable TF32 for better performance (unless debugging precision issues)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def log_system_info(rank=0):
    """Log essential system information only on rank 0."""
    if rank != 0:
        return
        
    print(f"🚀 UAV Navigation Training Pipeline")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda} | Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA: Not available - using CPU")

def compute_metrics(outputs: torch.Tensor, labels: torch.Tensor, pad_token_id: int) -> Dict[str, float]:
    """Compute accuracy and other metrics."""
    # Reshape outputs and labels
    outputs_reshaped = outputs.reshape(-1, outputs.size(-1))
    labels_reshaped = labels.reshape(-1)

    # Get predictions
    _, predicted = outputs_reshaped.max(1)
    predicted = predicted.reshape(outputs.size(0), outputs.size(1))

    # Create mask for non-padding tokens
    mask = (labels != pad_token_id)

    # Calculate metrics
    total_tokens = mask.sum().item()
    correct_tokens = ((predicted == labels) & mask).sum().item()
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    return {
        'accuracy': accuracy,
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens
    }

def get_weight_schedule(start_weight: float, end_weight: float, total_epochs: int):
    """
    Returns a function that linearly increases or decreases a weight from start_weight to end_weight.

    Args:
        start_weight (float): Initial value of the weight.
        end_weight (float): Final value of the weight.
        total_epochs (int): Total number of epochs over which the weight changes.

    Returns:
        Callable[[int], float]: A function that returns the weight at a given epoch.
    """
    def weight_fn(epoch: int) -> float:
        # Clamp epoch within range
        epoch = max(0, min(epoch, total_epochs))
        return start_weight + (end_weight - start_weight) * (epoch / total_epochs)

    return weight_fn

def get_smart_curriculum_schedule(planned_epochs: int):
    """
    Smart curriculum learning schedule aligned with 3-phase training.
    After planned_epochs, curriculum stays at final value (0.0).
    
    Phase 1 (0-30%): HIGH curriculum ratio (oracle destination helps build space)
    Phase 2 (30-70%): MODERATE curriculum (gradual independence)  
    Phase 3 (70-100%): LOW curriculum (model learns to navigate independently)
    Extended Phase (>100%): FIXED at 0.0 (no oracle assistance)
    """
    phase1_end = int(0.3 * planned_epochs)  # 30% for semantic space building
    phase2_end = int(0.7 * planned_epochs)  # 70% for balanced learning
    
    def curriculum_ratio_fn(epoch: int) -> float:
        if epoch <= phase1_end:
            # Phase 1: HIGH curriculum (oracle destination helps semantic space)
            progress = epoch / phase1_end
            return 1.0 * (1 - progress) + 0.7 * progress  # 1.0 → 0.7 (strong oracle help)
        elif epoch <= phase2_end:
            # Phase 2: MODERATE curriculum (gradual independence)
            progress = (epoch - phase1_end) / (phase2_end - phase1_end)
            return 0.7 * (1 - progress) + 0.3 * progress  # 0.7 → 0.3
        elif epoch < planned_epochs:
            # Phase 3: LOW curriculum (independent navigation)
            progress = (epoch - phase2_end) / (planned_epochs - phase2_end)
            return 0.3 * (1 - progress) + 0.0 * progress  # 0.3 → 0.0 (no oracle)
        else:
            # Extended Phase: FIXED at final value (no oracle assistance)
            return 0.0
    
    return curriculum_ratio_fn

def get_smart_contrastive_schedule(planned_epochs: int, max_epochs: int):
    """
    Smart 3-phase contrastive learning schedule based on user insights:
    Phase 1 (0-30%): Build semantic space - HIGH contrastive, LOW CE
    Phase 2 (30-70%): Balance learning - MODERATE both  
    Phase 3 (70-100%): Fine-tune with CE - LOW contrastive, HIGH CE
    Extended Phase (>100%): FIXED weights, but adaptive revival still active
    
    Includes contrastive signal revival when needed.
    """
    phase1_end = int(0.3 * planned_epochs)  # 30% for semantic space building
    phase2_end = int(0.7 * planned_epochs)  # 70% for balanced learning
    
    def contrastive_weight_fn(epoch: int) -> float:
        # Use config values: start=10.0, end=5.0 (better signal preservation!)
        start_weight = 10.0  # Strong semantic space building
        end_weight = 3.0     # Maintain contrastive signal (vs killing it at 1.0)
        
        if epoch <= phase1_end:
            # Phase 1: HIGH contrastive for semantic space building
            return start_weight
        elif epoch <= phase2_end:
            # Phase 2: GRADUAL transition from high to intermediate
            progress = (epoch - phase1_end) / (phase2_end - phase1_end)
            mid_weight = start_weight * 0.7  # 10.0 → 7.0 (smooth transition)
            return start_weight * (1 - progress) + mid_weight * progress
        elif epoch < planned_epochs:
            mid_weight = start_weight * 0.7  # 7.0
            return mid_weight
        elif epoch < 450:
            # Higher weight for 75 epochs for refreshed Hard Negatives 
            progress = (epoch - planned_epochs) / (450 - planned_epochs)
            mid_weight = start_weight * 0.7
            return mid_weight * (1 - progress) + end_weight * progress # 7.0 → 3.0 for 75 epochs
        else:
            # Extended Phase: FIXED at end weight (adaptive revival still works)
            return end_weight
    
    def ce_weight_fn(epoch: int) -> float:
        if epoch <= phase1_end:
            # Phase 1: LOW CE, let contrastive build space
            return 0.1  # Minimal CE interference
        elif epoch <= phase2_end:
            # Phase 2: GRADUAL CE increase
            progress = (epoch - phase1_end) / (phase2_end - phase1_end)
            return 0.1 + (0.8 - 0.1) * progress  # 0.1 → 0.8
        elif epoch < planned_epochs:
            # Phase 3: HIGH CE for final fine-tuning
            progress = (epoch - phase2_end) / (planned_epochs - phase2_end)
            return 0.8 + (1.2 - 0.8) * progress  # 0.8 → 1.2
        elif epoch < 525:
            # Keep CE modest while KD is still high
            return 0.6
        elif epoch < 550:
            # Gentle polish: 0.6 → 0.8 over 25 epochs
            progress = (epoch - 525) / 25
            return 0.6 * (1 - progress) + 0.8 * progress   # 0.6 → 0.8
        else:
            # Fix CE at 0.8 for the remainder of training (avoid over-fitting)
            return 0.8
    
    return contrastive_weight_fn, ce_weight_fn

def get_adaptive_contrastive_schedule(base_schedule_fn, revival_threshold: float = 0.001):
    """
    Adaptive contrastive scheduling with gentle target-based revival.
    
    Args:
        base_schedule_fn: Base scheduling function
        revival_threshold: Threshold below which to trigger revival (0.002 for sensitive detection)
    """
    last_contrastive_losses = []
    flag_revival = False
    last_revival_epoch = -1  # Track when revival was last triggered
    
    def adaptive_weight_fn(epoch: int, recent_contrastive_loss: float = None) -> tuple:
        nonlocal last_contrastive_losses
        nonlocal flag_revival
        nonlocal last_revival_epoch

        base_weight = base_schedule_fn(epoch)
        revival_info = {
            'triggered': False,
            'base_weight': base_weight,
            'revival_weight': base_weight,
            'target_loss': 0.005 if epoch < 1500 else 0.004,
            'recent_avg': None,
            'epoch': epoch
        }
        
        # Only update flag when new data is provided (epoch end)
        if recent_contrastive_loss is not None:
            last_contrastive_losses.append(recent_contrastive_loss)
            if len(last_contrastive_losses) > 5:  # Keep last 5 epochs
                last_contrastive_losses.pop(0)
            
            # Only calculate and update flag when we have enough data
            if len(last_contrastive_losses) >= 2:  # Need at least 2 epochs for comparison
                recent_avg = sum(last_contrastive_losses[-4:]) / len(last_contrastive_losses[-4:])
                revival_info['recent_avg'] = recent_avg
                
                old_flag = flag_revival
                if recent_avg < revival_threshold:
                    flag_revival = True
                else:
                    flag_revival = False
                    
                # Check if this is a new revival (not just continuation)
                if flag_revival and not old_flag:
                    last_revival_epoch = epoch
        
        # Use current flag state (whether updated this call or from previous epoch)
        if flag_revival:
            
            # Calculate target loss and recent average for revival logic
            if len(last_contrastive_losses) >= 2:
                recent_avg = sum(last_contrastive_losses[-4:]) / len(last_contrastive_losses[-4:])
                target_loss = 0.005 if epoch < 1500 else 0.004  # More conservative in late epochs
                
                # Estimate weight adjustment needed (gentle approximation)
                if recent_avg > 0.0001:  # Avoid division by zero
                    # Simple linear approximation: loss ∝ 1/weight
                    estimated_weight_needed = base_weight * (target_loss / recent_avg)
                    
                    # Apply conservative bounds based on training phase
                    if epoch < 400:  # Planning phase - allow larger adjustments
                        max_weight = base_weight * 2.0
                        min_weight = base_weight * 1.2
                    elif epoch < 2000:  # Early extended phase - moderate adjustments  
                        max_weight = base_weight * 1.5
                        min_weight = base_weight * 1.1
                    else:  # Late extended phase - minimal adjustments
                        max_weight = base_weight * 1.2
                        min_weight = base_weight * 1.05
                    
                    # Clamp the estimated weight within conservative bounds
                    revival_weight = max(min_weight, min(estimated_weight_needed, max_weight))
                    
                    # Update revival info
                    revival_info.update({
                        'triggered': True,
                        'revival_weight': revival_weight,
                        'target_loss': target_loss,
                        'recent_avg': recent_avg,
                        'is_new_revival': (epoch == last_revival_epoch)
                    })
                    
                    return revival_weight, revival_info
                else:
                    # If loss is extremely small, apply minimal boost
                    gentle_boost = 1.1 if epoch > 2000 else 1.3
                    revival_weight = base_weight * gentle_boost
                    
                    # Update revival info
                    revival_info.update({
                        'triggered': True,
                        'revival_weight': revival_weight,
                        'is_new_revival': (epoch == last_revival_epoch)
                    })
                    
                    return revival_weight, revival_info
    
        # No revival needed - return base weight
        return base_weight, revival_info
    
    return adaptive_weight_fn

def get_smart_destination_schedule(planned_epochs: int):
    """
    Smart destination loss scheduling aligned with curriculum learning.
    
    Phase 1 (0-30%): HIGH destination weight (oracle available, strong supervision)
    Phase 2 (30-70%): MODERATE destination (transition to independence)  
    Phase 3 (70-100%): LOW destination (minimal oracle supervision)
    Extended Phase (>100%): FIXED at final value
    """
    phase1_end = int(0.3 * planned_epochs)  # 30% for semantic space building
    phase2_end = int(0.7 * planned_epochs)  # 70% for balanced learning
    
    def destination_weight_fn(epoch: int) -> float:
        if epoch <= phase1_end:
            # Phase 1: HIGH destination weight (oracle available)
            return 0.8  # Strong destination supervision
        elif epoch <= phase2_end:
            # Phase 2: MODERATE destination (gradual independence)
            progress = (epoch - phase1_end) / (phase2_end - phase1_end)
            return 0.8 * (1 - progress) + 0.3 * progress  # 0.8 → 0.3
        elif epoch < planned_epochs:
            # Phase 3: LOW destination (minimal supervision)
            progress = (epoch - phase2_end) / (planned_epochs - phase2_end)
            return 0.3 * (1 - progress) + 0.05 * progress  # 0.3 → 0.05
        else:
            # Extended Phase: FIXED at final value
            return 0.05
    
    return destination_weight_fn

def get_smart_kd_schedule(planned_epochs: int):
    """
    Smart Knowledge Distillation scheduling aligned with 3-phase strategy.
    
    Phase 1 (0-30%): MODERATE KD (help build semantic space alongside contrastive)
    Phase 2 (30-70%): HIGH KD (teacher guidance during transition)  
    Phase 3 (70-100%): LOW KD (focus on task-specific learning)
    Extended Phase (>100%): FIXED at final value
    """
    phase1_end = int(0.3 * planned_epochs)  # 30% for semantic space building
    phase2_end = int(0.7 * planned_epochs)  # 70% for balanced learning
    
    def kd_weight_fn(epoch: int) -> float:
        if epoch <= phase1_end:
            # Phase 1: MODERATE KD (complementary to contrastive learning)
            return 0.5  # Moderate teacher guidance
        elif epoch <= phase2_end:
            # Phase 2: HIGH KD (strongest teacher guidance during transition)
            progress = (epoch - phase1_end) / (phase2_end - phase1_end)
            return 0.5 + (1.0 - 0.5) * progress  # 0.5 → 1.0
        elif epoch < planned_epochs:
            # Phase 3: LOW KD (focus on task-specific fine-tuning)
            progress = (epoch - phase2_end) / (planned_epochs - phase2_end)
            return 1.0 * (1 - progress) + 0.7 * progress  # 1.0 → 0.7
        elif epoch < 525:
            progress = (epoch - planned_epochs) / (525 - planned_epochs)
            return 0.7 * (1 - progress) + 1.2 * progress   # 0.7 → 1.4
        else:
            return 1.4                    # fixed tail
    
    return kd_weight_fn

def calculate_reconstruction_loss(reconstructed_features, original_features):
    reconstructed_features_norm = F.normalize(reconstructed_features, p=2, dim=1)
    original_features_norm = F.normalize(original_features, p=2, dim=1)
    reconstruction_loss = F.mse_loss(reconstructed_features_norm, original_features_norm)
    return reconstruction_loss

def calculate_cosine_similarity_loss(first_features, second_features):
    first_features_norm = F.normalize(first_features, p=2, dim=1)
    second_features_norm = F.normalize(second_features, p=2, dim=1)
    cosine_loss = 1 - F.cosine_similarity(first_features_norm, second_features_norm).mean()
    return cosine_loss

def calculate_distribution_similarity_loss(logits_reshaped, labels_reshaped, mask_flat, model, device):
    """
    Calculate sentence-level distribution similarity loss using embeddings.
    
    Args:
        logits_reshaped: Model logits [batch_size * seq_len, vocab_size]
        labels_reshaped: Token labels [batch_size * seq_len]
        mask_flat: Attention mask [batch_size * seq_len]
        model: The model (for accessing vocab size)
        device: Current device
        
    Returns:
        Sentence-level distribution similarity loss
    """
    distribution_loss = torch.tensor(0.0, device=device)
    
    # Only compute on non-padded tokens (where mask is 1)
    valid_positions = mask_flat.bool()
    if valid_positions.sum() > 0:
        # Get the vocabulary size
        model_to_use = model.module if hasattr(model, 'module') else model
        
        # Extract valid logits and labels
        valid_logits = logits_reshaped[valid_positions]  # [valid_count, vocab_size]
        valid_labels = labels_reshaped[valid_positions]  # [valid_count]
        
        # Get softmax of logits (predicted distribution)
        probs = F.softmax(valid_logits, dim=-1)
        
        # Get embeddings for both predicted and target distributions
        with torch.no_grad():
            embedding_layer = model_to_use.t5_model.decoder.embed_tokens
            
            # Get embeddings for target tokens
            target_embeddings = embedding_layer(valid_labels)  # [valid_count, hidden_dim]
            
            # Get embeddings for predicted distribution
            # Weight each embedding by its probability
            all_token_embeddings = embedding_layer.weight  # [vocab_size, hidden_dim]
            predicted_embeddings = torch.matmul(probs, all_token_embeddings)  # [valid_count, hidden_dim]
        
        # Normalize embeddings
        target_embeddings = F.normalize(target_embeddings, p=2, dim=1)
        predicted_embeddings = F.normalize(predicted_embeddings, p=2, dim=1)
        
        # Calculate cosine similarity loss
        distribution_loss = (1 - F.cosine_similarity(
            predicted_embeddings,
            target_embeddings
        )).mean()
    
    return distribution_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, checkpoint_dir, config, teacher_model=None, start_epoch=0,
                best_val_loss=float('inf'), rank=None, logger=None, is_distributed=False,
                checkpoint_data=None):
    """Train the model with mixed precision training and gradient accumulation."""
    global TRAINING_FAILED

    save_frequency = config.training.checkpoint_frequency
    log_frequency = max(10, len(train_loader) // 3)

    # Training configuration
    use_amp = config.training.mixed_precision  # Should be False now
    
    if rank == 0:
        logger.info(f"🎯 Training Configuration:")
        logger.info(f"  Mixed precision: {'✅' if use_amp else '❌ DISABLED'}")
        logger.info(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
        logger.info(f"  Contrastive learning: {'✅' if config.training.use_contrastive_learning else '❌'}")
        
        # Note about darknet device mapping
        logger.info(f"📝 Note: Darknet weights loaded on CPU first to prevent OOM, then moved to GPU")
    
    # Initialize Exponential Moving Average
    ema = EMA(model, decay=0.999)
    
    # Load optimizer, scheduler, and EMA states if resuming from checkpoint
    if checkpoint_data is not None:
        try:
            if rank == 0:
                logger.info(f"🔄 Loading optimizer, scheduler, and EMA states from checkpoint...")
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                # Move optimizer tensors to correct device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
                if rank == 0:
                    logger.info(f"✅ Optimizer state loaded and moved to {device}")
            
            # Load scheduler state  
            if 'scheduler_state_dict' in checkpoint_data and checkpoint_data['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                if rank == 0:
                    logger.info(f"✅ Scheduler state loaded")
            
            # Load EMA state
            if 'ema' in checkpoint_data:
                ema.load_state_dict(checkpoint_data['ema'])
                # Move EMA shadow parameters to correct device
                for name, param in ema.shadow.items():
                    ema.shadow[name] = param.to(device)
                if rank == 0:
                    logger.info(f"✅ EMA state loaded and moved to {device}")
                    
            if rank == 0:
                logger.info(f"🎯 Complete checkpoint state restored for epoch {start_epoch + 1}")
                
        except Exception as e:
            if rank == 0:
                logger.warning(f"⚠️ Could not load optimizer/scheduler/EMA states: {e}")
                logger.info(f"📝 Continuing with fresh optimizer/scheduler/EMA states")
    
    # Initialize contrastive loss if enabled
    contrastive_loss_fn = None
    if config.training.use_contrastive_learning:
        contrastive_loss_fn = ContrastiveLoss(
            margin=config.training.contrastive_margin,
            temperature=config.training.contrastive_temperature,
            loss_type=config.training.contrastive_loss_type,
            use_cosine_distance=config.training.use_cosine_distance,
            mean_all=config.training.contrastive_mean_all
        )
        if rank == 0:
            distance_type = "cosine" if config.training.use_cosine_distance else "L2"
            mean_type = "all" if config.training.contrastive_mean_all else "non-zero"
            logger.info(f"🔗 Contrastive learning: {config.training.contrastive_loss_type} loss ({distance_type} distance, {mean_type} mean)")
    
    # Initialize smart weight scheduling based on user insights
    if rank == 0:
        logger.info(f"🧠 SMART 3-Phase Planning + Extended Training Strategy:")
        logger.info(f"📈 Strategy based on your experience:")
        logger.info(f"   • Previous overfitting at ~100 epochs (before contrastive)")
        logger.info(f"   • Self-predicting went to 2000+ epochs")
        logger.info(f"   • Plan 3 phases until epoch {config.training.planned_epochs}, then continue with fixed weights")
        logger.info(f"")
        logger.info(f"🎯 3-Phase Planning (0-{config.training.planned_epochs}):")
        logger.info(f"  Phase 1 (0-30%): Build semantic space - HIGH contrastive, LOW CE")
        logger.info(f"  Phase 2 (30-70%): Balance learning - MODERATE both")
        logger.info(f"  Phase 3 (70-100%): Fine-tune with CE - LOW contrastive, HIGH CE")
        logger.info(f"")
        logger.info(f"🚀 Extended Training ({config.training.planned_epochs}-{num_epochs}):")
        logger.info(f"  • Fixed weights at mature values (CE=1.5, Contrastive=1.0)")
        logger.info(f"  • Adaptive contrastive revival still active (auto-boost when signal dies)")
        logger.info(f"  • Early stopping determines actual end ({config.training.early_stopping_patience} epochs patience)")
        logger.info(f"💾 Checkpoints every 100 epochs (less frequent for long training)")
        
        # Show the user the weight schedule visualization
        visualize_weight_schedule(config.training.planned_epochs, config.training.num_epochs, logger)
    
    # Get smart scheduling functions
    contrastive_weight_fn, ce_weight_fn = get_smart_contrastive_schedule(config.training.planned_epochs, config.training.num_epochs)
    curriculum_ratio_fn = get_smart_curriculum_schedule(config.training.planned_epochs)
    
    # Get adaptive contrastive scheduler with revival capability
    adaptive_contrastive_fn = get_adaptive_contrastive_schedule(contrastive_weight_fn, revival_threshold=0.002)  # Balanced threshold for current phase
    
    # Traditional schedulers for other losses
    destination_weight_fn = get_smart_destination_schedule(config.training.planned_epochs)
    kd_weight_fn = get_smart_kd_schedule(config.training.planned_epochs)
    
    # Clear cache once at beginning
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Keep track of the last best model's epoch
    last_best_epoch = None
    
    # Early stopping variables
    early_stopping_counter = 0
    early_stopping_triggered = False
        
    try:
        for epoch in range(start_epoch, num_epochs):
            # Check if early stopping was triggered
            if early_stopping_triggered:
                if rank == 0:
                    logger.info(f"🛑 Early stopping triggered after {epoch} epochs")
                
                # Add synchronization barrier to ensure all processes exit together
                if is_distributed and dist.is_initialized():
                    try:
                        dist.barrier()
                        if rank == 0:
                            logger.info("✅ All processes synchronized before exit")
                    except Exception as e:
                        if rank == 0:
                            logger.error(f"❌ Error during synchronization: {e}")
                
                break
                
            if is_distributed:
                train_loader.sampler.set_epoch(epoch)
                val_loader.sampler.set_epoch(epoch)

            model.train()
            total_loss = 0
            total_ce_loss = 0
            total_destination_loss = 0
            total_contrastive_loss = 0
            total_kd_loss = 0
            optimizer.zero_grad(set_to_none=True)

            epoch_start_time = time.time()

            # Only rank 0 logs the results
            if rank == 0:
                # Check for phase transitions and announce them
                phase1_end = int(0.3 * config.training.planned_epochs)
                phase2_end = int(0.7 * config.training.planned_epochs)
                
                if epoch == phase1_end:
                    logger.info("🔄 PHASE TRANSITION: Entering Phase 2 - Balance Learning!")
                    logger.info("  📈 CE weight increasing, Contrastive weight decreasing")
                    logger.info("  🎯 Curriculum ratio transitioning (0.7 → 0.3)")
                elif epoch == phase2_end:
                    logger.info("🔄 PHASE TRANSITION: Entering Phase 3 - CE Fine-tuning!")
                    logger.info("  🎯 Focus on cross-entropy loss for final optimization")
                    logger.info("  🚀 Curriculum diminishing (0.3 → 0.0) - Independent navigation!")
                elif epoch == config.training.planned_epochs:
                    logger.info("🔄 ENTERING EXTENDED PHASE: Fixed Weights Training!")
                    logger.info("  ⚖️ Weights fixed at mature values: CE=1.5, Contrastive=1.0")
                    logger.info("  🔥 Adaptive contrastive revival still active!")
                    logger.info("  🛑 Training continues until early stopping triggers")
                
                logger.info(f"🚀 Epoch {epoch + 1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Simple direct data loading - no prefetching
                    text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
                    
                    # Add separate components if available (for hierarchical processing)
                    if 'first_instruction_input' in batch:
                        text_input['first_instruction_input'] = {k: v.to(device, non_blocking=True) for k, v in batch['first_instruction_input'].items()}
                    if 'current_question_input' in batch:
                        text_input['current_question_input'] = {k: v.to(device, non_blocking=True) for k, v in batch['current_question_input'].items()}
                    
                    current_view = batch['current_view_image'].to(device, non_blocking=True)
                    previous_views = batch['previous_views_image'].to(device, non_blocking=True)
                    
                    # Handle text_label as a dictionary with input_ids and attention_mask
                    label_input_ids = batch['text_label']['input_ids'].to(device, non_blocking=True)
                    label_attention_mask = batch['text_label']['attention_mask'].to(device, non_blocking=True)
                    
                    # Set up destination view if available in batch and curriculum is active
                    destination_view = batch['destination_image'].to(device, non_blocking=True) if 'destination_image' in batch else None
                    
                    # Calculate curriculum learning ratio using smart schedule
                    curriculum_ratio = curriculum_ratio_fn(epoch)
                    
                    # Prepare contrastive examples if enabled
                    positive_input = None
                    positive_input_2 = None
                    negative_input = None
                    negative_input_2 = None
                    contrastive_examples_found = False
                    
                    if config.training.use_contrastive_learning and contrastive_loss_fn is not None:
                        # Get tokenized contrastive inputs from dataset (normalizer always provides these)
                        if 'positive_input' in batch:
                            positive_input = {k: v.to(device, non_blocking=True) for k, v in batch['positive_input'].items()}
                            contrastive_examples_found = True
                            
                            if 'positive_input_2' in batch:
                                positive_input_2 = {k: v.to(device, non_blocking=True) for k, v in batch['positive_input_2'].items()}
                            
                            if 'negative_input' in batch:
                                negative_input = {k: v.to(device, non_blocking=True) for k, v in batch['negative_input'].items()}
                                
                            if 'negative_input_2' in batch:
                                negative_input_2 = {k: v.to(device, non_blocking=True) for k, v in batch['negative_input_2'].items()}
                                   
                    
                    # Forward pass
                    outputs = model(
                        text_input, 
                        current_view, 
                        previous_views, 
                        labels=label_input_ids,
                        destination_view=destination_view,
                        curriculum_ratio=curriculum_ratio,
                        positive_input=positive_input,
                        positive_input_2=positive_input_2,
                        negative_input=negative_input,
                        negative_input_2=negative_input_2
                    )
                    
                    logits = outputs["logits"]
                    # Use the loss already calculated by T5 instead of recalculating
                    ce_loss = outputs.get("loss", torch.tensor(0.0, device=device))
                    feature_norm = outputs.get("feature_norm", torch.tensor(0.0, device=device))


                    if torch.isnan(logits).any():
                        if rank == 0:
                            logger.error(f"❌ NaN detected in logits at batch {batch_idx}")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    # No need to recalculate CE loss - T5 already did it!
                    # logits_reshaped = logits.contiguous().view(batch_size * seq_len, vocab_size)
                    # labels_reshaped = label_input_ids.contiguous().view(batch_size * seq_len)
                    # ce_loss = criterion(logits_reshaped, labels_reshaped)  # REMOVED: redundant calculation
                    
                    ce_loss_weight = ce_weight_fn(epoch)

                    # Add feature regularization with clipping to prevent explosion
                    feature_norm_clipped = feature_norm.clamp(max=1e3)  # Clip to prevent explosion
                    reg_loss = 1e-4 * feature_norm_clipped
                    loss = ce_loss_weight * ce_loss + reg_loss
                        
                    # Add destination loss if destination view is available
                    if destination_view is not None:
                        dest_features = outputs.get("destination_features", outputs["raw_adapted_features"])
                        # Use raw_adapted_features for destination loss - these are the pure T5 adapter outputs
                        # before contrastive projection, representing the model's learned visual-text alignment
                        destination_cosine_loss = calculate_cosine_similarity_loss(outputs["raw_adapted_features"], dest_features)
                        
                        destination_weight = destination_weight_fn(epoch)
                        loss = loss + destination_weight * destination_cosine_loss
                    else:
                        destination_cosine_loss = torch.tensor(0.0, device=device)

                    
                    # Calculate contrastive loss if enabled
                    contrastive_loss = torch.tensor(0.0, device=device)
                    if config.training.use_contrastive_learning and contrastive_loss_fn is not None:
                        
                        # Collect all positive and negative embeddings
                        anchor_emb = None
                        positive_embs = []
                        negative_emb = None
                        
                        if 'positive_adapted_features' in outputs and 'negative_adapted_features' in outputs:
                            anchor_emb = outputs['adapted_features']
                            positive_embs.append(outputs['positive_adapted_features'])
                            
                            # Gather negatives
                            negatives_embs = []
                            if 'negative_adapted_features' in outputs:
                                negatives_embs.append(outputs['negative_adapted_features'])
                            if 'negative_adapted_features_2' in outputs:
                                negatives_embs.append(outputs['negative_adapted_features_2'])
                            
                            # Add second positive if available
                            if 'positive_adapted_features_2' in outputs:
                                positive_embs.append(outputs['positive_adapted_features_2'])
                            
                            # Stack positives if we have multiple
                            if len(positive_embs) > 1:
                                positive_combined = torch.stack(positive_embs, dim=1)  # [batch, num_pos, hidden]
                            else:
                                positive_combined = positive_embs[0]  # [batch, hidden]
                            
                            # Compute contrastive loss for each negative and average
                            contrastive_losses_list = []
                            if not negatives_embs:
                                # Fall back to in-batch negatives if none provided
                                contrastive_losses_list.append(
                                    contrastive_loss_fn(anchor_emb, positive_combined, None)
                                )
                            else:
                                for neg_emb in negatives_embs:
                                    contrastive_losses_list.append(
                                        contrastive_loss_fn(anchor_emb, positive_combined, neg_emb)
                                    )
                            
                            contrastive_loss = torch.stack(contrastive_losses_list).mean()
                            
                            
                            # Add weighted contrastive loss to total loss
                            contrastive_weight, revival_info = adaptive_contrastive_fn(epoch)
                            loss = loss + contrastive_weight * contrastive_loss
                            total_contrastive_loss += contrastive_loss.item()
                        elif rank == 0 and batch_idx == 0 and epoch == 0:
                            logger.warning(f"⚠️ Contrastive learning enabled but no adapted features found in outputs!")
                            logger.info(f"🔍 Available output keys: {list(outputs.keys())}")
                
                    # KD loss using embeddings generated during preprocessing
                    kd_loss = torch.tensor(0.0, device=device)
                    if config.training.use_kd:
                        # Teacher embeddings are included in the batch from preprocessing
                        teacher_embeddings = batch['teacher_embed'].to(device)
                        
                        # Use raw_adapted_features for KD - these are the pure T5 adapter outputs before contrastive projection
                        # This aligns with the teacher's general-purpose embeddings for better knowledge transfer
                        student_hidden = F.normalize(outputs["raw_adapted_features"], p=2, dim=-1)
                        teacher_hidden = F.normalize(teacher_embeddings, p=2, dim=-1)
                        
                        kd_loss = 1 - F.cosine_similarity(student_hidden, teacher_hidden, dim=-1).mean()
                        
                        kd_weight = kd_weight_fn(epoch)
                        loss = loss + kd_weight * kd_loss
                        total_kd_loss += kd_loss.item()

                    # Apply gradient accumulation: normalize loss
                    loss = loss / config.training.gradient_accumulation_steps

                    # Accumulate statistics
                    total_ce_loss += ce_loss.item()
                    total_destination_loss += destination_cosine_loss.item()
                    
                    # Backpropagation
                    loss.backward()

                    # Gradient accumulation
                    if ((batch_idx + 1) % config.training.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                                
                        # Update parameters
                        optimizer.step()
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()      # release cached kernels
                            torch.cuda.ipc_collect()      # C++ side arena defrag

                        optimizer.zero_grad(set_to_none=True)
                        
                        # Update EMA
                        ema.update()

                    total_loss += loss.item() * config.training.gradient_accumulation_steps
                
                    # Only have rank 0 log progress
                    if rank == 0 and batch_idx % log_frequency == 0:
                        avg_loss = total_loss / (batch_idx + 1)
                        avg_ce = total_ce_loss / (batch_idx + 1)
                        avg_contrastive = total_contrastive_loss / (batch_idx + 1)
                        avg_destination = total_destination_loss / (batch_idx + 1)
                        avg_kd = total_kd_loss / (batch_idx + 1)
                        
                        
                        logger.info(f"📊 Batch {batch_idx}/{len(train_loader)} | "
                                  f"Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | "
                                  f"Contrast: {avg_contrastive:.4f} | KD: {avg_kd:.4f} | Dest: {avg_destination:.4f}")
                        
                        
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    # Log and continue in case of batch failure
                    if rank == 0:
                        logger.error(f"❌ Error in batch {batch_idx}: {str(e)}")
                        logger.error(f"📍 Full traceback: {traceback.format_exc()}")
                        logger.error(f"💾 Memory usage: {log_gpu_memory()}")
                    
                    # Zero out gradients to avoid accumulation
                    optimizer.zero_grad(set_to_none=True)
                    continue

            # Calculate average losses across distributed processes
            if is_distributed:
                # Gather losses from all processes
                loss_tensor = torch.tensor(total_loss, device=device)
                total_ce_loss_tensor = torch.tensor(total_ce_loss, device=device)   
                total_destination_loss_tensor = torch.tensor(total_destination_loss, device=device)
                total_contrastive_loss_tensor = torch.tensor(total_contrastive_loss, device=device)
                total_kd_loss_tensor = torch.tensor(total_kd_loss, device=device)
                
                # All-reduce to sum losses across processes
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_ce_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_destination_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_contrastive_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_kd_loss_tensor, op=dist.ReduceOp.SUM)

                
                # Calculate averages
                total_loss = loss_tensor.item() / dist.get_world_size()
                total_ce_loss = total_ce_loss_tensor.item() / dist.get_world_size()
                total_destination_loss = total_destination_loss_tensor.item() / dist.get_world_size()
                total_contrastive_loss = total_contrastive_loss_tensor.item() / dist.get_world_size()
                total_kd_loss = total_kd_loss_tensor.item() / dist.get_world_size()
            
            avg_epoch_loss = total_loss / len(train_loader)
            avg_ce_loss = total_ce_loss / len(train_loader)
            avg_destination_loss = total_destination_loss / len(train_loader)
            avg_contrastive_loss = total_contrastive_loss / len(train_loader)
            avg_kd_loss = total_kd_loss / len(train_loader)
            
            # Log the epoch summary (only rank 0)
            if rank == 0:
                epoch_time = time.time() - epoch_start_time
                
                # Calculate current weights for this epoch
                if hasattr(config.training, 'log_loss_weights') and config.training.log_loss_weights:
                    current_ce_weight = ce_weight_fn(epoch)
                    current_contrastive_weight, revival_info = adaptive_contrastive_fn(epoch, recent_contrastive_loss=avg_contrastive_loss)
                    current_dest_weight = destination_weight_fn(epoch)
                    current_curriculum_ratio = curriculum_ratio_fn(epoch)
                    current_kd_weight = kd_weight_fn(epoch)

                    
                    logger.info(f"✅ Epoch {epoch+1} | Loss: {avg_epoch_loss:.4f} | "
                              f"CE: {avg_ce_loss:.4f} | "
                              f"Contrast: {avg_contrastive_loss:.4f} | Dest: {avg_destination_loss:.4f} | KD: {avg_kd_loss:.4f} "
                              f"Time: {epoch_time:.1f}s")
                    logger.info(f"🎛️  Weights | CE: {current_ce_weight:.2f} | "
                              f"Contrastive: {current_contrastive_weight:.2f} | Dest: {current_dest_weight:.2f} | "
                              f"Curriculum: {current_curriculum_ratio:.2f} | KD: {current_kd_weight:.2f}")
                    
                    # Log revival information if triggered
                    if revival_info['triggered'] and revival_info.get('is_new_revival', False):
                        logger.info(f"🔥 GENTLE CONTRASTIVE REVIVAL at epoch {epoch+1}!")
                        logger.info(f"   📊 Target loss: {revival_info['target_loss']:.4f} | Current avg: {revival_info['recent_avg']:.4f}")
                        logger.info(f"   ⚖️  Base weight: {revival_info['base_weight']:.2f} → Revival weight: {revival_info['revival_weight']:.2f}")
                    
                    # Log effective contributions for debugging
                    effective_ce = avg_ce_loss * current_ce_weight
                    effective_contrastive = avg_contrastive_loss * current_contrastive_weight
                    effective_dest = avg_destination_loss * current_dest_weight
                    effective_kd = avg_kd_loss * current_kd_weight
                    effective_total = effective_ce + effective_contrastive + effective_dest + effective_kd
                    
                    logger.info(f"🔍 Effective | Total Loss: {effective_total:.4f} | CE: {effective_ce:.4f} | "
                              f"Contrastive: {effective_contrastive:.4f} | Dest: {effective_dest:.4f} | KD: {effective_kd:.4f} ")
                else:
                    logger.info(f"✅ Epoch {epoch+1} | Loss: {avg_epoch_loss:.4f} | "
                              f"CE: {avg_ce_loss:.4f} | "
                              f"Contrast: {avg_contrastive_loss:.4f} | Dest: {avg_destination_loss:.4f} | KD: {avg_kd_loss:.4f}"
                              f"Time: {epoch_time:.1f}s")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()          # throw away the giant arena
                torch.cuda.reset_peak_memory_stats()
                
            # Validation step
            val_loss = 0
                
            if (epoch + 1) % config.training.eval_freq == 0 or epoch == num_epochs - 1:
                model.eval()
                
                # Apply EMA for validation
                ema.apply_shadow()

                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        try:
                            # Load validation data
                            text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
                            
                            # Add separate components if available (for hierarchical processing)
                            if 'first_instruction_input' in batch:
                                text_input['first_instruction_input'] = {k: v.to(device, non_blocking=True) for k, v in batch['first_instruction_input'].items()}
                            if 'current_question_input' in batch:
                                text_input['current_question_input'] = {k: v.to(device, non_blocking=True) for k, v in batch['current_question_input'].items()}
                            
                            current_view = batch['current_view_image'].to(device, non_blocking=True)
                            previous_views = batch['previous_views_image'].to(device, non_blocking=True)
                            
                            label_input_ids = batch['text_label']['input_ids'].to(device, non_blocking=True)
                            label_attention_mask = batch['text_label']['attention_mask'].to(device, non_blocking=True)
                            
                            # Set up destination view if available
                            destination_view = batch['destination_image'].to(device, non_blocking=True) if 'destination_image' in batch else None
                            
                            # Prepare contrastive examples if available
                            positive_input = None
                            positive_input_2 = None
                            negative_input = None
                            negative_input_2 = None
                            
                            # Check for new format (both tokenized and raw text) vs old format
                            if 'positive_input' in batch:
                                # New format: tokenized inputs from normalizer
                                positive_input = {k: v.to(device, non_blocking=True) for k, v in batch['positive_input'].items()}
                                
                                if 'positive_input_2' in batch:
                                    positive_input_2 = {k: v.to(device, non_blocking=True) for k, v in batch['positive_input_2'].items()}
                                
                                if 'negative_input' in batch:
                                    negative_input = {k: v.to(device, non_blocking=True) for k, v in batch['negative_input'].items()}
                                    
                                if 'negative_input_2' in batch:
                                    negative_input_2 = {k: v.to(device, non_blocking=True) for k, v in batch['negative_input_2'].items()}
                                    
                        
                            # Use mixed precision for validation as well for consistent numerical behavior
                            outputs = model(
                                text_input, 
                                current_view, 
                                previous_views, 
                                labels=label_input_ids,
                                destination_view=destination_view,
                                curriculum_ratio=0.0,  # No curriculum during validation - pure model performance
                                positive_input=positive_input,
                                positive_input_2=positive_input_2,
                                negative_input=negative_input,
                                negative_input_2=negative_input_2
                            )
                            
                            logits = outputs["logits"]
                            # Use the loss already calculated by T5 instead of recalculating in validation too
                            ce_loss = outputs.get("loss", torch.tensor(0.0, device=device))
                            
                            # No need to manually calculate CE loss in validation either - T5 already did it!
                            # logits_reshaped = logits.contiguous().view(batch_size * seq_len, vocab_size)
                            # labels_reshaped = label_input_ids.contiguous().view(batch_size * seq_len)
                            # ce_loss = criterion(logits_reshaped, labels_reshaped)  # REMOVED: redundant calculation
                            
                            # Calculate validation loss
                            loss = ce_weight_fn(epoch) * ce_loss
                            
                            
                            # Add contrastive loss if enabled
                            if config.training.use_contrastive_learning and contrastive_loss_fn is not None:
                                contrastive_losses = []
                                
                                # First triplet: anchor, positive1, negative
                                if 'positive_adapted_features' in outputs and 'negative_adapted_features' in outputs:
                                    anchor_emb = outputs['adapted_features']
                                    positive_emb = outputs['positive_adapted_features']
                                    negatives_val = [outputs['negative_adapted_features']]
                                    if 'negative_adapted_features_2' in outputs:
                                        negatives_val.append(outputs['negative_adapted_features_2'])
                                    
                                    # Add shape validation for each negative
                                    for neg_emb in negatives_val:
                                        if anchor_emb.shape != positive_emb.shape or anchor_emb.shape != neg_emb.shape:
                                            logger.error(f"❌ Shape mismatch in validation contrastive loss: anchor={anchor_emb.shape}, "
                                                       f"positive={positive_emb.shape}, negative={neg_emb.shape}")
                                            continue
                                        contrastive_losses.append(
                                            contrastive_loss_fn(anchor_emb, positive_emb, neg_emb)
                                        )
                                
                                # Second triplet: anchor, positive2, negative (if available)
                                if 'positive_adapted_features_2' in outputs and 'negative_adapted_features' in outputs:
                                    anchor_emb = outputs['adapted_features']
                                    positive_emb_2 = outputs['positive_adapted_features_2']
                                    negative_emb = outputs['negative_adapted_features']
                                    
                                    # Add shape validation
                                    if anchor_emb.shape != positive_emb_2.shape or anchor_emb.shape != negative_emb.shape:
                                        logger.error(f"❌ Shape mismatch in validation contrastive loss (triplet 2): anchor={anchor_emb.shape}, "
                                                   f"positive_2={positive_emb_2.shape}, negative={negative_emb.shape}")
                                        continue
                                    
                                    contrastive_loss_2 = contrastive_loss_fn(anchor_emb, positive_emb_2, negative_emb)
                                    contrastive_losses.append(contrastive_loss_2)
                                    
                                # Average the contrastive losses and add to validation loss
                                if contrastive_losses:
                                    contrastive_loss = torch.stack(contrastive_losses).mean()
                                    # Use the same adaptive weight function for validation consistency
                                    val_contrastive_weight, _ = adaptive_contrastive_fn(epoch)
                                    loss = loss + val_contrastive_weight * contrastive_loss
                            
                            val_loss += loss.item()
                        except Exception as e:
                            if rank == 0:
                                logger.error(f"❌ Error in validation batch {batch_idx}: {str(e)}")
                            continue
                
                # Restore original weights
                ema.restore()

                # Average validation loss across all processes if distributed
                if is_distributed:
                    val_loss_tensor = torch.tensor(val_loss, device=device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    val_loss = val_loss_tensor.item() / dist.get_world_size()
                
                # Calculate average validation loss
                val_loss = val_loss / len(val_loader)

                if rank == 0:
                    logger.info(f"📋 Validation Loss: {val_loss:.4f}")
                    
                    # Check if this is the best model so far (only compare to best, not previous)
                    if val_loss < best_val_loss * (1 - config.training.early_stopping_min_delta):
                        improvement = (best_val_loss - val_loss) / best_val_loss * 100
                        logger.info(f"🎯 New best model! Improved by {improvement:.2f}% | Previous best: {best_val_loss:.4f} | New best: {val_loss:.4f}")
                        
                        # Update the tracking variable BEFORE saving so that the file records the correct best value
                        best_val_loss = val_loss

                        with torch.cuda.amp.autocast(enabled=False):
                            # Save best model with the *updated* best_val_loss
                            save_dict_fp32 = {
                                'epoch': epoch,
                                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                                'ema': ema.state_dict(),
                                'val_loss': val_loss,
                                'best_val_loss': best_val_loss,
                                'config': config,
                            }

                        with CHECKPOINT_LOCK:
                            best_model_path = os.path.join(checkpoint_dir, f'best_model_{epoch+1}_fp32.pth')
                            torch.save(save_dict_fp32, best_model_path)
                            logger.info("💾 Saved best model (updated best_val_loss)")

                        # Free memory after saving
                        del save_dict_fp32
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        last_best_epoch = epoch
                        
                        # Reset early stopping counter on improvement
                        early_stopping_counter = 0
                    else:
                        # Only increment counter if no improvement (don't compare to previous loss)
                        logger.info(f"🔍 previous best: {best_val_loss:.4f} | current loss: {val_loss:.4f}")
                        early_stopping_counter += 1
                        logger.info(f"🔍 Early stopping counter: {early_stopping_counter}/{config.training.early_stopping_patience}")
                        if config.training.early_stopping and early_stopping_counter >= config.training.early_stopping_patience:
                            early_stopping_triggered = True
                            logger.info(f"🛑 Early stopping triggered after {early_stopping_counter} epochs without improvement")
                            break
                    
            
            # Save checkpoint at regular intervals (only rank 0)
            if rank == 0 and (epoch + 1) % save_frequency == 0:
                with torch.cuda.amp.autocast(enabled=False):
                    save_dict_fp32 = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'ema': ema.state_dict(),
                    'val_loss': val_loss if 'val_loss' in locals() else None,
                    'best_val_loss': best_val_loss,  # Save the best validation loss achieved so far
                    'config': config,
                }
                
                with CHECKPOINT_LOCK:
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}_fp32.pth')
                    torch.save(save_dict_fp32, checkpoint_path)
                    logger.info(f"💾 Checkpoint saved")
                
                # Free memory after saving
                del save_dict_fp32
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Step the scheduler based on validation loss if available
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    # ReduceLROnPlateau needs the validation loss
                    if 'val_loss' in locals():
                        scheduler.step(val_loss)
                else:
                    scheduler.step()
        
        # End of training - save final model
        if rank == 0:
            with torch.cuda.amp.autocast(enabled=False):
                save_dict_fp32 = {
                'epoch': num_epochs - 1,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'ema': ema.state_dict(),
                'val_loss': val_loss if 'val_loss' in locals() else None,
                'best_val_loss': best_val_loss,  # Save the best validation loss achieved so far
                'config': config,
            }
            
            with CHECKPOINT_LOCK:
                final_model_path = os.path.join(checkpoint_dir, f'final_model_{epoch+1}_fp32.pth')
                torch.save(save_dict_fp32, final_model_path)
                logger.info(f"💾 Final model saved")
            
            # Free memory after saving
            del save_dict_fp32
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Print training summary
            if last_best_epoch is not None:
                logger.info(f"🎉 Training complete! Best val loss: {best_val_loss:.4f} at epoch {last_best_epoch + 1}")
            else:
                logger.info(f"🎉 Training complete! Best val loss: {best_val_loss:.4f} (no improvement during training)")
                
        return best_val_loss, last_best_epoch

    except Exception as e:
        if rank == 0:
            logger.error(f"❌ Training failed: {str(e)}")
        
        TRAINING_FAILED = True
        raise

def log_gpu_memory():
    """Log GPU memory usage (simplified)."""
    if not torch.cuda.is_available():
        return "No CUDA"
        
    try:
        current_device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024 ** 2
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024 ** 2
        
        return f"GPU {current_device}: {memory_allocated:.0f}MB/{memory_reserved:.0f}MB"
    except Exception:
        return "GPU memory error"

def setup_distributed():
    """Set up the distributed environment using environment variables set by torchrun."""
    # Get distributed training environment variables from torchrun
    if "LOCAL_RANK" not in os.environ:
        # Not running with torchrun, assume single-GPU
        return False, 0, 1
    
    # Check available GPU count
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("⚠️ No CUDA devices available! Running on CPU only.")
        return False, 0, 1
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Ensure local_rank is within the range of available devices
    if local_rank >= available_gpus:
        print(f"⚠️ local_rank ({local_rank}) >= available GPUs ({available_gpus})")
        local_rank = local_rank % available_gpus
    
    # Set the device
    torch.cuda.set_device(local_rank)
    
    # Ensure proper initialization of master address and port
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    
    # Initialize the process group
    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(minutes=30)
            )
            print(f"✅ Process group initialized for rank {rank}")
        except Exception as e:
            print(f"❌ Error initializing process group: {e}")
            raise
    
    return True, rank, world_size

def cleanup():
    """Force cleanup of resources."""
    if dist.is_initialized():
        dist.destroy_process_group()
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def visualize_weight_schedule(planned_epochs: int, max_epochs: int, logger=None):
    """
    Visualize the complete 3-phase weight scheduling to help user understand the plan.
    """
    contrastive_weight_fn, ce_weight_fn = get_smart_contrastive_schedule(planned_epochs, max_epochs)
    curriculum_ratio_fn = get_smart_curriculum_schedule(planned_epochs)
    destination_weight_fn = get_smart_destination_schedule(planned_epochs)
    kd_weight_fn = get_smart_kd_schedule(planned_epochs)
    
    if logger:
        logger.info("📊 COMPLETE TRAINING SCHEDULE VISUALIZATION:")
        logger.info(f"🎯 3-Phase Planning: {planned_epochs} epochs | Max Training: {max_epochs} epochs")
        logger.info("=" * 100)
        
        # Show key epochs from planning phase + extended phase
        phase1_end = int(0.3 * planned_epochs)
        phase2_end = int(0.7 * planned_epochs)
        
        epochs_to_show = [0, phase1_end//2, phase1_end, (phase1_end + phase2_end)//2, 
                         phase2_end, (phase2_end + planned_epochs)//2, planned_epochs-1, 
                         planned_epochs + 100, max_epochs-1]
        
        logger.info(f"{'Epoch':>6} | {'Phase':^30} | {'CE':>5} | {'Contrast':>8} | {'Curriculum':>10} | {'Dest':>6} | {'KD':>5}")
        logger.info("-" * 100)
        
        for epoch in epochs_to_show:
            ce_w = ce_weight_fn(epoch)
            cont_w = contrastive_weight_fn(epoch)
            curr_r = curriculum_ratio_fn(epoch)
            dest_w = destination_weight_fn(epoch)
            kd_w = kd_weight_fn(epoch)
            
            if epoch < phase1_end:
                phase = "PHASE 1: Build Space"
            elif epoch < phase2_end:
                phase = "PHASE 2: Balance"
            elif epoch < planned_epochs:
                phase = "PHASE 3: CE Fine-tune"
            else:
                phase = "EXTENDED: Fixed Weights"
                
            logger.info(f"{epoch:6d} | {phase:^30} | {ce_w:5.2f} | {cont_w:8.2f} | {curr_r:10.2f} | {dest_w:6.2f} | {kd_w:5.2f}")
        
        logger.info("=" * 100)
        logger.info("🎯 INTEGRATED STRATEGY - ALL COMPONENTS:")
        logger.info(f"  ✅ Phase 1 (0-{phase1_end}): Build Semantic Space")
        logger.info("      • HIGH contrastive (15.0) + LOW CE (0.1) + HIGH curriculum (1.0→0.7)")
        logger.info("      • HIGH destination (0.8) + MODERATE KD (0.5)")
        logger.info("      → Oracle helps build visual-text alignment with teacher guidance")
        logger.info(f"  ✅ Phase 2 ({phase1_end}-{phase2_end}): Balance Learning")
        logger.info("      • MODERATE contrastive (15.0→9.0) + MODERATE CE (0.1→0.8) + MODERATE curriculum (0.7→0.3)")
        logger.info("      • MODERATE destination (0.8→0.3) + HIGH KD (0.5→1.0)")
        logger.info("      → Balanced transition with strong teacher guidance")
        logger.info(f"  ✅ Phase 3 ({phase2_end}-{planned_epochs}): CE Fine-tuning")
        logger.info("      • PRESERVED contrastive (9.0→5.0) + HIGH CE (0.8→1.5) + NO curriculum (0.3→0.0)")
        logger.info("      • LOW destination (0.3→0.05) + LOW KD (1.0→0.1)")
        logger.info("      → Independent navigation with PRESERVED contrastive signal")
        logger.info(f"  🚀 Extended ({planned_epochs}-{max_epochs}): Fixed Weights + Adaptive Revival")
        logger.info("      • FIXED: CE=1.5, Contrastive=5.0, Curriculum=0.0, Destination=0.05, KD=0.1")
        logger.info("      → Continue with STRONG contrastive signal until early stopping")
        logger.info("  🔥 GENTLE contrastive revival active throughout!")
        logger.info("    → Target-based revival: <0.002 → 0.004-0.005 (prevents late-epoch noise)")
        logger.info("    → Conservative bounds: Early training (2x max) → Late training (1.2x max)")
        logger.info("=" * 100)

def main():
    """Main training function that works with torchrun."""
    global TRAINING_FAILED
    
    # Setup minimal environment (no excessive logging)
    setup_minimal_environment()
    
    # Parse arguments first
    parser = argparse.ArgumentParser(description='UAV Navigation Training')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training from', default=None)
    parser.add_argument('--single-gpu', action='store_true', help='Force running on a single GPU even with torchrun')
    parser.add_argument('--batch-size', type=int, help='Per-GPU batch size (overrides config value)', default=None)
    parser.add_argument('--grad-steps', type=int, help='Gradient accumulation steps (overrides config value)', default=None)
    parser.add_argument('--use-augmented-data', action='store_true', help='Use augmented dataset with paraphrases for contrastive learning')
    parser.add_argument('--no-augmented-data', action='store_true', help='Use original dataset without augmented paraphrases')
    args = parser.parse_args()
    
    # Check CUDA availability first
    if not torch.cuda.is_available():
        print("⚠️ CUDA is not available! Training will run on CPU only.")
        num_gpus = 0
    else:
        num_gpus = torch.cuda.device_count()
    
    if args.single_gpu or num_gpus <= 1:
        # Single GPU or CPU mode
        print("🖥️ Running in single device mode")
        is_distributed = False
        rank = 0
        world_size = 1
        local_rank = 0
        
        if num_gpus > 0:
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
        else:
            device = torch.device('cpu')
    else:
        # Multi-GPU mode with torchrun
        try:
            is_distributed, rank, world_size = setup_distributed()
            local_rank = int(os.environ.get("LOCAL_RANK", 0)) % max(1, torch.cuda.device_count())
            device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            print(f"❌ Error setting up distributed environment: {e}")
            print("🔄 Falling back to single GPU mode")
            is_distributed = False
            rank = 0
            world_size = 1
            local_rank = 0
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Log essential system info
    log_system_info(rank)
    
    # Set up signal handlers for proper cleanup
    def signal_handler(sig, frame):
        print(f"🛑 Process {rank} received signal {sig}, cleaning up...")
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    config = Config()
    
    # Set seed for reproducibility
    set_seed(config.training.seed)
    if rank == 0:
        print(f"🎲 Random seed: {config.training.seed}")

    # Initialize logger
    logger = setup_logger('training', log_dir=config.log_dir)

    # Override config values with command-line arguments if provided
    if args.batch_size is not None:
        config.training.per_gpu_batch_size = args.batch_size
        if rank == 0:
            logger.info(f"⚙️ Batch size override: {args.batch_size}")
            
    if args.grad_steps is not None:
        config.training.gradient_accumulation_steps = args.grad_steps
        if rank == 0:
            logger.info(f"⚙️ Gradient accumulation override: {args.grad_steps}")
    
    # Handle augmented data arguments
    if args.no_augmented_data:
        config.data.use_augmented_data = False
        if rank == 0:
            logger.info("📊 Using original dataset (no augmentation)")
    elif args.use_augmented_data:
        config.data.use_augmented_data = True
        if rank == 0:
            logger.info("📊 Using augmented dataset with paraphrases")
    
    # Log dataset configuration
    if rank == 0:
        status = "✅ enabled" if config.data.use_augmented_data else "❌ disabled"
        logger.info(f"📊 Augmented data: {status}")
        logger.info(f"🔗 Contrastive learning: {'✅' if config.training.use_contrastive_learning else '❌'}")
    
    # Silence non-rank-0 processes by setting logger level
    if rank != 0:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.ERROR)  # Only show errors on non-rank-0
    
    if rank == 0:
        logger.info(f"🚀 Starting UAV Navigation Training")
        logger.info(f"📍 Device: {device} | Rank: {rank} | World Size: {world_size}")

    try:
        # Initialize tokenizer
        tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, model_max_length=config.data.max_seq_length)
        
        if rank == 0:
            logger.info(f"🎯 Training on {max(1, num_gpus)} GPU(s), distributed: {is_distributed}")
        
        # Wait for rank 0 to finish preprocessing
        if is_distributed:
            dist.barrier()
        
        if rank == 0:
            logger.info("🏗️ Initializing model...")
            
        # Initialize model and move to correct GPU
        model = AnsweringAgent(config, tokenizer, logger)
        if is_distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)

        # Teacher embeddings are now generated during preprocessing in the normalizer
        # No need to load teacher model during training
        teacher_model = None
        if config.training.use_kd and rank == 0:
            logger.info(f"🧑‍🏫 Teacher embeddings generated during preprocessing for KD")

        # Initialize training variables
        start_epoch = 0
        best_val_loss = float('inf')

        # Wrap model with DDP if using distributed training
        if is_distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                broadcast_buffers=True
            )

        # Resume training if checkpoint is provided
        checkpoint_data = None
        if args.checkpoint and os.path.exists(args.checkpoint):
            if rank == 0:
                logger.info(f"📂 Loading checkpoint: {args.checkpoint}")
                
            # Load checkpoint on CPU first to avoid GPU memory issues
            try:
                checkpoint_data = torch.load(args.checkpoint, map_location='cpu')
                
                # Move model state dict to GPU after loading
                if hasattr(model, 'module'):
                    model.module.load_state_dict(checkpoint_data['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint_data['model_state_dict'])
                    
                start_epoch = checkpoint_data['epoch'] + 1 # +1 because we want to start from the next epoch
                # Load the best validation loss achieved so far, not just the current epoch's val_loss
                best_val_loss = checkpoint_data.get('best_val_loss', 
                                                  checkpoint_data.get('val_loss', float('inf')))
                
                # Validate config compatibility if available
                if 'config' in checkpoint_data:
                    saved_config = checkpoint_data['config']
                    # Check key compatibility
                    if (saved_config.model.hidden_size != config.model.hidden_size or 
                        saved_config.training.per_gpu_batch_size != config.training.per_gpu_batch_size):
                        if rank == 0:
                            logger.warning(f"⚠️ Config mismatch detected - continuing with current config")
                
                if rank == 0:
                    logger.info(f"▶️ Resuming from epoch {start_epoch+1}")
                    logger.info(f"💾 Model state loaded successfully")
                    
            except Exception as e:
                if rank == 0:
                    logger.error(f"❌ Error loading checkpoint: {str(e)}")
                    logger.info("🔄 Starting training from scratch")
                start_epoch = 0
                best_val_loss = float('inf')
                checkpoint_data = None

        
        
        # Load dataset
        try:
            if rank == 0:
                logger.info("📊 Loading datasets...")

            datasets = AnsweringDataset.create_datasets(config, logger=logger, splits=['train', 'val_seen'], tokenizer=tokenizer)

        except Exception as e:
            logger.error(f"❌ Dataset loading failed: {str(e)}")
            TRAINING_FAILED = True
            raise e
        
        if rank == 0:
            logger.info(f"📊 Dataset: {len(datasets['train'])} train, {len(datasets['val_seen'])} validation")

        # Use DistributedSampler for distributed training
        if is_distributed:
            train_sampler = DistributedSampler(datasets['train'], num_replicas=world_size, rank=rank)
            val_sampler = DistributedSampler(datasets['val_seen'], num_replicas=world_size, rank=rank)
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
        
        if rank == 0:
            batch_str = f"📊 Per-GPU batch size: {config.training.per_gpu_batch_size}"
            if is_distributed:
                effective_batch_size = config.training.per_gpu_batch_size * world_size * config.training.gradient_accumulation_steps
                batch_str += f" (effective: {effective_batch_size})"
            logger.info(batch_str)
            
            # Log validation batch size
            val_batch_str = f"📊 Validation batch size: {config.training.per_gpu_batch_size_val}"
            if is_distributed:
                effective_val_batch_size = config.training.per_gpu_batch_size_val * world_size
                val_batch_str += f" (effective: {effective_val_batch_size})"
            logger.info(val_batch_str)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()          # throw away the giant arena
            torch.cuda.reset_peak_memory_stats()

        if rank == 0:
            logger.info(f"💾 Memory usage: {log_gpu_memory()}")

        train_loader = DataLoader(
            datasets['train'],
            batch_size=config.training.per_gpu_batch_size,
            sampler=train_sampler,
            shuffle=shuffle,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=(config.training.num_workers > 0)
        )

        val_loader = DataLoader(
            datasets['val_seen'],
            batch_size=config.training.per_gpu_batch_size_val,  # Smaller validation batch size
            sampler=val_sampler,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=False,
            persistent_workers=False  # Disable persistent workers for validation to save VRAM
        )

        # Create warmup then decay scheduler
        def get_lr_schedule(optimizer, warmup_steps, total_steps):
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    # Curriculum-aware decay
                    curriculum_phase_steps = int(total_steps * 0.15)
                    if current_step < warmup_steps + curriculum_phase_steps:
                        progress = float(current_step - warmup_steps) / float(max(1, curriculum_phase_steps))
                        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    else:
                        progress = float(current_step - warmup_steps - curriculum_phase_steps) / float(
                            max(1, total_steps - warmup_steps - curriculum_phase_steps))
                        return max(0.0, 0.3 * (1.0 + math.cos(math.pi * progress)))
            return LambdaLR(optimizer, lr_lambda)

        # Optimizer, loss, and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id,
            label_smoothing=0.05
        )
        
        # Calculate total steps
        total_steps = len(train_loader) * config.training.num_epochs // config.training.gradient_accumulation_steps
        
        # Create scheduler with warmup
        scheduler = get_lr_schedule(
            optimizer, 
            warmup_steps=config.training.warmup_steps,
            total_steps=total_steps
        )

        # Training
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.training.num_epochs,
            device=device,
            checkpoint_dir=config.checkpoint_dir,
            config=config,
            teacher_model=teacher_model,
            start_epoch=start_epoch,
            best_val_loss=best_val_loss,
            rank=rank,
            logger=logger,
            is_distributed=is_distributed,
            checkpoint_data=checkpoint_data
        )

        # Normal cleanup
        cleanup()
        
        if rank == 0:
            logger.info("🎉 Training completed successfully!")

    except Exception as e:
        # Mark training as failed
        TRAINING_FAILED = True
        if logger:
            error_msg = f"❌ Fatal error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Write to error file
            with open(temp_error_file.name, 'a') as f:
                f.write(f"RANK {rank}: {error_msg}\n")
                f.write(traceback.format_exc())
        else:
            print(f"❌ Fatal error: {str(e)}")
            print(traceback.format_exc())
    finally:
        # Proper cleanup for distributed environment
        if is_distributed and dist.is_initialized():
            try:
                dist.barrier()
                dist.destroy_process_group()
                if rank == 0 and logger:
                    logger.info("✅ Distributed cleanup complete")
            except Exception as e:
                if rank == 0 and logger:
                    logger.error(f"❌ Cleanup error: {e}")
        
        cleanup()
        
        if rank == 0:
            if TRAINING_FAILED:
                if logger:
                    logger.error("❌ Training failed")
                sys.exit(1)
            else:
                if logger:
                    logger.info("✅ Training completed successfully")

if __name__ == '__main__':
    import argparse
    main()
