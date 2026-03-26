from dataclasses import dataclass, field
from pathlib import Path
import os
import torch

# Update paths for Docker container structure
PROJECT_ROOT = Path("/app/UAV-Language-Guided-Navigation")
DATASET_ROOT = Path("/app/datasets")

@dataclass
class ModelConfig:
    """Configuration for the AnsweringAgent model."""
    bert_model_name: str = 'bert-base-uncased'  # Legacy setting
    t5_model_name: str = 't5-base'  # New setting for T5
    hidden_size: int = 768  # Match T5-base hidden size (d_model)
    dropout: float = 0.2  # Lower dropout for UAV spatial reasoning retention
    feat_dropout: float = 0.3  # Reduced for better aerial feature preservation
    num_decoder_layers: int = 4  # Not used when using pretrained T5 decoder
    num_attention_heads: int = 8  # Match T5-base (8 heads)
    num_visual_tokens: int = 64  # Increased for richer aerial imagery representation
    feedforward_dim: int = 2048  # Match T5-base feed forward dimension
    max_answer_length: int = 128
    vocab_size: int = 32128  # T5 vocabulary size for t5-base
    img_size: int = 224  # Image size for Darknet/YOLO model
    use_t5: bool = True  # Flag to control which model type to use
    use_pretrained_decoder: bool = True  # Use T5's pretrained decoder instead of custom

@dataclass
class TrainingConfig:      
    num_epochs: int = 3000  # Maximum epochs - early stopping determines actual end
    planned_epochs: int = 400  # 3-phase planning horizon  
    learning_rate: float = 3e-5  # Slightly lower for stable UAV spatial learning convergence
    weight_decay: float = 0.008  # Moderate regularization for UAV feature preservation
    gradient_clip: float = 1.0
    warmup_steps: int = 1000  # Longer warmup for stability in long training
    log_freq: int = 20
    eval_freq: int = 25  # Less frequent evaluation for long training
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = False  # DISABLED - was causing hang at backward pass
    device: str = 'cuda'
    seed: int = 42
    checkpoint_frequency: int = 100  # Save every 100 epochs (much less frequent)
    scheduler_factor: float = 0.5
    scheduler_patience: int = 20  # More patience for longer training
    scheduler_verbose: bool = True
    gradient_accumulation_steps: int = 3  # Increased from 2 to compensate for smaller batch
    # Early stopping parameters
    early_stopping: bool = True
    early_stopping_patience: int = 20  # Much more patience for long 3-phase training
    early_stopping_min_delta: float = 0.002  # Smaller delta for long training
    # Validation parameters
    per_gpu_batch_size: int = 6  # Reduced from 8 for unfrozen decoder
    per_gpu_batch_size_val: int = 6  # Reduced from 8 for unfrozen decoder
    train_chunk_size: int = 1000
    # Curriculum learning parameters - ADAPTED FOR UAV SPATIAL LEARNING
    destination_loss_weight_start: float = 1.2  # Higher for UAV navigation importance
    destination_loss_weight_end: float = 0.1   # Maintain some spatial guidance
    
    # These are overridden by smart scheduling but kept for compatibility
    ce_loss_weight_start: float = 0.1  # Will be overridden by smart scheduler
    ce_loss_weight_end: float = 1.5   # Will be overridden by smart scheduler
    
    # Contrastive Learning Parameters - OPTIMIZED FOR UAV AERIAL IMAGERY
    use_contrastive_learning: bool = True
    contrastive_loss_type: str = "infonce"
    contrastive_margin: float = 0.4  # Larger margin for UAV aerial perspective differences
    contrastive_temperature: float = 0.15  # Higher temp for aerial imagery similarity learning
    contrastive_weight_start: float = 10.0  # Will be overridden by smart scheduler
    contrastive_weight_end: float = 3.0    # Will be overridden by smart scheduler
    # UAV-specific triplet loss options
    use_cosine_distance: bool = True  # Better for aerial visual similarities
    contrastive_mean_all: bool = False  # More selective for UAV landmark learning
    
    # Add per-epoch weight logging for debugging
    log_loss_weights: bool = True  
    
    # Knowledge-distillation (KD) parameters - NOW WITH SMART SCHEDULING
    use_kd: bool = True  # Enable KD with smart 3-phase scheduling
    kd_teacher_model_name: str = "sentence-transformers/all-mpnet-base-v2"  
    kd_weight_start: float = 0.5  # Overridden by smart scheduler but kept for compatibility
    kd_weight_end: float = 0.1    # Overridden by smart scheduler but kept for compatibility

    def __post_init__(self):
        """Initialize GPU settings and scale batch size/workers."""
        if not torch.cuda.is_available():
            print("CUDA is not available. Please ensure a compatible GPU is installed and drivers are set up correctly.")

        self.num_gpus = torch.cuda.device_count()
        self.device = 'cuda'

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    train_processed_path_dir: str = str(DATASET_ROOT / "train/")
    val_seen_processed_path: str = str(DATASET_ROOT / "val_seen_processed_dataset.pkl")
    val_unseen_processed_path: str = str(DATASET_ROOT / "val_unseen_processed_dataset.pkl")
    train_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/processed_data/train_data.json")
    val_seen_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/processed_data/val_seen_data.json")
    val_unseen_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/processed_data/val_unseen_data.json")
    
    # Augmented dataset paths with paraphrases (NEW - for contrastive learning)
    use_augmented_data: bool = True  # Toggle to use augmented data with paraphrases
    train_augmented_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/augmented_data/train_contrastive.json")
    val_seen_augmented_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/augmented_data/val_seen_contrastive.json")
    val_unseen_augmented_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/augmented_data/val_unseen_contrastive.json")
    
    # Data preprocessing settings
    use_augmentation: bool = False  # Enable/disable visual augmentation during preprocessing
    
    avdn_image_dir: str = str(DATASET_ROOT / "AVDN/train_images")
    avdn_annotations_dir: str = str(PROJECT_ROOT / "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations")
    darknet_config_path: str = str(PROJECT_ROOT / "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/pretrain_weights/yolo_v3.cfg")
    darknet_weights_path: str = str(PROJECT_ROOT / "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/pretrain_weights/best.pt")
    max_previous_views: int = 3
    train_val_split: float = 0.90  # Updated: 90% for training
    val_test_split: float = 0.5    # New: 50% of remaining data for validation, 50% for testing
    max_seq_length: int = 512

    def __post_init__(self):
        """Verify paths exist."""
        if self.use_augmented_data:
            # Check augmented data paths
            paths = [self.train_augmented_json_path, self.val_seen_augmented_json_path, 
                    self.val_unseen_augmented_json_path, self.avdn_image_dir,
                self.darknet_config_path, self.darknet_weights_path]
        else:
            # Check original data paths
            paths = [self.train_json_path, self.val_seen_json_path, self.val_unseen_json_path,
                    self.avdn_image_dir, self.darknet_config_path, self.darknet_weights_path]
        
        for path in paths:
            if not os.path.exists(path):
                print(f"Warning: Path does not exist: {path}")
                if self.use_augmented_data and "augmented_data" in path:
                    print(f"  Hint: Run comprehensive_avdn_pipeline.py to generate augmented data")
    
    def get_json_path(self, split: str) -> str:
        """Get the appropriate JSON path for a dataset split."""
        if self.use_augmented_data:
            if split == 'train':
                return self.train_augmented_json_path
            elif split == 'val_seen':
                return self.val_seen_augmented_json_path
            elif split == 'val_unseen':
                return self.val_unseen_augmented_json_path
        else:
            if split == 'train':
                return self.train_json_path
            elif split == 'val_seen':
                return self.val_seen_json_path
            elif split == 'val_unseen':
                return self.val_unseen_json_path
        
        raise ValueError(f"Unknown split: {split}")

@dataclass
class Config:
    """Main configuration class combining all settings."""
    checkpoint_dir: str = str(PROJECT_ROOT/ 'AnsweringAgent/outputs/checkpoints')
    log_dir: str = str(PROJECT_ROOT/ 'AnsweringAgent/outputs/logs')

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)