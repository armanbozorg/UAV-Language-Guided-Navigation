import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from models.feature_extractor import FeatureExtractor
from typing import Dict, Tuple, Optional, List
import math
from config import Config
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import BaseModelOutput
from transformers.models.t5.modeling_t5 import T5EncoderModel

class TemporalObservationEncoder(nn.Module):
    """Encodes temporal observations with attention mechanism."""
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Attention mechanism for temporal observations
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network for processing attended features
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, current_features: torch.Tensor, prev_features: torch.Tensor) -> torch.Tensor:
        """
        Process current and previous features with attention.
        
        Args:
            current_features: Current view features [batch_size, hidden_size]
            prev_features: Previous views features [batch_size, num_prev, hidden_size]
            
        Returns:
            Temporally contextualized features [batch_size, hidden_size]
        """
        batch_size = current_features.size(0)
        
        # Add current features to previous features for self-attention
        # Shape: [batch_size, 1 + num_prev, hidden_size]
        combined_features = torch.cat([
            current_features.unsqueeze(1),
            prev_features
        ], dim=1)
        
        # Apply attention with current features as query, all features as key/value
        attn_output, _ = self.temporal_attention(
            query=current_features.unsqueeze(1),
            key=combined_features,
            value=combined_features
        )
        
        # Shape: [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        attn_output = attn_output.squeeze(1)
            
        features = self.norm1(current_features + attn_output)
        
        # Feed-forward network
        ff_output = self.ff_network(features)
        
        # Final residual connection and normalization
        output = self.norm2(features + ff_output)
        
        return output


class CrossModalFusion(nn.Module):
    """Fuses text and visual features using cross-attention."""
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Visual -> Text attention
        self.visual_to_text_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Text -> Visual attention
        self.text_to_visual_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-5)
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, text_features: torch.Tensor, visual_features: torch.Tensor, 
                text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse text and visual features with bidirectional attention.
        
        Args:
            text_features: Text encoded features [batch_size, seq_len, hidden_size]
            visual_features: Visual features [batch_size, num_visual_tokens, hidden_size]
            text_mask: Attention mask for text [batch_size, seq_len]
            
        Returns:
            Fused features with the same shape as text_features
        """
        batch_size, seq_len, _ = text_features.size()
        
        # Create attention mask from padding mask if provided
        attn_mask = None
        if text_mask is not None:
            # Convert from [batch_size, seq_len] to attention mask
            attn_mask = ~text_mask.bool()
        
        # Visual conditioning on text
        attended_text, _ = self.text_to_visual_attention(
            query=visual_features,
            key=text_features,
            value=text_features,
            key_padding_mask=attn_mask
        )
        
        # Text conditioning on visual
        attended_visual, _ = self.visual_to_text_attention(
            query=text_features,
            key=visual_features,
            value=visual_features
        )

        # Handle dimensional mismatch properly:
        # attended_text has shape [batch, num_visual_tokens, hidden] from visual query
        # attended_visual has shape [batch, seq_len, hidden] from text query
        # We need both to have shape [batch, seq_len, hidden] for fusion
        
        # For attended_text: aggregate visual tokens to get single representation per text position
        # Use a learned aggregation instead of interpolation
        if attended_text.size(1) != seq_len:
            # Project visual tokens to text positions using attention pooling
            visual_to_text_pooling = torch.softmax(
                torch.matmul(text_features, attended_text.transpose(-2, -1)) / math.sqrt(text_features.size(-1)), 
                dim=-1
            )  # [batch, seq_len, num_visual_tokens]
            attended_text = torch.matmul(visual_to_text_pooling, attended_text)  # [batch, seq_len, hidden]

        attended_text = self.norm1(attended_text)
        attended_visual = self.norm2(attended_visual)
        
        # Compute fusion gate
        # Determine how much of each modality to use at each position
        gate = self.fusion_gate(torch.cat([attended_text, attended_visual], dim=-1))
        
        # Weighted combination of the two streams
        fused_features = gate * attended_visual + (1 - gate) * attended_text

        fused_features = self.norm3(fused_features)
        
        # Concatenate and project for rich feature representation
        output = text_features + self.output_projection(torch.cat([fused_features, text_features], dim=-1))
        
        return output


class SeparateEncodingTextProcessor(nn.Module):
    """
    Separate encoding approach: Use pre-tokenized components from dataset.
    
    This approach:
    1. Uses pre-tokenized first_instruction, current_question, and unified_context from dataset
    2. Encodes each component separately with specialized encoders
    3. Fuses the separate encodings for multi-level understanding
    
    Benefits:
    - No text parsing needed - uses dataset components directly
    - Better specialization for each component
    - Easier to debug and interpret
    - Simpler implementation
    """
    
    def __init__(self, config, shared_encoder):
        super().__init__()
        self.config = config
        
        self.encoder = shared_encoder
        
        # Feature projections
        self.goal_proj = nn.Linear(config.model.hidden_size, config.model.hidden_size)
        self.context_proj = nn.Linear(config.model.hidden_size, config.model.hidden_size)
        self.unified_proj = nn.Linear(config.model.hidden_size, config.model.hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(config.model.hidden_size * 3, config.model.hidden_size)
        
    def forward(self, unified_input, unified_mask, goal_input, goal_mask, context_input, context_mask):
        """
        Process pre-tokenized components with separate encoding.
        
        Args:
            unified_input: Unified dialog context token IDs
            unified_mask: Unified dialog context attention mask
            goal_input: First instruction token IDs
            goal_mask: First instruction attention mask
            context_input: Current question token IDs
            context_mask: Current question attention mask
            
        Returns:
            Multi-level processed features
        """
        # Encode each component separately
        
        # Goal encoding (first instruction)
        goal_output = self.encoder(
            input_ids=goal_input,
            attention_mask=goal_mask,
            return_dict=True
        )
        goal_features = self.goal_proj(goal_output.last_hidden_state.mean(dim=1))
        
        # Context encoding (current question)
        context_output = self.encoder(
            input_ids=context_input,
            attention_mask=context_mask,
            return_dict=True
        )
        context_features = self.context_proj(context_output.last_hidden_state.mean(dim=1))
        
        # Unified encoding (complete dialog context)
        unified_output = self.encoder(
            input_ids=unified_input,
            attention_mask=unified_mask,
            return_dict=True
        )
        unified_features = self.unified_proj(unified_output.last_hidden_state.mean(dim=1))
        
        # Fuse all components
        fused_features = self._fuse_components(goal_features, context_features, unified_features)
        
        return fused_features, unified_output.last_hidden_state
    
    def _fuse_components(self, goal_features, context_features, unified_features):
        """
        Fuse goal, context, and unified features.
        
        Args:
            goal_features: [batch_size, hidden_size]
            context_features: [batch_size, hidden_size]
            unified_features: [batch_size, hidden_size]
            
        Returns:
            Fused features [batch_size, hidden_size]
        """
        # Simple concatenation + projection approach
        concatenated = torch.cat([goal_features, context_features, unified_features], dim=-1)
        fused_output = self.output_proj(concatenated)
        
        return fused_output


class AnsweringAgent(nn.Module):
    """
    Answering Agent for aerial navigation.
    
    This model integrates:
    1. Pretrained T5 language model for text processing
    2. Visual feature extraction for individual images
    3. Temporal observation encoding (current + previous views)
    4. Cross-modal fusion between text and visual features
    
    Architecture follows cognitive science principles:
    - Explicit memory for past observations via the temporal encoder
    - Input formatting that highlights the first instruction naturally
    - Cross-modal alignment of vision and language
    - Fine-tuning only necessary parts of the pretrained model
    """
    
    def __init__(self, config: Config, tokenizer=None, logger=None):
        super().__init__()
        self.config = config
        
        # Set up logger for this instance
        self.logger = logger
        
        # Store T5 model name for loading correct tokenizer
        self.model_name = config.model.t5_model_name
        
        # Use provided tokenizer or create one
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, model_max_length=self.config.data.max_seq_length, add_special_tokens=True)
        
        # Load T5 base model (encoder-decoder)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.t5_config = self.t5_model.config
        
        # Feature extractor for visual processing
        self.feature_extractor = FeatureExtractor(config)
        
        # Temporal observation encoder for processing previous views
        self.temporal_encoder = TemporalObservationEncoder(
            hidden_size=config.model.hidden_size,
            num_heads=config.model.num_attention_heads,
            dropout=config.model.dropout
        )

        # Project visual context to 32 times the hidden size
        self.visual_context_projection = nn.Linear(config.model.hidden_size, config.model.num_visual_tokens * config.model.hidden_size)
        
        # Cross-modal fusion for text and visual features
        self.fusion_module = CrossModalFusion(
            hidden_size=config.model.hidden_size,
            num_heads=config.model.num_attention_heads,
            dropout=config.model.dropout
        )
        
        self.separate_encoding_processor = SeparateEncodingTextProcessor(config, self.t5_model.encoder)
        self.paraphrase_proj = nn.Linear(config.model.hidden_size, config.model.hidden_size)
        self.paraphrase_weight = nn.Parameter(torch.tensor(0.0))
        
        # Contrastive projection head - gives model capacity to separate embeddings
        self.contrastive_proj = nn.Sequential(
            nn.Linear(config.model.hidden_size, config.model.hidden_size),
            nn.ReLU(),
            nn.Linear(config.model.hidden_size, config.model.hidden_size),
            nn.LayerNorm(config.model.hidden_size)
        )
        
        # T5 Adapter layer - bridges the gap between our fused features and what T5 decoder expects
        self.t5_adapter = nn.Sequential(
            nn.Linear(config.model.hidden_size, config.model.hidden_size),
            nn.LayerNorm(config.model.hidden_size),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_size, config.model.hidden_size),
            nn.LayerNorm(config.model.hidden_size)
        )
        
        # Initialize adapter weights
        self._init_adapter_weights()
        
        # Freeze the entire T5 model
        self._freeze_t5_parameters()
        
    def _init_adapter_weights(self):
        """Initialize adapter weights carefully to ensure good initial performance"""
        for module in self.t5_adapter.modules():
            if isinstance(module, nn.Linear):
                # Use small initialization for stability
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def _freeze_t5_parameters(self):
        """Freeze ALL T5 parameters for efficiency."""
        # Count parameters
        total_params = 0
        
        # Freeze all T5 parameters
        for name, param in self.t5_model.named_parameters():
            total_params += param.numel()
            param.requires_grad = False
        
        # UNFREEZE the last two encoder blocks to give the model more capacity in stage-2 fine-tuning
        for idx in [-1, -2, -3]:
            for name, param in self.t5_model.encoder.block[idx].named_parameters():
                param.requires_grad = True
            
            # UNFREEZE the last two decoder blocks to allow the language
            # generation head to adapt to new encoder representations.
            # This is a light‐weight alternative to fully unfreezing the
            # decoder and keeps most of the pretrained knowledge intact while
            # giving the model capacity to generate less generic answers.
            for idx in [-1, -2]:
                for name, param in self.t5_model.decoder.block[idx].named_parameters():
                    param.requires_grad = True
            
            # Also unfreeze the decoder's final layer norm so that its output
            # distribution can shift together with the newly trainable blocks.
            for name, param in self.t5_model.decoder.final_layer_norm.named_parameters():
                param.requires_grad = True
                
        
        # Count our trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"Total trainable parameters: {trainable_params:,}")
        self.logger.info(f"Total T5 parameters: {total_params:,}")
        self.logger.info(f"T5 model: {trainable_params/total_params*100:.2f}% of parameters are trainable")
        
        # Detailed breakdown of trainable components
        encoder_trainable = sum(p.numel() for n, p in self.t5_model.encoder.named_parameters() if p.requires_grad)
        decoder_trainable = sum(p.numel() for n, p in self.t5_model.decoder.named_parameters() if p.requires_grad)
        other_trainable = trainable_params - encoder_trainable - decoder_trainable
        
        self.logger.info(f"📊 Trainable breakdown:")
        self.logger.info(f"  • Encoder (last 3 blocks): {encoder_trainable:,} parameters")
        self.logger.info(f"  • Decoder (last 2 blocks + final norm): {decoder_trainable:,} parameters")
        self.logger.info(f"  • Other modules (adapters, projections): {other_trainable:,} parameters")
       
    
    def forward(self, text_input: dict, current_view: torch.Tensor, 
                previous_views: torch.Tensor, labels: torch.Tensor = None, generate: bool = False,
                destination_view: Optional[torch.Tensor] = None, curriculum_ratio: float = 0.0,
                positive_input: Optional[dict] = None, positive_input_2: Optional[dict] = None, 
                negative_input: Optional[dict] = None, negative_input_2: Optional[dict] = None,
                **generation_kwargs) -> Dict:
        """
        Forward pass of the model.
        
        Args:
            text_input (dict): Tokenized input with keys 'input_ids' and 'attention_mask'
            current_view (torch.Tensor): Current view image tensor [batch_size, 3, H, W]
            previous_views (torch.Tensor): Previous views tensor [batch_size, max_prev, 3, H, W]
            labels (torch.Tensor, optional): Target labels for generation/loss calculation
            generate (bool): Whether to generate text instead of calculating loss
            destination_view (torch.Tensor, optional): Destination view for curriculum learning
            curriculum_ratio (float): Ratio for curriculum learning (0-1)
            positive_input (dict, optional): First positive example for contrastive learning
            positive_input_2 (dict, optional): Second positive example for contrastive learning
            negative_input (dict, optional): Negative example for contrastive learning
            
        Returns:
            Dict containing model outputs, including:
                - logits: Output logits
                - encoder_last_hidden_state: Encoder hidden states
                - visual_context: Visual context
                - adapted_features: Adapted features for contrastive learning
                - positive_adapted_features: First positive adapted features (if positive_input provided)
                - positive_adapted_features_2: Second positive adapted features (if positive_input_2 provided)
                - negative_adapted_features: Negative adapted features (if negative_input provided)
        """
        batch_size = current_view.size(0)
        device = current_view.device
        
        # --- Visual Processing ---
        # Extract visual features from current view
        # Extract visual features
        if hasattr(self, 'feature_extractor'):
            current_features = self.feature_extractor(current_view)
        else:
            # Handle cases where we might load a checkpoint with different architecture
            self.logger.warning("Feature extractor not found, returning zero features")
            current_features = torch.zeros(batch_size, self.config.model.hidden_size, 
                                        device=device)
        
        # Process previous views if available
        if previous_views.size(0) > 0:
            # Extract features for each previous view
            num_prev = min(previous_views.size(1), self.config.data.max_previous_views)
            
            # Reshape to process each view separately
            views_to_process = previous_views[:, :num_prev].contiguous()
            views_flat = views_to_process.view(-1, *views_to_process.shape[2:])
            
            # Process all views at once for efficiency
            all_prev_features = self.feature_extractor(views_flat)
            
            # Reshape back to [batch, num_prev, hidden]
            prev_features = all_prev_features.view(batch_size, num_prev, -1)
        else:
            # Default to empty tensor if no previous views
            prev_features = torch.zeros(batch_size, 1, self.config.model.hidden_size,
                                      device=device)
        
        # Apply temporal encoding to incorporate previous views
        visual_context = self.temporal_encoder(current_features, prev_features)

        # Process destination image if provided (for curriculum learning)
        dest_features = None
        if destination_view is not None:
            dest_features = self.feature_extractor(destination_view)
            if curriculum_ratio > 0:
            # Use linear interpolation for curriculum learning
            # As training progresses, curriculum_ratio decreases
            # - Early training: rely more on destination (oracle)
            # - Later training: rely more on visual context (learned)
                visual_context = (
                    curriculum_ratio * dest_features + 
                    (1 - curriculum_ratio) * visual_context
                )
        
        # --- Text Processing ---
        # Get T5 encoder outputs for the input text
        # NOTE: text_input contains unified dialog context:
        # "First Instruction: ... Question: ... Answer: ... Question: {current}"
        # This unified approach preserves natural conversation flow and inter-relationships
        # between goal, history, and current context - optimal for T5's sequence understanding
        
        # Separate encoding processing with pre-tokenized components
        text_fused_features, text_features_seq = self.separate_encoding_processor(
            unified_input=text_input["input_ids"],
            unified_mask=text_input["attention_mask"],
            goal_input=text_input["first_instruction_input"]["input_ids"],
            goal_mask=text_input["first_instruction_input"]["attention_mask"],
            context_input=text_input["current_question_input"]["input_ids"],
            context_mask=text_input["current_question_input"]["attention_mask"]
        )
        text_features_expanded = text_features_seq
        

        # --- Cross-Modal Fusion ---
        # Visual tokens need to be the same dimension as T5's hidden states
        # Project visual context to create multiple visual tokens
        visual_ctx_expanded = self.visual_context_projection(visual_context)
        visual_ctx_expanded = visual_ctx_expanded.view(
            batch_size, 
            self.config.model.num_visual_tokens, 
            self.config.model.hidden_size
        )
        
        # Get text features from encoder

        # Apply cross-modal fusion between text and visual features
        fused_features = self.fusion_module(
            text_features=text_features_expanded, # Use expanded text features
            visual_features=visual_ctx_expanded,
            text_mask=text_input["attention_mask"]
        )
            
        # Adapt the fused features to work with T5 decoder
        encoder_hidden_states = self.t5_adapter(fused_features)

        
        # --- Decoder Processing ---
        # Calculate logits or generate text
        if not generate:
            # Training or validation mode
            
            # Use the full T5 model which handles teacher forcing automatically
            # This properly creates decoder_input_ids by shifting labels and uses them correctly
            t5_outputs = self.t5_model(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
                attention_mask=text_input["attention_mask"],
                labels=labels,
                return_dict=True
            )
            
            logits = t5_outputs.logits
            
            # Compute once and reuse to avoid redundant computation
            encoder_hidden_mean = encoder_hidden_states.mean(dim=1)  # [batch, hidden]
                
            # Create output dictionary
            outputs = {
                "logits": logits,
                "loss": t5_outputs.loss,  # Include the T5-calculated loss
                "encoder_last_hidden_state": encoder_hidden_states,
                "visual_context": visual_context,
                "raw_adapted_features": encoder_hidden_mean,  # Raw features for destination similarity - reuse computation
                "adapted_features": self.contrastive_proj(encoder_hidden_mean),  # Contrastive-optimized features - reuse computation
                "feature_norm": encoder_hidden_states.norm(p=2, dim=-1).mean()  # Use adapted features norm, not raw visual context
            }
            
            
            if destination_view is not None:
                outputs["destination_features"] = dest_features
            
            p_weight = torch.sigmoid(self.paraphrase_weight)
            # --- Process positive examples for contrastive learning ---
            if positive_input is not None:                    
                # Encode positive paraphrase hint separately and combine with adapted features
                positive_hint_output = self.t5_model.encoder(
                    input_ids=positive_input["input_ids"].to(text_input["input_ids"].device),
                    attention_mask=positive_input["attention_mask"].to(text_input["input_ids"].device),
                    return_dict=True
                )
                positive_hint_features = self.paraphrase_proj(positive_hint_output.last_hidden_state.mean(dim=1))
                
                # Combine adapted features with positive hint (reuse precomputed mean)
                combined_positive_features = encoder_hidden_mean + p_weight * positive_hint_features  # Weighted combination
                outputs["positive_adapted_features"] = self.contrastive_proj(combined_positive_features)  # Apply contrastive projection
                
            # --- Process second positive example for contrastive learning ---
            if positive_input_2 is not None:                    
                # Encode second positive paraphrase hint
                positive_hint_output_2 = self.t5_model.encoder(
                    input_ids=positive_input_2["input_ids"].to(text_input["input_ids"].device),
                    attention_mask=positive_input_2["attention_mask"].to(text_input["input_ids"].device),
                    return_dict=True
                )
                positive_hint_features_2 = self.paraphrase_proj(positive_hint_output_2.last_hidden_state.mean(dim=1))
                
                # Combine adapted features with positive hint (reuse precomputed mean)
                combined_positive_features_2 = encoder_hidden_mean + p_weight * positive_hint_features_2
                outputs["positive_adapted_features_2"] = self.contrastive_proj(combined_positive_features_2)  # Apply contrastive projection
                
            # --- Process negative examples for contrastive learning ---
            if negative_input is not None:
                # Encode negative paraphrase hint separately
                negative_hint_output = self.t5_model.encoder(
                    input_ids=negative_input["input_ids"].to(text_input["input_ids"].device),
                    attention_mask=negative_input["attention_mask"].to(text_input["input_ids"].device),
                    return_dict=True
                )
                negative_hint_features = self.paraphrase_proj(negative_hint_output.last_hidden_state.mean(dim=1))
                
                # Combine adapted features with negative hint (reuse precomputed mean)
                combined_negative_features = encoder_hidden_mean + p_weight * negative_hint_features
                outputs["negative_adapted_features"] = self.contrastive_proj(combined_negative_features)  # Apply contrastive projection
                
                
            # --- Process negative_2 examples for contrastive learning (mined negatives) ---
            if negative_input_2 is not None:
                # Encode negative_2 hint separately
                negative_hint_output_2 = self.t5_model.encoder(
                    input_ids=negative_input_2["input_ids"].to(text_input["input_ids"].device),
                    attention_mask=negative_input_2["attention_mask"].to(text_input["input_ids"].device),
                    return_dict=True
                )
                negative_hint_features_2 = self.paraphrase_proj(negative_hint_output_2.last_hidden_state.mean(dim=1))
                
                # Combine adapted features with negative_2 hint (reuse precomputed mean)
                combined_negative_features_2 = encoder_hidden_mean + p_weight * negative_hint_features_2
                outputs["negative_adapted_features_2"] = self.contrastive_proj(combined_negative_features_2)  # Apply contrastive projection
                
            return outputs
        else:
            # -----------------------
            # GENERATION MODE
            # -----------------------

            # Memory optimization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Default generation settings – can be overridden via **generation_kwargs
            default_gen_args = {
                "max_new_tokens": 32,
                "min_length": 5,
                "num_beams": 3,
                "do_sample": False,
                "repetition_penalty": 1.1,
                "length_penalty": 1.0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "early_stopping": True,
            }

            # Update defaults with any user-provided overrides
            default_gen_args.update(generation_kwargs)

            generated_ids = self.t5_model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
                attention_mask=text_input["attention_mask"],
                **default_gen_args,
            )
            
            return {
                "sequences": generated_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": text_input["attention_mask"]
            }
    
            
    def generate_answer(self, text_input: dict, current_view: torch.Tensor, 
                       previous_views: torch.Tensor, **generation_kwargs) -> torch.Tensor:
        """
        Convenience wrapper that forwards generation_kwargs to the underlying
        T5 generate call via the modified forward().
        """

        with torch.no_grad():
            outputs = self.forward(
                text_input, current_view, previous_views,
                generate=True,
                **generation_kwargs,
            )

        return outputs["sequences"]