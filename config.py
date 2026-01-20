"""
Configuration module for QLoRA Fine-Tuning Pipeline.

This module centralizes all hyperparameters and training configurations,
making it easy to experiment with different settings without modifying
the core training code.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the base model."""
    model_name: str = "meta-llama/Llama-3-8B"  # or "mistralai/Mistral-7B-v0.1"
    trust_remote_code: bool = True
    # 4-bit quantization reduces memory footprint by ~75% while maintaining
    # similar performance to full precision
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"  # "float32" for more precision
    bnb_4bit_quant_type: str = "nf4"  # Normalized Float 4-bit quantization
    bnb_4bit_use_double_quant: bool = True  # Nested quantization for better accuracy


@dataclass
class LoRAConfig:
    """
    LoRA (Low-Rank Adaptation) Configuration.
    
    LoRA freezes the base model weights and adds trainable low-rank matrices
    to attention layers. This allows fine-tuning with ~1% of original parameters.
    
    Key Parameters:
    - r (rank): Controls the rank of the low-rank matrices. Higher r = more
      parameters and capacity, but also more memory. Common values: 8, 16, 32, 64.
      r=64 is a good balance for 8B models, providing enough capacity for
      task-specific adaptation.
    
    - alpha (scaling): LoRA scaling parameter. The effective learning rate is
      (alpha/r) * base_learning_rate. alpha=16 with r=64 gives a scaling factor
      of 0.25, which works well in practice. Higher alpha relative to r increases
      the influence of LoRA updates.
    
    - target_modules: Which modules to apply LoRA to. For Llama/Mistral, we
      target "q_proj", "v_proj", "k_proj", "o_proj" (attention) and optionally
      "gate_proj", "up_proj", "down_proj" (MLP layers) for more capacity.
    
    - dropout: Regularization to prevent overfitting.
    """
    r: int = 64  # LoRA rank - controls capacity
    alpha: int = 16  # LoRA alpha - scaling parameter
    dropout: float = 0.1  # LoRA dropout
    bias: str = "none"  # Don't train bias terms
    task_type: str = "CAUSAL_LM"  # Causal language modeling
    target_modules: Optional[list] = None  # Will be set based on model
    
    def __post_init__(self):
        """Set default target modules if not specified."""
        if self.target_modules is None:
            # Default for Llama models
            self.target_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    # Output directories
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    
    # Dataset configuration
    dataset_name: str = "timdettmers/guanaco-llama2-1k"  # Placeholder dataset
    dataset_text_field: str = "text"  # Field name in dataset
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4  # Adjust based on GPU memory
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = batch_size * grad_accum
    
    # Learning rate - lower for fine-tuning (typically 1e-5 to 2e-4)
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"  # Cosine annealing scheduler
    warmup_ratio: float = 0.1  # 10% of training for warmup
    
    # Optimizer - PagedAdamW for memory efficiency
    optim: str = "paged_adamw_8bit"  # Memory-efficient AdamW optimizer
    
    # Memory optimization
    gradient_checkpointing: bool = True  # Trade compute for memory
    fp16: bool = True  # Mixed precision training
    bf16: bool = False  # Use bf16 if your GPU supports it (A100, H100)
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3  # Keep only last 3 checkpoints
    
    # Evaluation
    evaluation_strategy: str = "steps"
    
    # Maximum sequence length
    max_seq_length: int = 2048  # Adjust based on GPU memory
    
    # Other settings
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    report_to: str = "wandb"  # "none" to disable wandb


@dataclass
class WandBConfig:
    """Weights & Biases configuration for experiment tracking."""
    project_name: str = "llm-qlora-finetuning"
    run_name: Optional[str] = None
    tags: Optional[list] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = ["qlora", "fine-tuning", "llama3"]


# Global configuration instances
MODEL_CONFIG = ModelConfig()
LORA_CONFIG = LoRAConfig()
TRAINING_CONFIG = TrainingConfig()
WANDB_CONFIG = WandBConfig()

