"""
Main Training Script for QLoRA Fine-Tuning.

This script implements the complete training pipeline using:
- 4-bit quantization (QLoRA)
- LoRA adapters
- SFTTrainer from TRL library
- Gradient checkpointing for memory efficiency
- PagedAdamW optimizer
- Weights & Biases integration for experiment tracking
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import wandb
from datetime import datetime

from config import MODEL_CONFIG, LORA_CONFIG, TRAINING_CONFIG, WANDB_CONFIG
from data_loader import load_and_prepare_data


class LoggingCallback(TrainerCallback):
    """Custom callback for logging training statistics to console."""
    
    def __init__(self, log_steps: int = 10):
        self.log_steps = log_steps
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics."""
        if state.global_step % self.log_steps == 0 and logs:
            print(f"\n[Step {state.global_step}/{state.max_steps}]")
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        print(f"\n{'='*50}")
        print(f"Epoch {int(state.epoch)} completed!")
        print(f"Global Step: {state.global_step}")
        print(f"Loss: {state.log_history[-1].get('loss', 'N/A'):.4f}")
        print(f"{'='*50}\n")


def setup_model_and_tokenizer():
    """
    Load and configure the base model with 4-bit quantization.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print("="*60)
    print("SETTING UP MODEL AND TOKENIZER")
    print("="*60)
    
    # Configure 4-bit quantization
    compute_dtype = getattr(torch, MODEL_CONFIG.bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=MODEL_CONFIG.load_in_4bit,
        bnb_4bit_quant_type=MODEL_CONFIG.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=MODEL_CONFIG.bnb_4bit_use_double_quant,
    )
    
    print(f"Loading model: {MODEL_CONFIG.model_name}")
    print(f"Quantization: 4-bit ({MODEL_CONFIG.bnb_4bit_quant_type})")
    print(f"Compute dtype: {compute_dtype}")
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG.model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically handle device placement
        trust_remote_code=MODEL_CONFIG.trust_remote_code,
        torch_dtype=compute_dtype,
    )
    
    # Enable gradient checkpointing for memory efficiency
    # This trades compute for memory by recomputing activations during backward pass
    if TRAINING_CONFIG.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
    
    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG.model_name,
        trust_remote_code=MODEL_CONFIG.trust_remote_code,
        use_fast=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def setup_lora(model):
    """
    Configure and apply LoRA adapters to the model.
    
    Args:
        model: The base model (with quantization)
        
    Returns:
        PEFT model with LoRA adapters
    """
    print("="*60)
    print("CONFIGURING LoRA ADAPTERS")
    print("="*60)
    
    # Prepare model for k-bit training
    # This wraps the model to enable efficient training with quantized weights
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    # r=64: Rank of low-rank matrices (higher = more parameters, more capacity)
    # alpha=16: Scaling parameter (effective learning rate = alpha/r * base_lr)
    # target_modules: Which layers to apply LoRA to
    lora_config = LoraConfig(
        r=LORA_CONFIG.r,
        lora_alpha=LORA_CONFIG.alpha,
        target_modules=LORA_CONFIG.target_modules,
        lora_dropout=LORA_CONFIG.dropout,
        bias=LORA_CONFIG.bias,
        task_type=LORA_CONFIG.task_type,
    )
    
    print(f"LoRA Configuration:")
    print(f"  Rank (r): {LORA_CONFIG.r}")
    print(f"  Alpha: {LORA_CONFIG.alpha}")
    print(f"  Scaling factor: {LORA_CONFIG.alpha / LORA_CONFIG.r:.4f}")
    print(f"  Dropout: {LORA_CONFIG.dropout}")
    print(f"  Target modules: {LORA_CONFIG.target_modules}")
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def setup_wandb():
    """Initialize Weights & Biases for experiment tracking."""
    if TRAINING_CONFIG.report_to != "wandb":
        print("WandB tracking disabled.")
        return
    
    print("="*60)
    print("INITIALIZING WEIGHTS & BIASES")
    print("="*60)
    
    run_name = WANDB_CONFIG.run_name or f"qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    wandb.init(
        project=WANDB_CONFIG.project_name,
        name=run_name,
        tags=WANDB_CONFIG.tags,
        config={
            "model_name": MODEL_CONFIG.model_name,
            "lora_r": LORA_CONFIG.r,
            "lora_alpha": LORA_CONFIG.alpha,
            "learning_rate": TRAINING_CONFIG.learning_rate,
            "batch_size": TRAINING_CONFIG.per_device_train_batch_size,
            "gradient_accumulation_steps": TRAINING_CONFIG.gradient_accumulation_steps,
            "num_epochs": TRAINING_CONFIG.num_train_epochs,
            "max_seq_length": TRAINING_CONFIG.max_seq_length,
        }
    )
    
    print(f"WandB initialized: {run_name}")


def train(
    train_dataset: Dataset,
    eval_dataset: Dataset = None,
    tokenizer: AutoTokenizer = None
):
    """
    Main training function.
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        tokenizer: Tokenizer for the model
    """
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Setup WandB
    setup_wandb()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Configure training arguments
    # PagedAdamW optimizer is specified via optim="paged_adamw_8bit"
    # This uses 8-bit quantized optimizer states, saving ~50% memory
    training_args = TrainingArguments(
        output_dir=TRAINING_CONFIG.output_dir,
        logging_dir=TRAINING_CONFIG.logging_dir,
        
        # Training hyperparameters
        num_train_epochs=TRAINING_CONFIG.num_train_epochs,
        per_device_train_batch_size=TRAINING_CONFIG.per_device_train_batch_size,
        per_device_eval_batch_size=TRAINING_CONFIG.per_device_eval_batch_size,
        gradient_accumulation_steps=TRAINING_CONFIG.gradient_accumulation_steps,
        
        # Learning rate
        learning_rate=TRAINING_CONFIG.learning_rate,
        lr_scheduler_type=TRAINING_CONFIG.lr_scheduler_type,
        warmup_ratio=TRAINING_CONFIG.warmup_ratio,
        
        # Optimizer - PagedAdamW for memory efficiency
        optim=TRAINING_CONFIG.optim,
        
        # Precision
        fp16=TRAINING_CONFIG.fp16,
        bf16=TRAINING_CONFIG.bf16,
        
        # Gradient checkpointing (already enabled on model)
        gradient_checkpointing=TRAINING_CONFIG.gradient_checkpointing,
        
        # Logging and saving
        logging_steps=TRAINING_CONFIG.logging_steps,
        save_steps=TRAINING_CONFIG.save_steps,
        eval_steps=TRAINING_CONFIG.eval_steps,
        save_total_limit=TRAINING_CONFIG.save_total_limit,
        
        # Evaluation
        evaluation_strategy=TRAINING_CONFIG.evaluation_strategy,
        eval_accumulation_steps=1,
        
        # Other settings
        dataloader_pin_memory=TRAINING_CONFIG.dataloader_pin_memory,
        remove_unused_columns=TRAINING_CONFIG.remove_unused_columns,
        report_to=TRAINING_CONFIG.report_to,
        
        # Save and load best model
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        
        # Push to hub (optional)
        push_to_hub=False,
    )
    
    print("\nTraining Arguments:")
    print(f"  Output directory: {TRAINING_CONFIG.output_dir}")
    print(f"  Learning rate: {TRAINING_CONFIG.learning_rate}")
    print(f"  Batch size: {TRAINING_CONFIG.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {TRAINING_CONFIG.gradient_accumulation_steps}")
    print(f"  Effective batch size: {TRAINING_CONFIG.per_device_train_batch_size * TRAINING_CONFIG.gradient_accumulation_steps}")
    print(f"  Optimizer: {TRAINING_CONFIG.optim}")
    print(f"  Gradient checkpointing: {TRAINING_CONFIG.gradient_checkpointing}")
    print(f"  Max sequence length: {TRAINING_CONFIG.max_seq_length}")
    
    # Initialize SFTTrainer
    # SFTTrainer (Supervised Fine-Tuning Trainer) is specifically designed
    # for instruction-tuning and handles prompt-response formatting automatically
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=None,  # Already applied via get_peft_model
        dataset_text_field=TRAINING_CONFIG.dataset_text_field,
        max_seq_length=TRAINING_CONFIG.max_seq_length,
        tokenizer=tokenizer,
        packing=False,  # Don't pack multiple sequences (better for instruction tuning)
        callbacks=[LoggingCallback(log_steps=TRAINING_CONFIG.logging_steps)],
    )
    
    # Start training
    print("\n" + "="*60)
    print("BEGINNING TRAINING LOOP")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(TRAINING_CONFIG.output_dir)
    
    print(f"\nTraining completed! Model saved to {TRAINING_CONFIG.output_dir}")
    
    # Finish WandB run
    if TRAINING_CONFIG.report_to == "wandb":
        wandb.finish()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LLM with QLoRA")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Hugging Face dataset name or path to local file"
    )
    parser.add_argument(
        "--file-type",
        type=str,
        default="json",
        choices=["json", "csv"],
        help="File type if using local dataset"
    )
    parser.add_argument(
        "--format-instructions",
        action="store_true",
        help="Format instruction/input/output structure"
    )
    
    args = parser.parse_args()
    
    # Load and prepare data
    print("Loading and preparing dataset...")
    train_dataset, eval_dataset, tokenizer = load_and_prepare_data(
        model_name=MODEL_CONFIG.model_name,
        dataset_name=args.dataset if args.dataset and not os.path.exists(args.dataset) else None,
        file_path=args.dataset if args.dataset and os.path.exists(args.dataset) else None,
        file_type=args.file_type,
        max_length=TRAINING_CONFIG.max_seq_length,
        format_instructions=args.format_instructions
    )
    
    # Create output directory
    os.makedirs(TRAINING_CONFIG.output_dir, exist_ok=True)
    os.makedirs(TRAINING_CONFIG.logging_dir, exist_ok=True)
    
    # Train
    train(train_dataset, eval_dataset, tokenizer)


if __name__ == "__main__":
    main()

