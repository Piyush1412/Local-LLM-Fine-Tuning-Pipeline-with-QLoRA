"""
Model Merging Script for QLoRA Fine-Tuning.

This script merges a fine-tuned LoRA adapter back into the base model,
creating a single model file that can be used without the adapter.
This is useful for:
1. Exporting models for deployment
2. Creating standalone models for inference
3. Further fine-tuning without LoRA overhead
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse
import os
from pathlib import Path

from config import MODEL_CONFIG


def merge_adapter(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    save_tokenizer: bool = True
):
    """
    Merge LoRA adapter into base model.
    
    Args:
        base_model_name: Base model identifier
        adapter_path: Path to LoRA adapter checkpoint
        output_path: Path to save merged model
        save_tokenizer: Whether to save tokenizer as well
    """
    print("="*60)
    print("MERGING LoRA ADAPTER INTO BASE MODEL")
    print("="*60)
    
    print(f"Base model: {base_model_name}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output path: {output_path}")
    
    # Load adapter configuration to verify
    print("\nLoading adapter configuration...")
    try:
        adapter_config = PeftConfig.from_pretrained(adapter_path)
        print(f"Adapter config loaded: r={adapter_config.r}, alpha={adapter_config.lora_alpha}")
    except Exception as e:
        print(f"Warning: Could not load adapter config: {e}")
    
    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16,
    )
    
    # Merge adapter into base model
    print("\nMerging adapter weights into base model...")
    print("This may take a few minutes depending on model size...")
    
    # Merge the LoRA weights
    merged_model = model.merge_and_unload()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save merged model
    print(f"\nSaving merged model to {output_path}...")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,  # Use safetensors for safer serialization
        max_shard_size="5GB"  # Split into 5GB chunks for large models
    )
    
    # Save tokenizer if requested
    if save_tokenizer:
        print("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(output_path)
    
    # Verify merged model
    print("\nVerifying merged model...")
    try:
        # Try loading the merged model
        test_model = AutoModelForCausalLM.from_pretrained(
            output_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✓ Merged model verified successfully!")
        
        # Print model info
        total_params = sum(p.numel() for p in test_model.parameters())
        trainable_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        del test_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Warning: Could not verify merged model: {e}")
    
    print(f"\n{'='*60}")
    print(f"Merging completed! Model saved to: {output_path}")
    print(f"{'='*60}\n")
    
    # Clean up
    del model
    del merged_model
    del base_model
    torch.cuda.empty_cache()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=MODEL_CONFIG.model_name,
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save merged model"
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Don't save tokenizer"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.adapter_path):
        raise ValueError(f"Adapter path does not exist: {args.adapter_path}")
    
    if not os.path.exists(args.adapter_path + "/adapter_config.json"):
        raise ValueError(
            f"Adapter path does not contain adapter_config.json: {args.adapter_path}\n"
            "Make sure you're pointing to the directory containing the adapter files."
        )
    
    # Merge
    merge_adapter(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        save_tokenizer=not args.no_tokenizer
    )


if __name__ == "__main__":
    main()

