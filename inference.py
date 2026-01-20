"""
Inference Script for Fine-Tuned QLoRA Model.

This script loads a fine-tuned LoRA adapter and runs inference
on user prompts. It handles both standalone adapter loading and
merged model loading.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from typing import Optional, List
import argparse

from config import MODEL_CONFIG


class QLoRAInference:
    """Wrapper class for running inference with QLoRA fine-tuned models."""
    
    def __init__(
        self,
        base_model_name: str,
        adapter_path: Optional[str] = None,
        merged_model_path: Optional[str] = None,
        load_in_4bit: bool = True,
        device_map: str = "auto"
    ):
        """
        Initialize inference model.
        
        Args:
            base_model_name: Base model identifier
            adapter_path: Path to LoRA adapter (if using adapter)
            merged_model_path: Path to merged model (if already merged)
            load_in_4bit: Whether to load in 4-bit for memory efficiency
            device_map: Device placement strategy
        """
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.merged_model_path = merged_model_path
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer."""
        print("Loading model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        if self.merged_model_path:
            # Load already merged model
            print(f"Loading merged model from {self.merged_model_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.merged_model_path,
                device_map=self.device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            # Load base model with quantization
            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    quantization_config=bnb_config,
                    device_map=self.device_map,
                    trust_remote_code=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    device_map=self.device_map,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
            
            # Load LoRA adapter
            if self.adapter_path:
                print(f"Loading LoRA adapter from {self.adapter_path}...")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.adapter_path,
                    torch_dtype=torch.float16,
                )
        
        print("Model loaded successfully!")
    
    def format_prompt(
        self,
        instruction: str,
        input_text: Optional[str] = None,
        template: str = "guanaco"
    ) -> str:
        """
        Format instruction prompt.
        
        Args:
            instruction: Instruction text
            input_text: Optional input context
            template: Prompt template format
            
        Returns:
            Formatted prompt string
        """
        if template == "guanaco":
            if input_text:
                return f"""### Human: {instruction}

### Input: {input_text}

### Assistant:"""
            else:
                return f"""### Human: {instruction}

### Assistant:"""
        elif template == "alpaca":
            if input_text:
                return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:"""
            else:
                return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""
        else:
            # Simple template
            return f"{instruction}\n\n"
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        return_full_text: bool = False
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Penalty for repetition
            do_sample: Whether to use sampling
            return_full_text: Whether to return prompt + generated text
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        if return_full_text:
            return generated_text
        else:
            # Remove input prompt from output
            return generated_text[len(prompt):].strip()
    
    def chat(
        self,
        instruction: str,
        input_text: Optional[str] = None,
        **generation_kwargs
    ) -> str:
        """
        Chat interface with instruction formatting.
        
        Args:
            instruction: Instruction text
            input_text: Optional input context
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Model response
        """
        prompt = self.format_prompt(instruction, input_text)
        response = self.generate(prompt, **generation_kwargs)
        return response


def main():
    """Interactive inference CLI."""
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned QLoRA model")
    parser.add_argument(
        "--base-model",
        type=str,
        default=MODEL_CONFIG.model_name,
        help="Base model name"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--merged-model-path",
        type=str,
        default=None,
        help="Path to merged model (if already merged)"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load base model in 4-bit for inference"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Single instruction to process (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    if not args.adapter_path and not args.merged_model_path:
        raise ValueError("Must provide either --adapter-path or --merged-model-path")
    
    # Initialize inference
    inference = QLoRAInference(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        merged_model_path=args.merged_model_path,
        load_in_4bit=args.load_in_4bit
    )
    
    # Load model
    inference.load_model()
    
    # Run inference
    if args.instruction:
        # Single instruction mode
        print("\n" + "="*60)
        print("GENERATING RESPONSE")
        print("="*60)
        print(f"Instruction: {args.instruction}\n")
        
        response = inference.chat(
            args.instruction,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print(f"Response:\n{response}\n")
    else:
        # Interactive mode
        print("\n" + "="*60)
        print("INTERACTIVE INFERENCE MODE")
        print("="*60)
        print("Enter your instructions (type 'quit' to exit)\n")
        
        while True:
            try:
                instruction = input("Instruction: ").strip()
                
                if instruction.lower() in ["quit", "exit", "q"]:
                    break
                
                if not instruction:
                    continue
                
                print("\nGenerating response...")
                response = inference.chat(
                    instruction,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature
                )
                
                print(f"\nResponse:\n{response}\n")
                print("-"*60 + "\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    main()

