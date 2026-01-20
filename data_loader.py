"""
Data Loading and Preprocessing Module.

This module handles dataset loading, tokenization, and formatting
for instruction-following fine-tuning tasks.
"""

from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import pandas as pd
from config import TRAINING_CONFIG


class DataProcessor:
    """Handles dataset loading and tokenization for LLM fine-tuning."""
    
    def __init__(
        self,
        model_name: str,
        max_length: int = 2048,
        dataset_name: Optional[str] = None,
        dataset_text_field: str = "text"
    ):
        """
        Initialize the data processor.
        
        Args:
            model_name: Hugging Face model identifier
            max_length: Maximum sequence length for tokenization
            dataset_name: Name of the dataset (HF dataset or path to local file)
            dataset_text_field: Field name containing text in the dataset
        """
        self.model_name = model_name
        self.max_length = max_length
        self.dataset_name = dataset_name or TRAINING_CONFIG.dataset_name
        self.dataset_text_field = dataset_text_field
        self.tokenizer = None
        
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and configure the tokenizer."""
        print(f"Loading tokenizer for {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Set padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Set padding side to right for causal LM
        self.tokenizer.padding_side = "right"
        
        print(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        return self.tokenizer
    
    def load_dataset_from_hf(self, split: str = "train") -> Dataset:
        """
        Load dataset from Hugging Face Hub.
        
        Args:
            split: Dataset split to load
            
        Returns:
            Hugging Face Dataset object
        """
        print(f"Loading dataset: {self.dataset_name} (split: {split})")
        
        try:
            dataset = load_dataset(self.dataset_name, split=split)
            print(f"Dataset loaded. Number of examples: {len(dataset)}")
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def load_dataset_from_file(
        self,
        file_path: str,
        file_type: str = "json"
    ) -> Dataset:
        """
        Load dataset from a local file (JSON or CSV).
        
        Args:
            file_path: Path to the dataset file
            file_type: Type of file ("json" or "csv")
            
        Returns:
            Hugging Face Dataset object
        """
        print(f"Loading dataset from {file_path}...")
        
        try:
            if file_type.lower() == "json":
                # Load JSON file
                if file_path.endswith(".jsonl"):
                    # JSONL format (one JSON object per line)
                    dataset = load_dataset("json", data_files=file_path, split="train")
                else:
                    # Regular JSON format
                    dataset = load_dataset("json", data_files=file_path, split="train")
            elif file_type.lower() == "csv":
                # Load CSV file
                dataset = load_dataset("csv", data_files=file_path, split="train")
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            print(f"Dataset loaded from file. Number of examples: {len(dataset)}")
            return dataset
        except Exception as e:
            print(f"Error loading dataset from file: {e}")
            raise
    
    def format_instruction_data(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Format instruction-following data.
        
        This function formats the dataset entries for instruction tuning.
        For guanaco dataset, it already has the right format, but you can
        customize this for your own datasets.
        
        Expected format for custom datasets:
        {
            "instruction": "...",
            "input": "...",  # Optional
            "output": "..."
        }
        
        Args:
            example: Single dataset example
            
        Returns:
            Formatted text string
        """
        # If dataset already has "text" field formatted, use it directly
        if self.dataset_text_field in example and isinstance(example[self.dataset_text_field], str):
            return {"text": example[self.dataset_text_field]}
        
        # Otherwise, format from instruction/input/output structure
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if input_text:
            formatted_text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            formatted_text = f"""### Instruction:
{instruction}

### Response:
{output}"""
        
        return {"text": formatted_text}
    
    def tokenize_function(
        self,
        examples: Dict[str, Any],
        add_special_tokens: bool = True
    ) -> Dict[str, Any]:
        """
        Tokenize dataset examples.
        
        Args:
            examples: Batch of examples from the dataset
            add_special_tokens: Whether to add special tokens (EOS, etc.)
            
        Returns:
            Tokenized examples with input_ids and attention_mask
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        
        # Get text from dataset
        texts = examples[self.dataset_text_field]
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # We'll pad during batching
            add_special_tokens=add_special_tokens,
            return_overflowing_tokens=False
        )
        
        # For causal LM, labels are the same as input_ids (shifted in trainer)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_dataset(
        self,
        dataset: Dataset,
        format_instructions: bool = False,
        train_split: float = 0.9
    ) -> tuple[Dataset, Optional[Dataset]]:
        """
        Prepare dataset for training by formatting and tokenizing.
        
        Args:
            dataset: Raw dataset
            format_instructions: Whether to format instruction data
            train_split: Fraction of data to use for training
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        print("Preparing dataset...")
        
        # Format instruction data if needed
        if format_instructions:
            print("Formatting instruction data...")
            dataset = dataset.map(
                self.format_instruction_data,
                remove_columns=[col for col in dataset.column_names if col != self.dataset_text_field],
                desc="Formatting instructions"
            )
        
        # Tokenize the dataset
        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        # Split into train/eval if not already split
        if "eval" not in dataset.info.splits if hasattr(dataset, "info") else True:
            dataset_dict = tokenized_dataset.train_test_split(test_size=1 - train_split)
            train_dataset = dataset_dict["train"]
            eval_dataset = dataset_dict["test"]
            print(f"Split dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
        else:
            train_dataset = tokenized_dataset
            eval_dataset = None
            print(f"Using full dataset for training: {len(train_dataset)} examples")
        
        return train_dataset, eval_dataset


def load_and_prepare_data(
    model_name: str,
    dataset_name: Optional[str] = None,
    file_path: Optional[str] = None,
    file_type: str = "json",
    max_length: int = 2048,
    format_instructions: bool = False
) -> tuple[Dataset, Optional[Dataset], PreTrainedTokenizer]:
    """
    Convenience function to load and prepare data in one go.
    
    Args:
        model_name: Model identifier
        dataset_name: HF dataset name (if loading from Hub)
        file_path: Path to local file (if loading from file)
        file_type: Type of local file ("json" or "csv")
        max_length: Maximum sequence length
        format_instructions: Whether to format instruction data
        
    Returns:
        Tuple of (train_dataset, eval_dataset, tokenizer)
    """
    processor = DataProcessor(
        model_name=model_name,
        max_length=max_length,
        dataset_name=dataset_name,
        dataset_text_field=TRAINING_CONFIG.dataset_text_field
    )
    
    # Load tokenizer
    tokenizer = processor.load_tokenizer()
    
    # Load dataset
    if file_path:
        dataset = processor.load_dataset_from_file(file_path, file_type)
    else:
        dataset = processor.load_dataset_from_hf()
    
    # Prepare dataset
    train_dataset, eval_dataset = processor.prepare_dataset(
        dataset,
        format_instructions=format_instructions
    )
    
    return train_dataset, eval_dataset, tokenizer


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...")
    
    train_ds, eval_ds, tokenizer = load_and_prepare_data(
        model_name="meta-llama/Llama-3-8B",
        dataset_name="timdettmers/guanaco-llama2-1k"
    )
    
    print(f"\nTrain dataset size: {len(train_ds)}")
    print(f"Eval dataset size: {len(eval_ds) if eval_ds else 0}")
    print(f"\nSample tokenized example:")
    print(train_ds[0])

