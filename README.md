# End-to-End LLM Fine-Tuning Pipeline using QLoRA

A professional, modular Python system for fine-tuning large language models (LLMs) using QLoRA (Quantized Low-Rank Adaptation). This pipeline supports Llama-3-8B, Mistral-7B, and other compatible models with 4-bit quantization for memory-efficient training.

## 🎯 Features

- **4-bit Quantization (QLoRA)**: Train LLMs with ~75% memory reduction
- **LoRA Adapters**: Fine-tune with ~1% of original parameters (r=64, alpha=16)
- **Gradient Checkpointing**: Trade compute for memory efficiency
- **PagedAdamW Optimizer**: Memory-efficient optimizer for large-scale training
- **SFTTrainer**: Supervised Fine-Tuning trainer from TRL library
- **Experiment Tracking**: Weights & Biases integration for loss visualization
- **Modular Architecture**: Clean separation of concerns (config, data, training, inference)
- **Flexible Dataset Support**: Hugging Face datasets or local JSON/CSV files
- **Gradio Interface**: Web-based UI for model testing (optional)

## 📁 Project Structure

```
.
├── config.py              # Hyperparameters and configuration
├── data_loader.py         # Dataset loading and tokenization
├── train.py              # Main training script
├── inference.py          # Model inference script
├── merge.py              # Merge LoRA adapter into base model
├── gradio_inference.py   # Gradio web interface (optional)
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore file
```

## 🔧 Installation

### Prerequisites

- **GPU**: NVIDIA GPU with CUDA support (recommended: 16GB+ VRAM for Llama-3-8B)
- **Python**: Python 3.10 or higher
- **CUDA**: CUDA 11.8 or 12.1 (depending on your GPU)

### Setup

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install PyTorch with CUDA support**:

For CUDA 11.8:
```bash
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

4. **Install remaining dependencies**:
```bash
pip install -r requirements.txt
```

5. **Set up Weights & Biases** (optional, for experiment tracking):
```bash
wandb login
```

## 🚀 Quick Start

### 1. Configure Hyperparameters

Edit `config.py` to customize:
- Model name (Llama-3-8B, Mistral-7B, etc.)
- LoRA parameters (r, alpha, target_modules)
- Training hyperparameters (learning rate, batch size, epochs)
- Dataset configuration

### 2. Train a Model

**Using a Hugging Face dataset:**
```bash
python train.py --dataset timdettmers/guanaco-llama2-1k
```

**Using a local JSON/CSV file:**
```bash
python train.py --dataset path/to/your/dataset.json --file-type json
```

**With custom instruction formatting:**
```bash
python train.py --dataset your_dataset.json --file-type json --format-instructions
```

### 3. Run Inference

**Interactive mode:**
```bash
python inference.py --adapter-path ./checkpoints
```

**Single instruction:**
```bash
python inference.py --adapter-path ./checkpoints --instruction "Explain quantum computing"
```

**Using merged model:**
```bash
python inference.py --merged-model-path ./merged_model --instruction "Your question here"
```

### 4. Merge Adapter (Optional)

Merge LoRA adapter into base model for standalone deployment:
```bash
python merge.py --adapter-path ./checkpoints --output-path ./merged_model
```

### 5. Launch Gradio Interface (Optional)

Test your model with a web interface:
```bash
python gradio_inference.py --checkpoint-dir ./checkpoints --share
```

## 📊 Understanding QLoRA Parameters

### LoRA Configuration

The default LoRA configuration uses:
- **r=64**: Rank of low-rank matrices. Higher values = more parameters and capacity.
  - For 8B models: r=64 provides a good balance (sufficient capacity without excessive memory)
  - For smaller tasks: r=16 or r=32 may be sufficient
  - For complex tasks: r=128 or higher can help
  
- **alpha=16**: Scaling parameter. Effective learning rate = (alpha/r) * base_lr
  - alpha/r = 16/64 = 0.25, which is a standard scaling factor
  - Higher alpha (relative to r) increases the influence of LoRA updates
  - Common ratios: alpha = r/4 to r/2

- **target_modules**: Which layers to apply LoRA to
  - Default: All attention and MLP layers for maximum capacity
  - Attention-only: ["q_proj", "v_proj", "k_proj", "o_proj"] for faster training
  - Custom selection based on your needs

### Training Configuration

- **Gradient Checkpointing**: Enabled by default. Reduces memory by ~50% at the cost of ~20% slower training
- **PagedAdamW**: Uses 8-bit quantized optimizer states, saving ~50% memory
- **Batch Size**: Default is 4 per device. Increase if you have more VRAM
- **Gradient Accumulation**: Default is 4, giving effective batch size of 16

## 📝 Dataset Format

### Hugging Face Dataset

The pipeline supports any Hugging Face dataset with a text field. Example:
```python
# Dataset should have a 'text' field
{
    "text": "### Instruction: ...\n### Response: ..."
}
```

### Custom Dataset (JSON)

For instruction-following tasks:
```json
{
    "instruction": "Explain quantum computing",
    "input": "",
    "output": "Quantum computing is a type of computation..."
}
```

Or with input context:
```json
{
    "instruction": "Translate to French",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
}
```

### Custom Dataset (CSV)

CSV files should have columns: `instruction`, `input` (optional), `output`.

## 🔍 Experiment Tracking

Training metrics are automatically logged to Weights & Biases. View your experiments:
1. Go to https://wandb.ai
2. Navigate to your project (default: `llm-qlora-finetuning`)
3. Monitor loss curves, learning rate, and other metrics

To disable WandB:
```python
# In config.py
TRAINING_CONFIG.report_to = "none"
```

## 💻 GPU Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 12GB+ VRAM
- **Model**: Mistral-7B or smaller
- **Batch Size**: 2 with gradient accumulation

### Recommended Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4090, A100)
- **Model**: Llama-3-8B or Mistral-7B
- **Batch Size**: 4-8 with gradient accumulation

### Google Colab Setup

For free GPU access on Google Colab:

1. **Change runtime to GPU**: Runtime → Change runtime type → GPU (T4)
2. **Install dependencies**:
```python
!pip install -r requirements.txt
```
3. **Authenticate with Hugging Face** (if using gated models):
```python
from huggingface_hub import login
login(token="your_hf_token")
```
4. **Run training** as usual

## 🎓 Example Workflows

### Workflow 1: Quick Fine-Tuning on Public Dataset

```bash
# 1. Train on guanaco dataset
python train.py --dataset timdettmers/guanaco-llama2-1k

# 2. Test the model
python inference.py --adapter-path ./checkpoints --instruction "What is machine learning?"

# 3. Launch web interface
python gradio_inference.py --checkpoint-dir ./checkpoints
```

### Workflow 2: Custom Dataset Training

```bash
# 1. Prepare your dataset in JSON format
# 2. Train
python train.py --dataset ./data/my_dataset.json --file-type json --format-instructions

# 3. Merge for deployment
python merge.py --adapter-path ./checkpoints --output-path ./production_model

# 4. Use merged model
python inference.py --merged-model-path ./production_model
```

### Workflow 3: Memory-Constrained Training

For GPUs with limited VRAM, edit `config.py`:
```python
TRAINING_CONFIG.per_device_train_batch_size = 1
TRAINING_CONFIG.gradient_accumulation_steps = 8
TRAINING_CONFIG.max_seq_length = 1024
LORA_CONFIG.r = 32  # Reduce LoRA rank
```

## 🔧 Advanced Configuration

### Custom Model Configuration

Edit `config.py` to use different models:
```python
MODEL_CONFIG.model_name = "mistralai/Mistral-7B-v0.1"
```

### Adjust LoRA for Your Task

For instruction-following (default):
```python
LORA_CONFIG.r = 64
LORA_CONFIG.alpha = 16
LORA_CONFIG.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj"]
```

For faster training (attention-only):
```python
LORA_CONFIG.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### Optimize for Your Hardware

High-end GPU (A100, H100):
```python
TRAINING_CONFIG.per_device_train_batch_size = 8
TRAINING_CONFIG.bf16 = True  # Use bfloat16 if supported
TRAINING_CONFIG.gradient_checkpointing = False  # Disable for faster training
```

Low-end GPU (T4, RTX 3060):
```python
TRAINING_CONFIG.per_device_train_batch_size = 1
TRAINING_CONFIG.gradient_accumulation_steps = 16
TRAINING_CONFIG.max_seq_length = 1024
TRAINING_CONFIG.gradient_checkpointing = True
```

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

1. **Reduce batch size**: Set `per_device_train_batch_size = 1`
2. **Increase gradient accumulation**: Compensate with higher `gradient_accumulation_steps`
3. **Reduce sequence length**: Set `max_seq_length = 1024` or lower
4. **Reduce LoRA rank**: Set `LORA_CONFIG.r = 32` or `16`
5. **Enable gradient checkpointing**: Already enabled by default

### Slow Training

1. **Disable gradient checkpointing**: If you have enough memory
2. **Increase batch size**: If VRAM allows
3. **Use bfloat16**: If your GPU supports it (A100, H100)
4. **Reduce LoRA target modules**: Only target attention layers

### Model Not Loading

1. **Check CUDA version**: Ensure PyTorch CUDA matches your GPU driver
2. **Verify model access**: Some models require Hugging Face authentication
3. **Check disk space**: Models can be 10GB+ each

### WandB Errors

If you don't want to use WandB:
```python
# In config.py
TRAINING_CONFIG.report_to = "none"
WANDB_CONFIG.project_name = "none"
```

## 📚 Key Concepts Explained

### QLoRA (Quantized Low-Rank Adaptation)

QLoRA combines:
1. **4-bit Quantization**: Reduces model memory by ~75% with minimal accuracy loss
2. **LoRA Adapters**: Adds trainable low-rank matrices to selected layers
3. **Double Quantization**: Further reduces memory overhead

### LoRA (Low-Rank Adaptation)

Instead of fine-tuning all parameters, LoRA:
- Freezes the base model
- Adds small trainable matrices to attention/MLP layers
- Uses matrix factorization: W = W₀ + BA, where B and A are low-rank
- Typically trains <1% of parameters while achieving comparable performance

### Gradient Checkpointing

Stores only some activations during forward pass, recomputes others during backward pass:
- **Memory**: ~50% reduction
- **Speed**: ~20% slower
- **Trade-off**: Worth it for memory-constrained setups

### PagedAdamW

8-bit quantized optimizer states:
- Reduces optimizer memory by ~50%
- Maintains training stability
- Recommended for all QLoRA training

## 🤝 Contributing

This is a portfolio project, but feel free to:
- Report issues
- Suggest improvements
- Fork and customize for your needs

## 📄 License

This project is provided as-is for educational and portfolio purposes. Please check individual library licenses:
- Transformers: Apache 2.0
- PEFT: Apache 2.0
- TRL: Apache 2.0

## 🙏 Acknowledgments

- Hugging Face for the amazing `transformers`, `peft`, and `trl` libraries
- Tim Dettmers for the QLoRA paper and implementation
- The open-source ML community

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments (extensively documented)
3. Consult the Hugging Face documentation

---

**Happy Fine-Tuning! 🚀**

