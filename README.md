# Local-LLM-Fine-Tuning-Pipeline-with-QLoRA
A production-grade pipeline for fine-tuning LLMs (Llama-3, Mistral) using QLoRA. Features modular architecture, experiment tracking with WandB, and automated adapter merging.

# 🧠 NeuroTune: End-to-End QLoRA Fine-Tuning Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PEFT](https://img.shields.io/badge/PEFT-State%20of%20the%20art-green)](https://github.com/huggingface/peft)

## 📖 Overview
**NeuroTune** is a modular, high-performance Machine Learning pipeline designed to fine-tune 7B+ parameter Large Language Models (LLMs) on consumer-grade GPUs. 

Leveraging **QLoRA (Quantized Low-Rank Adaptation)**, this project demonstrates how to fine-tune massive models like **Llama-3-8B** or **Mistral-7B** with minimal VRAM usage (under 16GB) while maintaining full 16-bit finetuning performance. It moves beyond simple notebooks, offering a structured, CLI-based architecture suitable for production workflows.

## 🚀 Key Features (Advanced)

* **⚡ 4-Bit Quantization:** Implements `bitsandbytes` NF4 quantization to load large models into limited GPU memory without significant accuracy loss.
* **🔧 Modular Architecture:** Decoupled logic for data loading, training configurations, and inference, avoiding "spaghetti code."
* **📊 Experiment Tracking:** Integrated **Weights & Biases (WandB)** for real-time visualization of loss curves, learning rates, and gradient norms.
* **🛠️ Optimized Training:** Utilizes **Gradient Checkpointing** and **PagedAdamW** optimizer to prevent OOM (Out of Memory) errors during backpropagation.
* **🔄 Adapter Merging:** Includes standalone scripts to merge LoRA adapters back into the base model for standalone deployment.

## 📂 Project Structure

```bash
NeuroTune/
├── config/
│   └── config.py          # Hyperparameters (Learning rate, Batch size, LoRA rank)
├── src/
│   ├── data_loader.py     # Custom dataset processing & tokenization
│   ├── model.py           # Model loading with 4-bit quantization config
│   └── trainer.py         # SFTTrainer setup with PEFT config
├── scripts/
│   ├── train.py           # Main entry point for training
│   ├── inference.py       # Test the model with prompt injection
│   └── merge_adapters.py  # Export final model for deployment
├── requirements.txt       # Optimized dependencies for CUDA environment
└── README.md
