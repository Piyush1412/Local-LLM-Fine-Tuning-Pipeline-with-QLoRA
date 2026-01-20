"""
Gradio Interface for QLoRA Fine-Tuned Model Inference.

This provides a web-based UI for testing the fine-tuned model.
Run this script after training to interact with your model via a web interface.
"""

import gradio as gr
import argparse
import os
from inference import QLoRAInference

from config import MODEL_CONFIG, TRAINING_CONFIG


def create_gradio_interface(
    base_model_name: str,
    adapter_path: str = None,
    merged_model_path: str = None,
    load_in_4bit: bool = True,
    share: bool = False
):
    """
    Create Gradio interface for model inference.
    
    Args:
        base_model_name: Base model name
        adapter_path: Path to LoRA adapter
        merged_model_path: Path to merged model
        load_in_4bit: Load in 4-bit for memory efficiency
        share: Whether to create a shareable link
    """
    # Initialize inference
    print("Initializing inference model...")
    inference = QLoRAInference(
        base_model_name=base_model_name,
        adapter_path=adapter_path,
        merged_model_path=merged_model_path,
        load_in_4bit=load_in_4bit
    )
    
    # Load model
    print("Loading model... This may take a few minutes...")
    inference.load_model()
    print("Model loaded successfully!")
    
    def generate_response(
        instruction: str,
        input_text: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float
    ):
        """Generate response from user input."""
        if not instruction.strip():
            return "Please enter an instruction."
        
        try:
            response = inference.chat(
                instruction=instruction,
                input_text=input_text if input_text.strip() else None,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0
            )
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create Gradio interface
    with gr.Blocks(title="QLoRA Fine-Tuned Model") as demo:
        gr.Markdown(
            """
            # 🚀 QLoRA Fine-Tuned Model Inference
            
            Interact with your fine-tuned model using the interface below.
            Adjust the generation parameters to control the output behavior.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                instruction = gr.Textbox(
                    label="Instruction",
                    placeholder="Enter your instruction here...",
                    lines=3
                )
                input_text = gr.Textbox(
                    label="Input (Optional)",
                    placeholder="Enter additional context if needed...",
                    lines=2
                )
                
                with gr.Accordion("Generation Parameters", open=False):
                    max_tokens = gr.Slider(
                        label="Max Tokens",
                        minimum=1,
                        maximum=2048,
                        value=512,
                        step=1
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        info="Lower = more deterministic, Higher = more creative"
                    )
                    top_p = gr.Slider(
                        label="Top-p (Nucleus Sampling)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05
                    )
                    top_k = gr.Slider(
                        label="Top-k",
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=1
                    )
                    repetition_penalty = gr.Slider(
                        label="Repetition Penalty",
                        minimum=1.0,
                        maximum=2.0,
                        value=1.1,
                        step=0.1
                    )
                
                submit_btn = gr.Button("Generate", variant="primary")
                clear_btn = gr.Button("Clear")
            
            with gr.Column(scale=2):
                output = gr.Textbox(
                    label="Response",
                    lines=15,
                    placeholder="Response will appear here..."
                )
        
        # Examples
        gr.Markdown("### Example Instructions")
        examples = [
            ["Explain quantum computing in simple terms.", ""],
            ["Write a Python function to calculate factorial.", ""],
            ["Translate the following to French:", "Hello, how are you?"],
            ["Summarize the following article:", "Machine learning is..."],
        ]
        gr.Examples(examples=examples, inputs=[instruction, input_text])
        
        # Event handlers
        submit_btn.click(
            fn=generate_response,
            inputs=[
                instruction,
                input_text,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty
            ],
            outputs=output
        )
        
        instruction.submit(
            fn=generate_response,
            inputs=[
                instruction,
                input_text,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty
            ],
            outputs=output
        )
        
        clear_btn.click(
            fn=lambda: ("", "", ""),
            outputs=[instruction, input_text, output]
        )
    
    # Launch interface
    demo.launch(share=share, server_name="0.0.0.0", server_port=7860)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch Gradio interface for QLoRA model inference"
    )
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
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Use latest checkpoint from training output directory"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit for inference"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a shareable Gradio link"
    )
    
    args = parser.parse_args()
    
    # Determine adapter path
    if args.checkpoint_dir:
        adapter_path = args.checkpoint_dir
        merged_model_path = None
    elif args.merged_model_path:
        adapter_path = None
        merged_model_path = args.merged_model_path
    elif args.adapter_path:
        adapter_path = args.adapter_path
        merged_model_path = None
    else:
        # Default: use training output directory
        adapter_path = TRAINING_CONFIG.output_dir
        merged_model_path = None
    
    if not adapter_path and not merged_model_path:
        raise ValueError(
            "Must provide either --adapter-path, --merged-model-path, "
            "or --checkpoint-dir"
        )
    
    # Launch interface
    create_gradio_interface(
        base_model_name=args.base_model,
        adapter_path=adapter_path,
        merged_model_path=merged_model_path,
        load_in_4bit=args.load_in_4bit,
        share=args.share
    )


if __name__ == "__main__":
    main()

