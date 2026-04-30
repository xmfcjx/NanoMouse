"""
LoRA Inference Script
Load base model + LoRA adapter for local inference

Usage:
  python inference_lora.py                           # Interactive chat
  python inference_lora.py --test                    # Run test cases
  python inference_lora.py --merge --output merged/  # Merge and save
"""
import os
import sys
import argparse
import torch
from pathlib import Path

os.environ["TRANSFORMERS_DISABLE_FLASH_ATTN"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


SHORT_SYSTEM_PROMPT = "You are a helpful assistant. Use tools: calc, time, weather, convert, base_convert, solve. Format: Thought/Action/Action Input/Observation/Final Answer."


def load_base_model(model_path, quantization="int4"):
    print(f"[1/2] Loading base model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if quantization == "int4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    return model, tokenizer


def load_lora_adapter(model, adapter_path):
    print(f"[2/2] Loading LoRA adapter: {adapter_path}")
    
    if not os.path.exists(adapter_path):
        print(f"[Error] Adapter not found: {adapter_path}")
        print("Please download adapter from cloud server first:")
        print(f"  scp -r user@server:~/NanoChat-Lab/models/lora_adapter/11gb ./models/lora_adapter/")
        return None
    
    model = PeftModel.from_pretrained(model, adapter_path)
    print("  LoRA adapter loaded successfully")
    return model


def merge_and_save(model, tokenizer, output_path):
    print(f"\n[Merge] Merging LoRA weights...")
    
    merged_model = model.merge_and_unload()
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"  Merged model saved to: {output_path}")
    return merged_model


def generate_response(model, tokenizer, user_input, max_new_tokens=256):
    messages = [
        {"role": "system", "content": SHORT_SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    return response


def run_test_cases(model, tokenizer):
    test_cases = [
        "Calculate 25 * 4",
        "What time is it?",
        "Weather in Beijing",
        "Convert 100km to miles",
        "Convert 255 to hex",
        "Solve 2x + 5 = 15",
        "Calculate 15 + 27",
        "Convert 36 celsius to fahrenheit",
    ]
    
    print("\n" + "=" * 60)
    print("Test Cases")
    print("=" * 60)
    
    for i, question in enumerate(test_cases):
        print(f"\n[{i+1}] Q: {question}")
        response = generate_response(model, tokenizer, question)
        print(f"    A: {response[:200]}{'...' if len(response) > 200 else ''}")


def interactive_chat(model, tokenizer):
    print("\n" + "=" * 60)
    print("Interactive Chat (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            
            response = generate_response(model, tokenizer, user_input)
            print(f"Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="LoRA Inference")
    parser.add_argument("--model_path", type=str, default="models/Qwen2.5-1.5B")
    parser.add_argument("--adapter_path", type=str, default="11gb")
    parser.add_argument("--quantization", type=str, choices=["int4", "fp16"], default="int4")
    parser.add_argument("--test", action="store_true", help="Run test cases")
    parser.add_argument("--merge", action="store_true", help="Merge and save model")
    parser.add_argument("--output", type=str, default="models/merged")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("LoRA Inference")
    print(f"  Base model: {args.model_path}")
    print(f"  Adapter: {args.adapter_path}")
    print(f"  Quantization: {args.quantization}")
    print("=" * 60)
    
    model, tokenizer = load_base_model(args.model_path, args.quantization)
    model = load_lora_adapter(model, args.adapter_path)
    
    if model is None:
        return
    
    if args.merge:
        model = merge_and_save(model, tokenizer, args.output)
    
    if args.test:
        run_test_cases(model, tokenizer)
    else:
        interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
