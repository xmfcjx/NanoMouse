import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class LLM:
    def __init__(self, model_path="models/qwen2.5-0.5b"):
        os.environ["HF_HUB_OFFLINE"] = "1"

        self.model_path = os.path.abspath(model_path)
        print(f"[LLM] Loading model from: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def generate1(self, prompt):

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]

        # 1️⃣ 构造 chat 输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        input_ids = inputs["input_ids"]

        # 2️⃣ 生成（关键参数改动）
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],

            max_new_tokens=128,

            # 🔥 关键：降低发散
            do_sample=True,
            temperature=0.3,
            top_p=0.8,

            repetition_penalty=1.2,

            # 🔥 关键：停止控制
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # 3️⃣ 只取“新生成部分”（核心！！！）
        generated_ids = outputs[0][input_ids.shape[-1]:]

        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        return response.strip()