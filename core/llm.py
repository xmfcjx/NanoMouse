import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:
    def __init__(self, model_path="models/Qwen2.5-1.5B"):
        os.environ["HF_HUB_OFFLINE"] = "1"

        self.model_path = os.path.abspath(model_path)
        print(f"[LLM] Loading model from: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str,max_new_tokens=128) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,   # 先关采样，稳定性更高
                temperature=None,
                top_p=None,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 只截取新生成部分，避免把 system/user/prompt 一起 decode 回来
        new_tokens = outputs[0][input_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return answer