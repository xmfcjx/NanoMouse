import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config.config import get_config


class LLM:
    QUANTIZATION_MODES = ["fp16", "int8", "int4", "fp4", "gguf"]
    
    def __init__(self, model_path="models/Qwen2.5-1.5B", quantization: str = None, adapter_path: str = None):
        os.environ["HF_HUB_OFFLINE"] = "1"

        self.quantization = (quantization or get_config("model.quantization", "fp16")).lower()
        if self.quantization not in self.QUANTIZATION_MODES:
            raise ValueError(f"不支持的量化模式: {quantization}，可选: {self.QUANTIZATION_MODES}")

        self.model_path = os.path.abspath(model_path)
        self.adapter_path = adapter_path
        print(f"[LLM] Loading model from: {self.model_path}")
        print(f"[LLM] Quantization mode: {self.quantization.upper()}")
        if adapter_path:
            print(f"[LLM] LoRA adapter: {adapter_path}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.default_max_new_tokens = get_config("model.max_new_tokens", 128)
        self.gguf_ctx = get_config("model.gguf_ctx", 512)
        self.repetition_penalty = get_config("model.repetition_penalty", 1.05)
        
        if self.quantization == "gguf":
            self._load_gguf_model()
        else:
            self._load_transformers_model()
        
        self.default_stop_strings = ["\nObservation:", "\nQuestion:", "\nHuman:", "\n\n\n"]
    
    def _load_transformers_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        if self.quantization == "fp16":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif self.quantization == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto"
            )
        elif self.quantization == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto"
            )
        elif self.quantization == "fp4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto"
            )
        
        self.model.eval()

        if self.adapter_path and os.path.exists(self.adapter_path):
            print(f"[LLM] Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model.eval()
            print(f"[LLM] LoRA adapter loaded successfully")

        self._gguf_llm = None
    
    def _load_gguf_model(self):
        gguf_path = self.model_path
        if os.path.isdir(gguf_path):
            gguf_files = [f for f in os.listdir(gguf_path) if f.endswith('.gguf')]
            if gguf_files:
                gguf_path = os.path.join(gguf_path, gguf_files[0])
            else:
                raise FileNotFoundError(f"在 {self.model_path} 中未找到 GGUF 文件")
        
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF 文件不存在: {gguf_path}")
        
        try:
            from llama_cpp import Llama
            self._gguf_llm = Llama(
                model_path=gguf_path,
                n_ctx=self.gguf_ctx,
                n_gpu_layers=-1,
                verbose=False
            )
            self.model = None
            self.tokenizer = None
            print(f"[LLM] GGUF model loaded: {gguf_path}")
        except ImportError:
            raise ImportError("需要安装 llama-cpp-python: pip install llama-cpp-python")
    
    def generate(self, prompt: str, max_new_tokens: int = None, stop_strings=None, use_chat_template: bool = True) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        if self.quantization == "gguf":
            return self._generate_gguf(prompt, max_new_tokens)
        return self._generate_transformers(prompt, max_new_tokens, stop_strings, use_chat_template)
    
    def _generate_gguf(self, prompt: str, max_new_tokens: int) -> str:
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        response = self._gguf_llm(
            full_prompt,
            max_tokens=max_new_tokens,
            stop=["<|im_end|>"],
            echo=False
        )
        return response["choices"][0]["text"].strip()
    
    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

    LORA_SYSTEM_PROMPT = """You are a helpful assistant. You MUST use tools to answer questions.

Tools:
- calc: Python math. Input: print(expr). Example: print(3*8)
- time: Current time. No input.
- weather: City weather. Input: city name.
- convert: Unit convert. Input: value+unit to target. Example: 100km to miles
- base_convert: Base convert. Input: number to base. Example: 255 to hex
- solve: Equations. Input: equation. Example: 3*x+1=10

Format: Thought: / Action: / Action Input: / Final Answer:
For multi-step tasks, call tools one at a time."""

    def _generate_transformers(self, prompt: str, max_new_tokens: int, stop_strings, use_chat_template: bool) -> str:
        if use_chat_template:
            system_content = self.LORA_SYSTEM_PROMPT if self.adapter_path else self.DEFAULT_SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        stops = stop_strings if stop_strings is not None else self.default_stop_strings

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stop_strings=stops,
                tokenizer=self.tokenizer,
            )

        new_tokens = outputs[0][input_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return answer
    
    @staticmethod
    def get_recommended_quantization(vram_gb: float) -> str:
        if vram_gb >= 8:
            return "fp16"
        elif vram_gb >= 4:
            return "int4"
        else:
            return "gguf"
