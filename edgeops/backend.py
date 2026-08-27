"""Model backend boundary for legacy and future adapter-aware runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class GenerationBackend(ABC):
    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    def generate(self, prompt: str, adapter: Optional[str] = None) -> str:
        raise NotImplementedError


class NullBackend(GenerationBackend):
    @property
    def available(self) -> bool:
        return False

    def generate(self, prompt: str, adapter: Optional[str] = None) -> str:
        raise RuntimeError("No model backend is configured.")


class LegacyLLMBackend(GenerationBackend):
    """Adapter for core.llm.LLM. Adapter selection is recorded but not switched."""

    def __init__(self, llm: object) -> None:
        self.llm = llm

    def generate(self, prompt: str, adapter: Optional[str] = None) -> str:
        return self.llm.generate(prompt)


class PeftMultiAdapterBackend(GenerationBackend):
    """One quantized base model with route-selected PEFT adapters."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an EdgeOps assistant. Use only available evidence and never "
        "issue safety-critical motion commands."
    )

    def __init__(
        self,
        model_path: str,
        adapter_root: str,
        quantization: str = "int4",
        max_new_tokens: int = 160,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_kwargs = {"device_map": "auto", "trust_remote_code": True}
        if quantization == "int4":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "int8":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        else:
            raise ValueError("Unsupported quantization: %s" % quantization)

        base_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        adapter_paths = self._discover_adapters(Path(adapter_root))
        if not adapter_paths:
            raise FileNotFoundError("No PEFT adapters found under %s" % adapter_root)

        first_name = sorted(adapter_paths)[0]
        self.model = PeftModel.from_pretrained(
            base_model,
            str(adapter_paths[first_name]),
            adapter_name=first_name,
        )
        for name, path in sorted(adapter_paths.items()):
            if name != first_name:
                self.model.load_adapter(str(path), adapter_name=name)
        self.model.eval()
        self.adapter_names = set(adapter_paths)
        self.default_adapter = first_name

    @staticmethod
    def _discover_adapters(root: Path):
        v2_tool = root / "route_tool"
        tool_path = (
            v2_tool if (v2_tool / "adapter_config.json").exists() else root / "tool"
        )
        expected = {
            # V2 release layout.
            "tool_adapter": tool_path,
            "safety_adapter": root / "safety_refusal",
            # Legacy experimental layout.
            "rag_adapter": root / "rag",
            "memory_adapter": root / "memory",
            "general_adapter": root / "general",
        }
        return {
            name: path
            for name, path in expected.items()
            if (path / "adapter_config.json").exists()
        }

    def generate(self, prompt: str, adapter: Optional[str] = None) -> str:
        selected = adapter if adapter in self.adapter_names else self.default_adapter
        self.model.set_adapter(selected)
        messages = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        ).strip()
