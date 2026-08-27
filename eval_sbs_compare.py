"""
SBS (Side-by-Side) 对比评估脚本
对比 [原生模型 + 长 Prompt] vs [微调模型 + 短 Prompt]
用法:
  python eval_sbs_compare.py                    # 完整评估
  python eval_sbs_compare.py --quick --n 20    # 快速测试
"""
import os
import sys
import json
import time
import argparse
import datetime
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

RESULTS_DIR = Path("eval/results/sbs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DPO_OUTPUT_PATH = Path("data/dpo_pairs.jsonl")

PREFERENCE_WEIGHTS = {
    "tool_correct": 0.40,
    "answer_correct": 0.30,
    "format_correct": 0.20,
    "conciseness": 0.10,
}

SCORE_THRESHOLD = 0.1


LONG_SYSTEM_PROMPT = """You are a helpful AI assistant with access to the following tools:

Tools:
- calc: Execute Python math expressions. Input must be valid Python code. Use print() for results. Example: 'print(3 * 8)'
- time: Get current date and time. No input needed.
- weather: Query city weather. Input city name. Example: 'Beijing'
- convert: Unit conversion. Supports length, weight, temperature. Format: 'value+unit to target'. Example: '100km to miles'
- base_convert: Number base conversion. Format: 'number to base'. Example: '255 to hex'
- solve: Solve equations. Supports linear and quadratic. Example: '3*x*x=27'

Rules:
1. You MUST use tools to answer questions. Never answer from memory.
2. If unsure which tool to use, output: Final Answer: I don't know
3. For multi-step tasks, call tools one at a time.

Format:
Thought: Analyze the problem
Action: tool_name
Action Input: tool_input

Example:
Question: calculate 25 * 4
Thought: I need to compute a mathematical expression
Action: calc
Action Input: print(25 * 4)
Observation: 100
Thought: The calculation is complete
Final Answer: 100"""


SHORT_SYSTEM_PROMPT = """You are a helpful assistant. You MUST use tools to answer questions.

Tools:
- calc: Python math. Input: print(expr). Example: print(3*8)
- time: Current time. No input.
- weather: City weather. Input: city name.
- convert: Unit convert. Input: value+unit to target. Example: 100km to miles
- base_convert: Base convert. Input: number to base. Example: 255 to hex
- solve: Equations. Input: equation. Example: 3*x+1=10

Format: Thought: / Action: / Action Input: / Final Answer:
For multi-step tasks, call tools one at a time."""


TEST_CASES = [
    {"question": "Calculate 25 * 4", "expected_tool": "calc", "expected_answer": "100", "type": "single_tool"},
    {"question": "What time is it?", "expected_tool": "time", "expected_answer": None, "type": "single_tool"},
    {"question": "Weather in Beijing", "expected_tool": "weather", "expected_answer": None, "type": "single_tool"},
    {"question": "Convert 100km to miles", "expected_tool": "convert", "expected_answer": "62.14", "type": "single_tool"},
    {"question": "Convert 255 to hex", "expected_tool": "base_convert", "expected_answer": "FF", "type": "single_tool"},
    {"question": "Solve 2x + 5 = 15", "expected_tool": "solve", "expected_answer": "x = 5", "type": "single_tool"},
    {"question": "Calculate 15 + 27", "expected_tool": "calc", "expected_answer": "42", "type": "single_tool"},
    {"question": "Convert 36 celsius to fahrenheit", "expected_tool": "convert", "expected_answer": "96.8", "type": "single_tool"},
    {"question": "Convert 64 to binary", "expected_tool": "base_convert", "expected_answer": "1000000", "type": "single_tool"},
    {"question": "Calculate 2 ** 10", "expected_tool": "calc", "expected_answer": "1024", "type": "single_tool"},
    {"question": "What is the weather in Tokyo?", "expected_tool": "weather", "expected_answer": None, "type": "single_tool"},
    {"question": "Convert 50 fahrenheit to celsius", "expected_tool": "convert", "expected_answer": "10.0", "type": "single_tool"},
    {"question": "Calculate 100 / 4", "expected_tool": "calc", "expected_answer": "25", "type": "single_tool"},
    {"question": "Convert 10 kg to pounds", "expected_tool": "convert", "expected_answer": "22.05", "type": "single_tool"},
    {"question": "Solve 3x = 12", "expected_tool": "solve", "expected_answer": "x = 4", "type": "single_tool"},
    {"question": "Calculate sqrt(144)", "expected_tool": "calc", "expected_answer": "12", "type": "single_tool"},
    {"question": "Convert 1010 binary to decimal", "expected_tool": "base_convert", "expected_answer": "10", "type": "single_tool"},
    {"question": "What's the weather like in London?", "expected_tool": "weather", "expected_answer": None, "type": "single_tool"},
    {"question": "Calculate (7 + 3) * 5", "expected_tool": "calc", "expected_answer": "50", "type": "single_tool"},
    {"question": "Convert 1 mile to km", "expected_tool": "convert", "expected_answer": "1.61", "type": "single_tool"},
    {"question": "What is 100 divided by 4?", "expected_tool": "calc", "expected_answer": "25", "type": "single_tool"},
    {"question": "F2C 100", "expected_tool": "convert", "expected_answer": "37.8", "type": "single_tool"},
    {"question": "Calculate 88 mod 7", "expected_tool": "calc", "expected_answer": "4", "type": "single_tool"},
    {"question": "Convert 200 kg to pounds", "expected_tool": "convert", "expected_answer": "440.92", "type": "single_tool"},
    {"question": "Solve 5x - 3 = 22", "expected_tool": "solve", "expected_answer": "x = 5", "type": "single_tool"},
    {"question": "Convert 128 to octal", "expected_tool": "base_convert", "expected_answer": "200", "type": "single_tool"},
    {"question": "Weather in Sydney", "expected_tool": "weather", "expected_answer": None, "type": "single_tool"},
    {"question": "Calculate 3 to the power of 5", "expected_tool": "calc", "expected_answer": "243", "type": "single_tool"},
    {"question": "Convert 0 celsius to fahrenheit", "expected_tool": "convert", "expected_answer": "32", "type": "single_tool"},
    
    {"question": "What is the capital of France?", "expected_tool": None, "expected_answer": "Paris", "type": "no_tool"},
    {"question": "Who wrote Romeo and Juliet?", "expected_tool": None, "expected_answer": "Shakespeare", "type": "no_tool"},
    {"question": "What is the largest planet in our solar system?", "expected_tool": None, "expected_answer": "Jupiter", "type": "no_tool"},
    {"question": "What is the chemical symbol for gold?", "expected_tool": None, "expected_answer": "Au", "type": "no_tool"},
    {"question": "Who painted the Mona Lisa?", "expected_tool": None, "expected_answer": "Leonardo da Vinci", "type": "no_tool"},
    {"question": "What is the capital of Japan?", "expected_tool": None, "expected_answer": "Tokyo", "type": "no_tool"},
    {"question": "How many continents are there?", "expected_tool": None, "expected_answer": "7", "type": "no_tool"},
    {"question": "What is 2 + 2?", "expected_tool": None, "expected_answer": "4", "type": "no_tool"},
    {"question": "Hello!", "expected_tool": None, "expected_answer": None, "type": "no_tool"},
    
    {"question": "Convert 100°F to Celsius, then add 10 to the result", "expected_tool": "convert", "expected_answer": "47.8", "type": "multi_turn"},
    {"question": "Calculate 25 * 4, then convert that number to binary", "expected_tool": "calc", "expected_answer": "1100100", "type": "multi_turn"},
    {"question": "Solve 5x = 25, then convert x to hexadecimal", "expected_tool": "solve", "expected_answer": "5", "type": "multi_turn"},
    
    {"question": "Translate 'hello' to Chinese", "expected_tool": None, "expected_answer": None, "type": "reject"},
    {"question": "Write a poem about the ocean", "expected_tool": None, "expected_answer": None, "type": "reject"},
    {"question": "What's the weather like today?", "expected_tool": None, "expected_answer": None, "type": "reject"},
    {"question": "Tell me a joke", "expected_tool": None, "expected_answer": None, "type": "reject"},
    {"question": "What is the meaning of life?", "expected_tool": None, "expected_answer": None, "type": "reject"},
]


def load_model(model_path, adapter_path=None, quantization="int4"):
    print(f"  加载模型: {model_path}")
    
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
    
    if adapter_path and os.path.exists(adapter_path):
        print(f"  加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer


def generate_response(model, tokenizer, system_prompt, user_input, max_new_tokens=128):
    messages = [
        {"role": "system", "content": system_prompt},
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


def extract_tool_and_answer(response):
    tool = None
    answer = None
    
    action_match = re.search(r"Action:\s*(\w+)", response, re.IGNORECASE)
    if action_match:
        tool = action_match.group(1).lower()
    
    final_match = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
    if final_match:
        answer = final_match.group(1).strip()
    
    return tool, answer


def evaluate_response(response, expected_tool, expected_answer):
    tool, answer = extract_tool_and_answer(response)
    
    tool_correct = tool == expected_tool if expected_tool else True
    
    if expected_answer:
        answer_correct = expected_answer.lower() in answer.lower() if answer else False
    else:
        answer_correct = True
    
    format_correct = "Action:" in response or "Final Answer:" in response
    
    return {
        "tool_correct": tool_correct,
        "answer_correct": answer_correct,
        "format_correct": format_correct,
        "extracted_tool": tool,
        "extracted_answer": answer,
    }


def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))


def score_response(eval_result, response_text, tokenizer):
    scores = {
        "tool_correct": 1.0 if eval_result["tool_correct"] else 0.0,
        "answer_correct": 1.0 if eval_result["answer_correct"] else 0.0,
        "format_correct": 1.0 if eval_result["format_correct"] else 0.0,
        "conciseness": 1.0,
    }
    token_count = count_tokens(response_text, tokenizer) if response_text else 999
    base_score = sum(PREFERENCE_WEIGHTS[k] * scores[k] for k in PREFERENCE_WEIGHTS)
    return base_score, scores, token_count


def compute_preference(base_score, base_scores, base_tokens,
                       ft_score, ft_scores, ft_tokens, threshold=SCORE_THRESHOLD):
    max_tokens = max(base_tokens, ft_tokens, 1)
    base_conciseness = 1.0 - (base_tokens / (max_tokens + 1))
    ft_conciseness = 1.0 - (ft_tokens / (max_tokens + 1))

    base_final = base_score - PREFERENCE_WEIGHTS["conciseness"] * base_scores["conciseness"] + PREFERENCE_WEIGHTS["conciseness"] * base_conciseness
    ft_final = ft_score - PREFERENCE_WEIGHTS["conciseness"] * ft_scores["conciseness"] + PREFERENCE_WEIGHTS["conciseness"] * ft_conciseness

    diff = ft_final - base_final

    if abs(diff) < threshold:
        avg = (base_final + ft_final) / 2
        if avg >= 0.7:
            return "tied_good", base_final, ft_final
        else:
            return "tied_bad", base_final, ft_final
    elif diff > 0:
        return "chosen_ft", base_final, ft_final
    else:
        return "chosen_base", base_final, ft_final


def build_dpo_pair(question, chosen_response, rejected_response):
    return {
        "prompt": [
            {"role": "system", "content": SHORT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "chosen": [{"role": "assistant", "content": chosen_response}],
        "rejected": [{"role": "assistant", "content": rejected_response}],
    }


def generate_preference_report(dpo_pairs, preferences):
    total = len(preferences)
    ft_wins = sum(1 for p in preferences if p["result"] == "chosen_ft")
    base_wins = sum(1 for p in preferences if p["result"] == "chosen_base")
    tied_good = sum(1 for p in preferences if p["result"] == "tied_good")
    tied_bad = sum(1 for p in preferences if p["result"] == "tied_bad")

    dim_wins = {"tool_correct": 0, "answer_correct": 0, "format_correct": 0}
    for p in preferences:
        for dim in dim_wins:
            if p["ft_scores"][dim] > p["base_scores"][dim]:
                dim_wins[dim] += 1

    report = {
        "total_cases": total,
        "valid_dpo_pairs": len(dpo_pairs),
        "preference_distribution": {
            "finetuned_wins": ft_wins,
            "base_wins": base_wins,
            "tied_good": tied_good,
            "tied_bad": tied_bad,
        },
        "win_rates": {
            "finetuned": round(ft_wins / total * 100, 1) if total else 0,
            "base": round(base_wins / total * 100, 1) if total else 0,
            "tied": round((tied_good + tied_bad) / total * 100, 1) if total else 0,
        },
        "per_dimension_ft_wins": {k: v for k, v in dim_wins.items()},
        "avg_score_diff": round(
            sum(p["ft_score"] - p["base_score"] for p in preferences) / total, 4
        ) if total else 0,
    }
    return report


def load_test_cases(jsonl_path=None):
    if jsonl_path and os.path.exists(jsonl_path):
        cases = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        print(f"  从 {jsonl_path} 加载 {len(cases)} 条测试用例")
        return cases
    return TEST_CASES


def run_sbs_comparison(model_path=None, adapter_path=None, quick_test=False, test_n=None, enable_dpo=True, verify_mode=False, test_cases_path=None):
    print("\n" + "=" * 70)
    print("SBS (Side-by-Side) 对比评估")
    if enable_dpo:
        print("  [DPO 偏好数据生成: 已启用]")
    if verify_mode:
        print("  [验证模式: 微调模型也用长Prompt测试]")
    print("=" * 70)

    if model_path is None:
        model_path = "models/Qwen2.5-1.5B"

    test_cases = load_test_cases(test_cases_path)
    if quick_test and test_n:
        test_cases = test_cases[:test_n]

    print(f"\n测试用例数: {len(test_cases)}")

    print("\n[1/2] 加载原生模型...")
    base_model, tokenizer = load_model(model_path, quantization="int4")

    print("\n[2/2] 加载微调模型...")
    if adapter_path and os.path.exists(adapter_path):
        finetuned_model, _ = load_model(model_path, adapter_path, quantization="int4")
    else:
        print("  未找到 adapter，使用原生模型模拟微调效果")
        finetuned_model = base_model

    results = []
    dpo_pairs = []
    preferences = []

    type_results = {"single_tool": [], "no_tool": [], "multi_turn": [], "reject": [], "base_convert": [], "solve": []}

    long_prompt_tokens = count_tokens(LONG_SYSTEM_PROMPT, tokenizer)
    short_prompt_tokens = count_tokens(SHORT_SYSTEM_PROMPT, tokenizer)
    token_savings = long_prompt_tokens - short_prompt_tokens
    token_savings_percent = (token_savings / long_prompt_tokens) * 100

    print(f"\nPrompt Token 对比:")
    print(f"  长 Prompt: {long_prompt_tokens} tokens")
    print(f"  短 Prompt: {short_prompt_tokens} tokens")
    print(f"  节省: {token_savings} tokens ({token_savings_percent:.1f}%)")

    print("\n开始评估...")

    for i, case in enumerate(test_cases):
        question = case["question"]
        expected_tool = case["expected_tool"]
        expected_answer = case["expected_answer"]
        case_type = case.get("type", "single_tool")

        print(f"\n[{i+1}/{len(test_cases)}] [{case_type}] {question}")

        start = time.time()
        base_response = generate_response(base_model, tokenizer, LONG_SYSTEM_PROMPT, question)
        base_latency = time.time() - start
        base_eval = evaluate_response(base_response, expected_tool, expected_answer)

        start = time.time()
        ft_response = generate_response(finetuned_model, tokenizer, SHORT_SYSTEM_PROMPT, question)
        ft_latency = time.time() - start
        ft_eval = evaluate_response(ft_response, expected_tool, expected_answer)

        if verify_mode:
            start = time.time()
            ft_long_response = generate_response(finetuned_model, tokenizer, LONG_SYSTEM_PROMPT, question)
            ft_long_latency = time.time() - start
            ft_long_eval = evaluate_response(ft_long_response, expected_tool, expected_answer)

        result = {
            "id": i + 1,
            "question": question,
            "expected_tool": expected_tool,
            "expected_answer": expected_answer,
            "base_model": {
                "response": base_response[:200],
                "tool_correct": base_eval["tool_correct"],
                "answer_correct": base_eval["answer_correct"],
                "format_correct": base_eval["format_correct"],
                "latency": round(base_latency, 3),
            },
            "finetuned_model": {
                "response": ft_response[:200],
                "tool_correct": ft_eval["tool_correct"],
                "answer_correct": ft_eval["answer_correct"],
                "format_correct": ft_eval["format_correct"],
                "latency": round(ft_latency, 3),
            },
        }

        if verify_mode:
            result["finetuned_model_long"] = {
                "response": ft_long_response[:200],
                "tool_correct": ft_long_eval["tool_correct"],
                "answer_correct": ft_long_eval["answer_correct"],
                "format_correct": ft_long_eval["format_correct"],
                "latency": round(ft_long_latency, 3),
            }

        if enable_dpo:
            base_score, base_scores, base_tokens = score_response(base_eval, base_response, tokenizer)
            ft_score, ft_scores, ft_tokens = score_response(ft_eval, ft_response, tokenizer)

            pref_result, base_final, ft_final = compute_preference(
                base_score, base_scores, base_tokens,
                ft_score, ft_scores, ft_tokens,
            )

            pref_record = {
                "id": i + 1,
                "question": question,
                "result": pref_result,
                "base_score": round(base_final, 4),
                "ft_score": round(ft_final, 4),
                "base_scores": {k: round(v, 2) for k, v in base_scores.items()},
                "ft_scores": {k: round(v, 2) for k, v in ft_scores.items()},
            }
            preferences.append(pref_record)

            if pref_result == "chosen_ft":
                dpo_pairs.append(build_dpo_pair(question, ft_response, base_response))
            elif pref_result == "chosen_base":
                dpo_pairs.append(build_dpo_pair(question, base_response, ft_response))

            result["preference"] = pref_result
            result["preference_scores"] = {"base": round(base_final, 4), "ft": round(ft_final, 4)}

            pref_symbol = {"chosen_ft": "FT>", "chosen_base": "BASE>", "tied_good": "=+", "tied_bad": "=-"}
            print(f"  偏好: [{pref_symbol[pref_result]}] base={base_final:.3f} ft={ft_final:.3f}")

        results.append(result)
        type_results[case_type].append(result)

        print(f"  原生模型(长): 工具{'✓' if base_eval['tool_correct'] else '✗'} 答案{'✓' if base_eval['answer_correct'] else '✗'} | {base_latency:.2f}s")
        print(f"  微调模型(短): 工具{'✓' if ft_eval['tool_correct'] else '✗'} 答案{'✓' if ft_eval['answer_correct'] else '✗'} | {ft_latency:.2f}s")
        if verify_mode:
            print(f"  微调模型(长): 工具{'✓' if ft_long_eval['tool_correct'] else '✗'} 答案{'✓' if ft_long_eval['answer_correct'] else '✗'} | {ft_long_latency:.2f}s")

    def calc_type_stats(type_name, results_list):
        if not results_list:
            return None
        return {
            "count": len(results_list),
            "ft_tool_acc": sum(1 for r in results_list if r["finetuned_model"]["tool_correct"]) / len(results_list) * 100,
            "ft_answer_acc": sum(1 for r in results_list if r["finetuned_model"]["answer_correct"]) / len(results_list) * 100,
            "base_tool_acc": sum(1 for r in results_list if r["base_model"]["tool_correct"]) / len(results_list) * 100,
            "base_answer_acc": sum(1 for r in results_list if r["base_model"]["answer_correct"]) / len(results_list) * 100,
        }

    type_stats = {t: calc_type_stats(t, r) for t, r in type_results.items() if r}

    base_tool_acc = sum(1 for r in results if r["base_model"]["tool_correct"]) / len(results) * 100
    base_answer_acc = sum(1 for r in results if r["base_model"]["answer_correct"]) / len(results) * 100
    base_format_acc = sum(1 for r in results if r["base_model"]["format_correct"]) / len(results) * 100
    base_avg_latency = sum(r["base_model"]["latency"] for r in results) / len(results)

    ft_tool_acc = sum(1 for r in results if r["finetuned_model"]["tool_correct"]) / len(results) * 100
    ft_answer_acc = sum(1 for r in results if r["finetuned_model"]["answer_correct"]) / len(results) * 100
    ft_format_acc = sum(1 for r in results if r["finetuned_model"]["format_correct"]) / len(results) * 100
    ft_avg_latency = sum(r["finetuned_model"]["latency"] for r in results) / len(results)

    if verify_mode:
        ft_long_tool_acc = sum(1 for r in results if r["finetuned_model_long"]["tool_correct"]) / len(results) * 100
        ft_long_answer_acc = sum(1 for r in results if r["finetuned_model_long"]["answer_correct"]) / len(results) * 100
        ft_long_format_acc = sum(1 for r in results if r["finetuned_model_long"]["format_correct"]) / len(results) * 100
        ft_long_avg_latency = sum(r["finetuned_model_long"]["latency"] for r in results) / len(results)

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "test_cases": len(results),
        "verify_mode": verify_mode,
        "prompt_comparison": {
            "long_prompt_tokens": long_prompt_tokens,
            "short_prompt_tokens": short_prompt_tokens,
            "token_savings": token_savings,
            "token_savings_percent": round(token_savings_percent, 1),
        },
        "base_model": {
            "tool_accuracy": round(base_tool_acc, 1),
            "answer_accuracy": round(base_answer_acc, 1),
            "format_accuracy": round(base_format_acc, 1),
            "avg_latency": round(base_avg_latency, 3),
        },
        "finetuned_model": {
            "tool_accuracy": round(ft_tool_acc, 1),
            "answer_accuracy": round(ft_answer_acc, 1),
            "format_accuracy": round(ft_format_acc, 1),
            "avg_latency": round(ft_avg_latency, 3),
        },
        "improvement": {
            "tool_accuracy": round(ft_tool_acc - base_tool_acc, 1),
            "answer_accuracy": round(ft_answer_acc - base_answer_acc, 1),
            "format_accuracy": round(ft_format_acc - base_format_acc, 1),
        },
        "detailed_results": results,
    }

    if verify_mode:
        summary["finetuned_model_long"] = {
            "tool_accuracy": round(ft_long_tool_acc, 1),
            "answer_accuracy": round(ft_long_answer_acc, 1),
            "format_accuracy": round(ft_long_format_acc, 1),
            "avg_latency": round(ft_long_avg_latency, 3),
        }
        summary["diagnosis"] = {
            "ft_long_vs_base": {
                "tool_diff": round(ft_long_tool_acc - base_tool_acc, 1),
                "answer_diff": round(ft_long_answer_acc - base_answer_acc, 1),
                "conclusion": "微调模型能力正常" if ft_long_tool_acc >= base_tool_acc * 0.8 else "微调可能有问题",
            },
            "ft_long_vs_ft_short": {
                "tool_diff": round(ft_long_tool_acc - ft_tool_acc, 1),
                "answer_diff": round(ft_long_answer_acc - ft_answer_acc, 1),
                "conclusion": "短Prompt导致性能下降" if ft_long_tool_acc > ft_tool_acc else "Prompt长度影响不大",
            },
        }

    output_file = RESULTS_DIR / "sbs_comparison_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if enable_dpo and preferences:
        DPO_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DPO_OUTPUT_PATH, "w", encoding="utf-8") as f:
            for pair in dpo_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        pref_report = generate_preference_report(dpo_pairs, preferences)
        pref_report["timestamp"] = datetime.datetime.now().isoformat()
        pref_report_file = RESULTS_DIR / "dpo_preference_report.json"
        with open(pref_report_file, "w", encoding="utf-8") as f:
            json.dump(pref_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)

    if verify_mode:
        print(f"\n{'指标':<20} {'原生(长)':<15} {'微调(短)':<15} {'微调(长)':<15}")
        print("-" * 65)
        print(f"{'工具识别准确率':<20} {base_tool_acc:.1f}%{'':<10} {ft_tool_acc:.1f}%{'':<10} {ft_long_tool_acc:.1f}%")
        print(f"{'答案准确率':<20} {base_answer_acc:.1f}%{'':<10} {ft_answer_acc:.1f}%{'':<10} {ft_long_answer_acc:.1f}%")
        print(f"{'格式正确率':<20} {base_format_acc:.1f}%{'':<10} {ft_format_acc:.1f}%{'':<10} {ft_long_format_acc:.1f}%")
        
        print("\n" + "=" * 70)
        print("诊断分析")
        print("=" * 70)
        print(f"\n  [对比1] 微调(长) vs 原生(长):")
        print(f"    工具准确率差异: {ft_long_tool_acc - base_tool_acc:+.1f}%")
        print(f"    结论: {summary['diagnosis']['ft_long_vs_base']['conclusion']}")
        print(f"\n  [对比2] 微调(长) vs 微调(短):")
        print(f"    工具准确率差异: {ft_long_tool_acc - ft_tool_acc:+.1f}%")
        print(f"    结论: {summary['diagnosis']['ft_long_vs_ft_short']['conclusion']}")
    else:
        print(f"\n{'指标':<25} {'原生模型(长Prompt)':<20} {'微调模型(短Prompt)':<20} {'提升':<10}")
        print("-" * 75)
        print(f"{'工具识别准确率':<25} {base_tool_acc:.1f}%{'':<15} {ft_tool_acc:.1f}%{'':<15} {ft_tool_acc - base_tool_acc:+.1f}%")
        print(f"{'答案准确率':<25} {base_answer_acc:.1f}%{'':<15} {ft_answer_acc:.1f}%{'':<15} {ft_answer_acc - base_answer_acc:+.1f}%")
        print(f"{'格式正确率':<25} {base_format_acc:.1f}%{'':<15} {ft_format_acc:.1f}%{'':<15} {ft_format_acc - base_format_acc:+.1f}%")
        print(f"{'平均延迟':<25} {base_avg_latency:.2f}s{'':<15} {ft_avg_latency:.2f}s{'':<15} {ft_avg_latency - base_avg_latency:+.2f}s")

    print("\n" + "=" * 70)
    print("分类型统计")
    print("=" * 70)
    type_names = {"single_tool": "单步工具调用", "no_tool": "无需工具", "multi_turn": "多步推理", "reject": "拒绝样本", "base_convert": "进制转换", "solve": "方程求解"}
    for t, stats in type_stats.items():
        if stats:
            print(f"\n  [{type_names.get(t, t)}] {stats['count']} 条")
            print(f"    微调模型: 工具准确率 {stats['ft_tool_acc']:.1f}% | 答案准确率 {stats['ft_answer_acc']:.1f}%")
            print(f"    原生模型: 工具准确率 {stats['base_tool_acc']:.1f}% | 答案准确率 {stats['base_answer_acc']:.1f}%")

    print(f"\n{'Prompt Token 节省':<25} {token_savings} tokens ({token_savings_percent:.1f}%)")

    if enable_dpo and preferences:
        print("\n" + "=" * 70)
        print("DPO 偏好对比结果")
        print("=" * 70)
        print(f"  有效偏好对: {len(dpo_pairs)} / {len(preferences)}")
        print(f"  微调模型胜: {pref_report['preference_distribution']['finetuned_wins']} ({pref_report['win_rates']['finetuned']:.1f}%)")
        print(f"  原生模型胜: {pref_report['preference_distribution']['base_wins']} ({pref_report['win_rates']['base']:.1f}%)")
        print(f"  平局: {pref_report['preference_distribution']['tied_good'] + pref_report['preference_distribution']['tied_bad']} ({pref_report['win_rates']['tied']:.1f}%)")
        print(f"  平均分差: {pref_report['avg_score_diff']:+.4f}")
        print(f"\n  DPO 数据已保存到: {DPO_OUTPUT_PATH}")
        print(f"  偏好报告已保存到: {RESULTS_DIR / 'dpo_preference_report.json'}")

    print(f"\n结果已保存到: {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="SBS 对比评估")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--n", type=int, default=10, help="快速测试的问题数量")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--adapter_path", type=str, default="models/lora_adapter", help="LoRA adapter 路径")
    parser.add_argument("--test_cases", type=str, default=None, help="测试用例 JSONL 文件路径")
    parser.add_argument("--dpo", action="store_true", dest="dpo", default=True, help="生成 DPO 偏好数据 (默认开启)")
    parser.add_argument("--no-dpo", action="store_false", dest="dpo", help="不生成 DPO 偏好数据")
    parser.add_argument("--verify", action="store_true", help="验证模式: 微调模型也用长Prompt测试")
    
    args = parser.parse_args()
    
    run_sbs_comparison(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        quick_test=args.quick,
        test_n=args.n if args.quick else None,
        enable_dpo=args.dpo,
        verify_mode=args.verify,
        test_cases_path=args.test_cases,
    )


if __name__ == "__main__":
    main()
