"""
测试用例自动生成脚本
生成 200+ 条测试用例用于 SBS 评估，重点覆盖 base_convert 和 solve 的弱项
输出: eval/test_cases.jsonl

用法:
  python build_test_cases.py                    # 生成默认 200 条
  python build_test_cases.py --count 500        # 生成 500 条
  python build_test_cases.py --seed 42          # 固定随机种子
"""
import os
import sys
import json
import random
import argparse
import math

OUTPUT_PATH = "eval/test_cases.jsonl"

CATEGORY_RATIO = {
    "single_tool": 0.50,
    "no_tool": 0.10,
    "multi_turn": 0.15,
    "reject": 0.05,
    "base_convert": 0.12,
    "solve": 0.08,
}


def gen_calc_cases(count):
    cases = []
    templates = [
        ("Calculate {a} + {b}", lambda a, b: a + b, "calc"),
        ("Calculate {a} - {b}", lambda a, b: a - b, "calc"),
        ("Calculate {a} * {b}", lambda a, b: a * b, "calc"),
        ("Calculate {a} / {b}", lambda a, b: round(a / b, 4), "calc"),
        ("What is {a} plus {b}?", lambda a, b: a + b, "calc"),
        ("What is {a} minus {b}?", lambda a, b: a - b, "calc"),
        ("What is {a} times {b}?", lambda a, b: a * b, "calc"),
        ("What is {a} divided by {b}?", lambda a, b: round(a / b, 4), "calc"),
        ("Compute {a} + {b}", lambda a, b: a + b, "calc"),
        ("Compute {a} * {b}", lambda a, b: a * b, "calc"),
        ("Add {a} and {b}", lambda a, b: a + b, "calc"),
        ("Subtract {b} from {a}", lambda a, b: a - b, "calc"),
        ("Multiply {a} by {b}", lambda a, b: a * b, "calc"),
        ("Divide {a} by {b}", lambda a, b: round(a / b, 4), "calc"),
        ("What is {a} mod {b}?", lambda a, b: a % b, "calc"),
        ("Calculate {a} % {b}", lambda a, b: a % b, "calc"),
        ("What is {a} to the power of {b}?", lambda a, b: a ** b, "calc"),
        ("Calculate {a} ** {b}", lambda a, b: a ** b, "calc"),
        ("What is the square root of {a}?", lambda a: round(math.sqrt(a), 4), "calc"),
        ("Calculate sqrt({a})", lambda a: round(math.sqrt(a), 4), "calc"),
        ("What is {a} squared?", lambda a: a ** 2, "calc"),
        ("Calculate {a}^2", lambda a: a ** 2, "calc"),
        ("What is the cube of {a}?", lambda a: a ** 3, "calc"),
        ("Calculate {a}^3", lambda a: a ** 3, "calc"),
        ("What is the absolute value of {a}?", lambda a: abs(a), "calc"),
        ("Calculate |{a}|", lambda a: abs(a), "calc"),
        ("What is {a} factorial?", lambda a: math.factorial(a), "calc"),
        ("Calculate {a}!", lambda a: math.factorial(a), "calc"),
        ("Calculate ({a} + {b}) * {c}", lambda a, b, c: (a + b) * c, "calc"),
        ("What is ({a} - {b}) * {c}?", lambda a, b, c: (a - b) * c, "calc"),
        ("Compute {a} * {b} + {c}", lambda a, b, c: a * b + c, "calc"),
        ("What is {a} * ({b} + {c})?", lambda a, b, c: a * (b + c), "calc"),
    ]

    perfect_squares = [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 400, 625, 900, 1024]
    small_factorials = [3, 4, 5, 6, 7, 8]

    for _ in range(count):
        tpl = random.choice(templates)
        text, fn, tool = tpl

        if "{c}" in text:
            a, b, c = random.randint(2, 50), random.randint(2, 50), random.randint(2, 10)
            try:
                ans = fn(a, b, c)
                if not (math.isfinite(ans) and abs(ans) < 1e8):
                    continue
            except:
                continue
            q = text.format(a=a, b=b, c=c)
        elif "square root" in text.lower() or "sqrt" in text.lower():
            a = random.choice(perfect_squares)
            try:
                ans = fn(a)
            except:
                continue
            q = text.format(a=a)
        elif "factorial" in text.lower() or "!" in text:
            a = random.choice(small_factorials)
            try:
                ans = fn(a)
            except:
                continue
            q = text.format(a=a)
        elif "absolute" in text.lower() or "|{" in text:
            a = random.choice([-50, -25, -10, -5, -1, 1, 5, 10, 25, 50])
            try:
                ans = fn(a)
            except:
                continue
            q = text.format(a=a)
        elif "squared" in text.lower() or "^2" in text:
            a = random.randint(2, 30)
            try:
                ans = fn(a)
            except:
                continue
            q = text.format(a=a)
        elif "cube" in text.lower() or "^3" in text:
            a = random.randint(2, 15)
            try:
                ans = fn(a)
            except:
                continue
            q = text.format(a=a)
        elif "power" in text.lower() or "**" in text or "^" in text:
            a = random.randint(2, 10)
            b = random.randint(2, 6)
            try:
                ans = fn(a, b)
                if not (math.isfinite(ans) and abs(ans) < 1e8):
                    continue
            except:
                continue
            q = text.format(a=a, b=b)
        elif "divided" in text.lower() or "/" in text:
            b = random.randint(2, 20)
            a = b * random.randint(2, 20)
            try:
                ans = fn(a, b)
            except:
                continue
            q = text.format(a=a, b=b)
        elif "mod" in text.lower() or "%" in text:
            a = random.randint(10, 200)
            b = random.randint(2, 20)
            try:
                ans = fn(a, b)
            except:
                continue
            q = text.format(a=a, b=b)
        else:
            a = random.randint(2, 100)
            b = random.randint(2, 100)
            try:
                ans = fn(a, b)
                if not (math.isfinite(ans) and abs(ans) < 1e8):
                    continue
            except:
                continue
            q = text.format(a=a, b=b)

        if isinstance(ans, float):
            if ans == int(ans):
                ans_str = str(int(ans))
            else:
                ans_str = str(round(ans, 2))
        else:
            ans_str = str(ans)

        cases.append({
            "question": q,
            "expected_tool": tool,
            "expected_answer": ans_str,
            "type": "single_tool",
        })

    return cases


def gen_convert_cases(count):
    cases = []
    templates = [
        ("Convert {v} {fu} to {tu}", "standard"),
        ("What is {v} {fu} in {tu}?", "question"),
        ("How many {tu} is {v} {fu}?", "question"),
        ("{v} {fu} to {tu}", "short"),
        ("Convert {v} {fu} into {tu}", "standard"),
    ]

    conversions = [
        ("km", "miles", lambda v: round(v * 0.621371, 2), (1, 500)),
        ("miles", "km", lambda v: round(v * 1.60934, 2), (1, 300)),
        ("kg", "pounds", lambda v: round(v * 2.20462, 2), (1, 500)),
        ("pounds", "kg", lambda v: round(v * 0.453592, 2), (1, 500)),
        ("celsius", "fahrenheit", lambda v: round(v * 9 / 5 + 32, 1), (-50, 200)),
        ("fahrenheit", "celsius", lambda v: round((v - 32) * 5 / 9, 1), (-50, 400)),
        ("meters", "feet", lambda v: round(v * 3.28084, 2), (1, 1000)),
        ("feet", "meters", lambda v: round(v * 0.3048, 2), (1, 3000)),
        ("inches", "cm", lambda v: round(v * 2.54, 2), (1, 100)),
        ("cm", "inches", lambda v: round(v * 0.393701, 2), (1, 250)),
        ("liters", "gallons", lambda v: round(v * 0.264172, 2), (1, 200)),
        ("gallons", "liters", lambda v: round(v * 3.78541, 2), (1, 100)),
        ("grams", "ounces", lambda v: round(v * 0.035274, 2), (1, 1000)),
        ("ounces", "grams", lambda v: round(v * 28.3495, 2), (1, 100)),
    ]

    for _ in range(count):
        tpl_text, tpl_type = random.choice(templates)
        fu, tu, fn, (lo, hi) = random.choice(conversions)
        v = random.randint(lo, hi)
        try:
            ans = fn(v)
        except:
            continue

        q = tpl_text.format(v=v, fu=fu, tu=tu)

        cases.append({
            "question": q,
            "expected_tool": "convert",
            "expected_answer": str(ans),
            "type": "single_tool",
        })

    return cases


def gen_base_convert_cases(count):
    cases = []
    templates = [
        ("Convert {v} to {base}", "standard"),
        ("What is {v} in {base}?", "question"),
        ("Convert {v} from decimal to {base}", "explicit"),
        ("Express {v} in {base}", "formal"),
        ("{v} to {base}", "short"),
        ("Convert the decimal number {v} to {base}", "verbose"),
        ("What is the {base} representation of {v}?", "formal"),
        ("Change {v} to {base}", "casual"),
        ("Rewrite {v} in {base}", "formal"),
        ("Convert {v} into {base} format", "standard"),
    ]

    base_names = {
        "binary": 2,
        "octal": 8,
        "hex": 16,
        "hexadecimal": 16,
    }

    reverse_templates = [
        ("Convert {v} from {base} to decimal", "reverse"),
        ("What is {base} number {v} in decimal?", "reverse_question"),
        ("Convert {v} ({base}) to decimal", "reverse_paren"),
        ("What is the decimal value of {base} {v}?", "reverse_formal"),
        ("{base} {v} to decimal", "reverse_short"),
    ]

    for _ in range(count):
        if random.random() < 0.6:
            tpl_text, _ = random.choice(templates)
            base_name, base_n = random.choice(list(base_names.items()))
            v = random.randint(2, 512)
            try:
                if base_n == 2:
                    ans = bin(v)[2:]
                elif base_n == 8:
                    ans = oct(v)[2:]
                elif base_n == 16:
                    ans = hex(v)[2:].upper()
            except:
                continue
            q = tpl_text.format(v=v, base=base_name)
        else:
            tpl_text, _ = random.choice(reverse_templates)
            base_name, base_n = random.choice(list(base_names.items()))
            v_dec = random.randint(2, 512)
            if base_n == 2:
                v_str = bin(v_dec)[2:]
            elif base_n == 8:
                v_str = oct(v_dec)[2:]
            elif base_n == 16:
                v_str = hex(v_dec)[2:].upper()
            ans = str(v_dec)
            q = tpl_text.format(v=v_str, base=base_name)

        cases.append({
            "question": q,
            "expected_tool": "base_convert",
            "expected_answer": ans,
            "type": "base_convert",
        })

    return cases


def gen_solve_cases(count):
    cases = []
    templates = [
        ("Solve {eq}", "standard"),
        ("Find x in {eq}", "find_x"),
        ("What is x if {eq}?", "question"),
        ("Solve for x: {eq}", "explicit"),
        ("Solve the equation {eq}", "verbose"),
        ("What value of x satisfies {eq}?", "formal"),
        ("Determine x from {eq}", "formal"),
        ("Find the solution to {eq}", "verbose"),
    ]

    for _ in range(count):
        eq_type = random.choice(["simple_linear", "two_step", "with_parens", "fraction"])

        if eq_type == "simple_linear":
            a = random.randint(2, 20)
            x_val = random.randint(-10, 20)
            b = a * x_val
            if b >= 0:
                eq = f"{a}x = {b}"
            else:
                eq = f"{a}x = {b}"
            ans = str(x_val)

        elif eq_type == "two_step":
            a = random.randint(2, 10)
            c = random.randint(-20, 20)
            x_val = random.randint(-10, 15)
            b = a * x_val + c
            if c >= 0:
                eq = f"{a}x + {c} = {b}"
            else:
                eq = f"{a}x - {abs(c)} = {b}"
            ans = str(x_val)

        elif eq_type == "with_parens":
            a = random.randint(2, 8)
            d = random.randint(1, 5)
            x_val = random.randint(-5, 10)
            inner = a * x_val
            rhs = inner + d
            eq = f"{a}(x) + {d} = {rhs}"
            ans = str(x_val)

        elif eq_type == "fraction":
            a = random.choice([2, 3, 4, 5, 6, 8, 10])
            x_val = random.randint(-5, 15)
            b = a * x_val
            eq = f"{a}x = {b}"
            ans = str(x_val)

        tpl_text, _ = random.choice(templates)
        q = tpl_text.format(eq=eq)

        cases.append({
            "question": q,
            "expected_tool": "solve",
            "expected_answer": f"x = {ans}",
            "type": "solve",
        })

    return cases


def gen_weather_cases(count):
    cities = [
        "Beijing", "Shanghai", "Tokyo", "London", "Paris", "New York",
        "Sydney", "Moscow", "Dubai", "Singapore", "Berlin", "Rome",
        "Seoul", "Bangkok", "Mumbai", "Cairo", "Toronto", "Vancouver",
        "Amsterdam", "Stockholm", "Helsinki", "Oslo", "Lisbon", "Madrid",
    ]
    templates = [
        "What is the weather in {city}?",
        "Weather in {city}",
        "What's the weather like in {city}?",
        "How is the weather in {city}?",
        "Tell me the weather in {city}",
        "What's the current weather in {city}?",
        "Is it raining in {city}?",
        "What's the temperature in {city}?",
        "Check weather for {city}",
        "Give me the weather forecast for {city}",
    ]
    cases = []
    for _ in range(count):
        city = random.choice(cities)
        tpl = random.choice(templates)
        q = tpl.format(city=city)
        cases.append({
            "question": q,
            "expected_tool": "weather",
            "expected_answer": None,
            "type": "single_tool",
        })
    return cases


def gen_time_cases(count):
    templates = [
        "What time is it?",
        "What's the current time?",
        "Tell me the current date and time",
        "What is today's date?",
        "What day is it today?",
        "Current time please",
        "What's the date?",
        "Give me the current time",
        "What time is it right now?",
        "Date and time please",
    ]
    cases = []
    for _ in range(count):
        q = random.choice(templates)
        cases.append({
            "question": q,
            "expected_tool": "time",
            "expected_answer": None,
            "type": "single_tool",
        })
    return cases


def gen_no_tool_cases(count):
    qa_pairs = [
        ("What is the capital of France?", "Paris"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("What is the chemical symbol for gold?", "Au"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the capital of Japan?", "Tokyo"),
        ("How many continents are there?", "7"),
        ("What is 2 + 2?", "4"),
        ("What is the speed of light?", "299792458"),
        ("Who discovered gravity?", "Newton"),
        ("What is the largest ocean?", "Pacific"),
        ("What is the smallest country?", "Vatican City"),
        ("Who invented the telephone?", "Bell"),
        ("What is the hardest natural substance?", "Diamond"),
        ("How many sides does a hexagon have?", "6"),
        ("What is the capital of Germany?", "Berlin"),
        ("Who wrote The Odyssey?", "Homer"),
        ("What is the boiling point of water?", "100"),
        ("What color is an emerald?", "Green"),
        ("How many bones are in the human body?", "206"),
        ("What is the capital of Australia?", "Canberra"),
        ("Who developed the theory of relativity?", "Einstein"),
        ("What is the largest mammal?", "Blue whale"),
        ("What is the chemical symbol for water?", "H2O"),
        ("Who was the first person on the moon?", "Armstrong"),
        ("What is the capital of Brazil?", "Brasilia"),
        ("How many planets are in our solar system?", "8"),
        ("What is the freezing point of water?", "0"),
        ("Who wrote Hamlet?", "Shakespeare"),
        ("What is the tallest mountain?", "Everest"),
        ("What is the capital of Canada?", "Ottawa"),
        ("Who discovered penicillin?", "Fleming"),
        ("What is the chemical symbol for iron?", "Fe"),
        ("How many days are in a leap year?", "366"),
        ("What is the capital of India?", "New Delhi"),
        ("Who painted the Sistine Chapel ceiling?", "Michelangelo"),
        ("What is the largest desert?", "Sahara"),
        ("What is the chemical symbol for silver?", "Ag"),
        ("How many chromosomes do humans have?", "46"),
        ("What is the capital of Egypt?", "Cairo"),
    ]

    chitchat = [
        "Hello!", "Hi there!", "Good morning!", "How are you?",
        "Nice to meet you!", "What's up?", "Hey!", "Greetings!",
        "Good evening!", "Howdy!", "Thanks for your help!",
        "I appreciate it!", "That's great!", "Interesting!",
        "Tell me about yourself.", "What can you do?",
        "How are you today?", "What's your name?",
        "Goodbye!", "See you later!",
    ]

    cases = []
    for _ in range(count):
        if random.random() < 0.7:
            q, a = random.choice(qa_pairs)
            cases.append({
                "question": q,
                "expected_tool": None,
                "expected_answer": a,
                "type": "no_tool",
            })
        else:
            q = random.choice(chitchat)
            cases.append({
                "question": q,
                "expected_tool": None,
                "expected_answer": None,
                "type": "no_tool",
            })
    return cases


def gen_reject_cases(count):
    reject_questions = [
        "Translate 'hello' to Chinese",
        "Write a poem about the ocean",
        "Tell me a joke",
        "What is the meaning of life?",
        "Write a short story about a cat",
        "Compose a haiku about spring",
        "Translate 'goodbye' to Spanish",
        "Write an essay on climate change",
        "Sing me a song",
        "Draw a picture of a house",
        "How do I hack a website?",
        "Write a love letter",
        "Create a recipe for chocolate cake",
        "What's your favorite color?",
        "Tell me a bedtime story",
        "Write a movie review",
        "How to make friends?",
        "Give me fashion advice",
        "What should I eat for dinner?",
        "Recommend a good book",
        "How do I learn to play guitar?",
        "Write a birthday card message",
        "What's the best vacation spot?",
        "How to improve my writing?",
        "Give me relationship advice",
        "How to start a business?",
        "What's a good workout routine?",
        "How to cook pasta?",
        "Write a cover letter",
        "How to meditate?",
    ]
    cases = []
    for _ in range(count):
        q = random.choice(reject_questions)
        cases.append({
            "question": q,
            "expected_tool": None,
            "expected_answer": None,
            "type": "reject",
        })
    return cases


def gen_multi_turn_cases(count):
    cases = []

    multi_turn_templates = [
        {
            "questions": [
                "Convert 100°F to Celsius, then add 10 to the result",
                "Convert 72°F to Celsius, then multiply by 2",
                "Convert 50°F to Celsius, then subtract 5",
                "Convert 212°F to Celsius, then divide by 5",
                "Convert 32°F to Celsius, then add 100",
            ],
            "expected_tool": "convert",
            "type": "multi_turn",
        },
        {
            "questions": [
                "Calculate 25 * 4, then convert that number to binary",
                "Calculate 16 * 4, then convert the result to hex",
                "Calculate 8 * 8, then convert to octal",
                "Calculate 10 * 10, then convert to binary",
                "Calculate 15 * 2, then convert to hex",
            ],
            "expected_tool": "calc",
            "type": "multi_turn",
        },
        {
            "questions": [
                "Solve 5x = 25, then convert x to hexadecimal",
                "Solve 3x = 48, then convert x to binary",
                "Solve 2x = 32, then convert x to octal",
                "Solve 4x = 64, then convert x to hex",
                "Solve 7x = 56, then convert x to binary",
            ],
            "expected_tool": "solve",
            "type": "multi_turn",
        },
        {
            "questions": [
                "Convert 100 km to miles, then multiply by 2",
                "Convert 50 kg to pounds, then add 10",
                "Convert 30 celsius to fahrenheit, then subtract 32",
                "Convert 5 miles to km, then divide by 2",
                "Convert 200 pounds to kg, then add 5",
            ],
            "expected_tool": "convert",
            "type": "multi_turn",
        },
        {
            "questions": [
                "Calculate 2 ** 8, then convert to hexadecimal",
                "Calculate 3 ** 4, then convert to binary",
                "Calculate 2 ** 10, then convert to octal",
                "Calculate 5 ** 3, then convert to hex",
                "Calculate 2 ** 6, then convert to binary",
            ],
            "expected_tool": "calc",
            "type": "multi_turn",
        },
        {
            "questions": [
                "What is the weather in Tokyo? Then convert 25 celsius to fahrenheit",
                "Check weather in London, then convert 10 km to miles",
                "What's the weather in Paris? Also calculate 15 * 8",
                "Get weather for New York, then solve 3x = 21",
                "Weather in Berlin, then convert 100 pounds to kg",
            ],
            "expected_tool": "weather",
            "type": "multi_turn",
        },
        {
            "questions": [
                "Solve 2x + 3 = 13, then calculate x squared",
                "Solve 4x - 7 = 17, then multiply x by 3",
                "Solve 5x + 2 = 22, then add 10 to x",
                "Solve 3x - 5 = 16, then divide x by 7",
                "Solve 6x + 1 = 37, then subtract 2 from x",
            ],
            "expected_tool": "solve",
            "type": "multi_turn",
        },
        {
            "questions": [
                "Convert 255 to hex, then convert the result to decimal",
                "Convert 64 to binary, then count the number of 1s",
                "Convert 100 to octal, then add 50",
                "Convert 128 to hex, then multiply by 2",
                "Convert 32 to binary, then convert back to decimal",
            ],
            "expected_tool": "base_convert",
            "type": "multi_turn",
        },
    ]

    for _ in range(count):
        tpl = random.choice(multi_turn_templates)
        q = random.choice(tpl["questions"])
        cases.append({
            "question": q,
            "expected_tool": tpl["expected_tool"],
            "expected_answer": None,
            "type": "multi_turn",
        })

    return cases


def build_test_cases(total_count=200, seed=42):
    random.seed(seed)

    counts = {}
    for cat, ratio in CATEGORY_RATIO.items():
        counts[cat] = max(1, int(total_count * ratio))

    remainder = total_count - sum(counts.values())
    counts["single_tool"] += remainder

    all_cases = []

    calc_count = int(counts["single_tool"] * 0.35)
    convert_count = int(counts["single_tool"] * 0.35)
    weather_count = int(counts["single_tool"] * 0.15)
    time_count = counts["single_tool"] - calc_count - convert_count - weather_count

    all_cases.extend(gen_calc_cases(calc_count))
    all_cases.extend(gen_convert_cases(convert_count))
    all_cases.extend(gen_weather_cases(weather_count))
    all_cases.extend(gen_time_cases(time_count))
    all_cases.extend(gen_base_convert_cases(counts["base_convert"]))
    all_cases.extend(gen_solve_cases(counts["solve"]))
    all_cases.extend(gen_no_tool_cases(counts["no_tool"]))
    all_cases.extend(gen_multi_turn_cases(counts["multi_turn"]))
    all_cases.extend(gen_reject_cases(counts["reject"]))

    random.shuffle(all_cases)

    for i, case in enumerate(all_cases):
        case["id"] = i + 1

    return all_cases


def main():
    parser = argparse.ArgumentParser(description="测试用例自动生成")
    parser.add_argument("--count", type=int, default=200, help="生成测试用例总数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="输出路径")
    args = parser.parse_args()

    cases = build_test_cases(args.count, args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    type_counts = {}
    for c in cases:
        t = c.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"生成 {len(cases)} 条测试用例 -> {args.output}")
    print(f"\n类别分布:")
    for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {cnt} ({cnt/len(cases)*100:.1f}%)")

    tool_counts = {}
    for c in cases:
        t = c.get("expected_tool")
        tool_counts[t] = tool_counts.get(t, 0) + 1

    print(f"\n工具分布:")
    for t, cnt in sorted(tool_counts.items(), key=lambda x: -x[1]):
        label = t if t else "None (no_tool/reject)"
        print(f"  {label}: {cnt} ({cnt/len(cases)*100:.1f}%)")


if __name__ == "__main__":
    main()
