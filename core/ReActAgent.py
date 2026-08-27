import re
import ast
import operator
from typing import Dict, Callable, Optional
import datetime
import math
import random
import signal
import threading
from config.config import get_config

class ReActAgent:
    def __init__(self, llm_instance: object, max_steps: int = None):
        self.llm_instance = llm_instance
        self.max_steps = max_steps if max_steps is not None else get_config("agent.max_steps", 3)
        self.max_new_tokens = get_config("agent.max_new_tokens", 128)
        self.tools = {}

        # 批量注册工具定义 (名称, 函数, 描述) - 短名称设计，防止模型匹配不上
        TOOL_DEFS = [
            ("calc", self._python_repl, "Execute Python math expressions. Input must be valid Python code. Use print() for results. Example: 'print(3 * 8)'"),
            ("time", self._time_tool, "Get current date and time. No input needed."),
            ("weather", self._weather, "Query city weather. Input city name. Example: 'Beijing', 'Shanghai', 'London'"),
            ("date", self._date_calc, "Date calculation. Format: 'YYYY-MM-DD +/- N days'. Example: '2025-07-05 + 30 days'"),
            ("weekday", self._weekday, "Get day of week for a date. Input date string or empty for today. Example: '2025-07-05'"),
            ("days_between", self._days_between, "Calculate days between two dates. Format: 'YYYY-MM-DD, YYYY-MM-DD'. Example: '2025-01-01, 2025-07-05'"),
            ("convert", self._convert, "Unit conversion. Supports length, weight, temperature. Format: 'value+unit to target'. Example: '100km to miles', '36c to f'"),
            ("base_convert", self._base_convert, "Number base conversion. Format: 'number to base'. Example: '255 to hex', '1010 to dec'"),
            ("str_tools", self._str_tools, "String operations: len, reverse, upper, lower. Format: 'string, operation'. Example: 'hello, len'"),
            ("random", self._random_tool, "Random generation: dice, coin. Format: 'type count'. Example: 'dice 3', 'coin 2'"),
            ("statistics", self._statistics, "Statistics: max, min, avg, sum. Format: 'numbers, operation'. Example: '1,5,3,9,2, max'"),
            ("solve", self._solve, "Solve equations. Supports linear and quadratic. Example: '3*x*x=27', '2*x+5=15'"),
        ]
        for name, func, desc in TOOL_DEFS:
            self.register_tool(name, func, desc)

        self.system_prompt = """You are a tool-calling assistant.

RULES:
1. You MUST use tools to answer questions. Never answer from memory.
2. If unsure which tool to use, output: Final Answer: I don't know
3. For multi-step tasks, call tools one at a time.

Available tools:
{tools_desc}

Format:
Thought: Analyze the problem
Action: tool_name
Action Input: tool_input

Example:
Question: calculate 25 * 4, then convert to binary
Thought: First I need to calculate 25 * 4
Action: calc
Action Input: print(25 * 4)
Observation: 100
Thought: Now I need to convert 100 to binary
Action: base_convert
Action Input: 100 to bin
Observation: 1100100
Final Answer: 1100100

WRONG example (DO NOT do this):
Question: what time is it now?
Thought: I know the answer
Final Answer: 2023-04-15
(This is WRONG! You MUST call the time tool!)

CORRECT example:
Question: what time is it now?
Thought: I need to get current time
Action: time
Action Input:"""

    # ==================== 工具函数 ====================

    _SAFE_OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    _SAFE_FUNCS = {
        "abs": abs, "round": round, "int": int, "float": float,
        "max": max, "min": min, "sum": sum, "len": len,
        "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e,
    }

    def _safe_eval_node(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value}")
        if isinstance(node, ast.UnaryOp):
            op_func = self._SAFE_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
            return op_func(self._safe_eval_node(node.operand))
        if isinstance(node, ast.BinOp):
            op_func = self._SAFE_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported op: {type(node.op).__name__}")
            return op_func(self._safe_eval_node(node.left), self._safe_eval_node(node.right))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in self._SAFE_FUNCS:
                args = [self._safe_eval_node(a) for a in node.args]
                return self._SAFE_FUNCS[node.func.id](*args)
            raise ValueError(f"Unsupported function: {ast.dump(node.func)}")
        if isinstance(node, ast.Name):
            if node.id in self._SAFE_FUNCS:
                return self._SAFE_FUNCS[node.id]
            raise ValueError(f"Unknown name: {node.id}")
        raise ValueError(f"Unsupported expression: {type(node).__name__}")

    def _python_repl(self, code: str) -> str:
        expr = code.strip()
        if expr.startswith("print(") and expr.endswith(")"):
            expr = expr[6:-1]
        try:
            tree = ast.parse(expr, mode='eval')
            result = self._safe_eval_node(tree.body)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def _time_tool(self, query: str) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _weather(self, city: str) -> str:
        try:
            import requests
            city = city.strip()
            r = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
            data = r.json()
            current = data['current_condition'][0]
            temp = current['temp_C']
            desc = current['weatherDesc'][0]['value']
            humidity = current['humidity']
            wind = current['windspeedKmph']
            return f"{city}: {temp}°C, {desc}, Humidity: {humidity}%, Wind: {wind}km/h"
        except Exception as e:
            return f"Weather query failed: {str(e)}"

    def _date_calc(self, query: str) -> str:
        try:
            match = re.search(r'(\d{4}-\d{2}-\d{2})\s*([+-])\s*(\d+)\s*(?:days?|天)?', query, re.IGNORECASE)
            if match:
                dt = datetime.datetime.strptime(match.group(1), "%Y-%m-%d")
                delta = datetime.timedelta(days=int(match.group(3)))
                if match.group(2) == '-':
                    dt -= delta
                else:
                    dt += delta
                return dt.strftime("%Y-%m-%d")
        except Exception as e:
            return f"Date calculation failed: {str(e)}"
        return "Format error. Example: '2025-07-05 + 30 days'"

    def _weekday(self, query: str) -> str:
        try:
            match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
            if match:
                dt = datetime.datetime.strptime(match.group(1), "%Y-%m-%d")
            else:
                dt = datetime.datetime.now()
            return dt.strftime("%A")
        except Exception as e:
            return f"Weekday query failed: {str(e)}"

    def _days_between(self, query: str) -> str:
        try:
            dates = re.findall(r'(\d{4}-\d{2}-\d{2})', query)
            if len(dates) == 2:
                d1 = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
                d2 = datetime.datetime.strptime(dates[1], "%Y-%m-%d")
                return str(abs((d2 - d1).days))
        except Exception as e:
            return f"Days calculation failed: {str(e)}"
        return "Format error. Example: '2025-01-01, 2025-07-05'"

    def _convert(self, query: str) -> str:
        q = query.lower().strip()
        try:
            if 'c to f' in q or 'celsius to fahrenheit' in q:
                val = float(re.search(r'[\d.]+', q).group())
                return f"{val * 9/5 + 32}F"
            if 'f to c' in q or 'fahrenheit to celsius' in q:
                val = float(re.search(r'[\d.]+', q).group())
                return f"{(val - 32) * 5/9:.2f}C"

            base_map = {'km': 1000, 'mile': 1609.34, 'miles': 1609.34, 'meters': 1, 'feet': 0.3048,
                        'kg': 1, 'pound': 0.453592, 'pounds': 0.453592, 'jin': 0.5}
            match = re.match(r'([\d.]+)\s*(km|mile|miles|meters|feet|kg|pound|pounds|jin)\s*(?:to)?\s*(km|mile|miles|meters|feet|kg|pound|pounds|jin)', q)
            if match:
                val = float(match.group(1))
                from_u = match.group(2)
                to_u = match.group(3)
                base = val * base_map[from_u]
                result = base / base_map[to_u]
                return f"{round(result, 4)} {to_u}"
        except Exception as e:
            return f"Conversion failed: {str(e)}"
        return "Unsupported format. Supports: km/miles/kg/pounds/jin, c/f"

    def _base_convert(self, query: str) -> str:
        try:
            q = query.lower()
            match = re.match(r'([\da-fA-F]+)\s*(?:to)?\s*(hex|hexadecimal|dec|decimal|bin|binary|oct|octal)', q)
            if match:
                val_str = match.group(1)
                target_raw = match.group(2)
                
                target = target_raw[:3]
                
                source_base = 10
                if val_str.startswith('0b'):
                    source_base = 2
                    val_str = val_str[2:]
                elif val_str.startswith('0o'):
                    source_base = 8
                    val_str = val_str[2:]
                elif val_str.startswith('0x'):
                    source_base = 16
                    val_str = val_str[2:]
                elif all(c in '01' for c in val_str) and target in ['dec', 'hex', 'oct']:
                    source_base = 2

                num = int(val_str, source_base)
                formats = {'hex': hex(num).upper()[2:], 'dec': str(num), 'bin': bin(num)[2:], 'oct': oct(num)[2:]}
                return formats[target]
        except Exception as e:
            return f"Base conversion failed: {str(e)}"
        return "Format error. Example: '255 to hex', '1010 to dec'"


    def _str_tools(self, query: str) -> str:
        try:
            parts = [p.strip() for p in query.rsplit(',', 1)]
            if len(parts) == 2:
                text, op = parts
                op = op.lower().strip()
                if op == 'len': return str(len(text))
                if op == 'reverse': return text[::-1]
                if op == 'upper': return text.upper()
                if op == 'lower': return text.lower()
        except Exception as e:
            return f"String operation failed: {str(e)}"
        return "Format error. Example: 'hello, len'"

    def _random_tool(self, query: str) -> str:
        q = query.lower().strip()
        try:
            if 'dice' in q:
                n = int(re.search(r'\d+', q).group()) if re.search(r'\d+', q) else 1
                return ", ".join(str(random.randint(1, 6)) for _ in range(n))
            if 'coin' in q:
                n = int(re.search(r'\d+', q).group()) if re.search(r'\d+', q) else 1
                return ", ".join(random.choice(["Head", "Tail"]) for _ in range(n))
        except Exception as e:
            return f"Random generation failed: {str(e)}"
        return "Format error. Example: 'dice 3', 'coin 2'"

    def _statistics(self, query: str) -> str:
        try:
            parts = [p.strip() for p in query.rsplit(',', 1)]
            if len(parts) == 2:
                nums_str, op = parts
                nums = []
                for x in nums_str.split(','):
                    try: nums.append(float(x.strip()))
                    except: pass

                if not nums: return "No valid numbers"
                op = op.lower().strip()
                if op == 'max': return str(max(nums))
                if op == 'min': return str(min(nums))
                if op in ['avg', 'mean']: return str(sum(nums)/len(nums))
                if op == 'sum': return str(sum(nums))
        except Exception as e:
            return f"Statistics failed: {str(e)}"
        return "Format error. Example: '1,5,3,9,2, max'"

    def _solve(self, query: str) -> str:
        try:
            import sympy
            q = query.strip().lower()
            for word in ["solve", "find", "calculate", "for", "the", "equation"]:
                q = q.replace(word, "")
            q = q.replace("?", "").replace(".", "").replace(",", "").strip()
            q = q.replace(" ", "")

            eq_match = re.search(r"(.+=.+)", q)
            if not eq_match:
                return "Format error: equation must contain '='"
            eq = eq_match.group(1)

            eq = eq.replace("x*x", "x**2").replace("^", "**")
            eq = re.sub(r'(\d)(x)', r'\1*\2', eq)
            eq = re.sub(r'(x)(\d)', r'\1*\2', eq)

            parts = eq.split('=')
            if len(parts) != 2:
                return "Format error: invalid equation"

            x = sympy.Symbol('x')
            equation = sympy.Eq(sympy.sympify(parts[0]), sympy.sympify(parts[1]))
            solutions = sympy.solve(equation, x)

            if len(solutions) == 0:
                return "No real solution"
            else:
                formatted = []
                for s in solutions:
                    val = float(s)
                    if val == int(val):
                        formatted.append(str(int(val)))
                    else:
                        formatted.append(str(round(val, 4)))
                if len(formatted) == 1:
                    return f"x = {formatted[0]}"
                else:
                    return " or ".join([f"x = {f}" for f in formatted])
        except ImportError:
            return "Error: sympy not installed. Run: pip install sympy"
        except Exception as e:
            return f"Equation solving failed: {str(e)}"

    # ==================== 注册 & 执行 ====================

    def register_tool(self, name: str, func: Callable, description: str) -> None:
        self.tools[name] = {
            "func": func,
            "desc": description
        }

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        name_clean = tool_name.lower().strip().replace(" ", "_")

        if name_clean in self.tools:
            return self.tools[name_clean]["func"](tool_input)

        for tname in self.tools:
            if name_clean in tname or tname in name_clean:
                return self.tools[tname]["func"](tool_input)

        return f"Error: Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"

    def direct_call(self, tool_name: str, tool_input: str = "") -> str:
        return self._execute_tool(tool_name, tool_input)

    def _parse_response(self, text: str) -> Dict[str, Optional[str]]:
        final_answer = re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE).search(text)
        if final_answer is not None:
            return {"type": "final_answer", "content": final_answer.group(1)}

        action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
        input_match = re.search(r"Action Input:\s*(.*)", text, re.IGNORECASE | re.DOTALL)

        if action_match and input_match:
            return {"type": "action", "name": action_match.group(1), "input": input_match.group(1)}

        return {"type": "error", "content": None}

    def run(self, question: str, debug: bool = False) -> str:
        tools_desc = "\n".join([f"- {name}: {info['desc']}" for name, info in self.tools.items()])

        prompt_context = self.system_prompt.format(tools_desc=tools_desc)
        prompt_context += f"\nQuestion: {question}\nThought:"

        for i in range(self.max_steps):
            ans = self.llm_instance.generate(prompt_context, max_new_tokens=self.max_new_tokens, use_chat_template=False)
            
            if debug:
                print(f"\n[DEBUG Step {i+1}] Model output:", flush=True)
                print(f"  {repr(ans[:300])}", flush=True)
            
            prompt_context += ans
            response = self._parse_response(ans)

            if response["type"] == "final_answer":
                if debug:
                    print(f"[DEBUG] Final answer found: {response['content']}", flush=True)
                return response["content"]
            elif response["type"] == "action":
                res = self._execute_tool(response["name"], response["input"])
                if debug:
                    print(f"[DEBUG] Tool: {response['name']} -> {res}", flush=True)
                prompt_context += f"\nObservation: {res}\nThought:"
            else:
                if debug:
                    print(f"[DEBUG] Parse failed, raw output: {repr(ans)}", flush=True)
                return "Model output format error"

        return "Max steps reached."
