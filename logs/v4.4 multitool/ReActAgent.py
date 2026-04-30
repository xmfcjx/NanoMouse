import re
from typing import Dict, Callable, Optional
import datetime
import math
import random

class ReActAgent:
    def __init__(self, llm_instance: object, max_steps: int = 5):
        self.llm_instance = llm_instance
        self.max_steps = max_steps
        self.tools = {}

        # 批量注册工具定义 (名称, 函数, 描述) - 短名称设计，防止模型匹配不上
        TOOL_DEFS = [
            ("calc", self._python_repl, "用于执行Python数学表达式。输入必须是合法的Python代码字符串，如果是计算结果，请务必使用 print() 输出。例如：'print(3 * 8)'"),
            ("time", self._time_tool, "用户获取当前日期和时间"),
            ("date", self._date_calc, "用于日期加减计算。格式：'YYYY-MM-DD +/- N天'。例如：'2025-07-05 + 30天'"),
            ("weekday", self._weekday, "查询某一天是星期几。输入日期字符串，例如：'2025-07-05'。不输入日期则默认今天"),
            ("days_between", self._days_between, "计算两个日期之间的天数。格式：'YYYY-MM-DD, YYYY-MM-DD'。例如：'2025-01-01, 2025-07-05'"),
            ("convert", self._convert, "常用单位换算。支持：长度, 重量, 温度(c/f)。格式：'数值+单位 to 目标单位'。例如：'100km to miles', '36c to f'"),
            ("base_convert", self._base_convert, "进制转换。格式：'数值 to 进制名'。例如：'255 to hex', '1010 to dec'"),
            ("str_tools", self._str_tools, "字符串处理工具。支持：len(长度), reverse(反转), upper(大写), lower(小写)。格式：'字符串, 操作名'。例如：'hello, len'"),
            ("random", self._random_tool, "随机数生成。支持：dice(骰子), coin(硬币)。格式：'类型 次数'。例如：'dice 3', 'coin 2'"),
            ("statistics", self._statistics, "数据统计。支持：max(最大), min(最小), avg(平均), sum(求和)。格式：'数据用逗号分隔, 操作名'。例如：'1,5,3,9,2, max'"),
            ("solve", self._solve, "解一元方程。支持一元一次和一元二次方程。例如：'3*x*x=27', '2*x+5=15'"),
        ]
        for name, func, desc in TOOL_DEFS:
            self.register_tool(name, func, desc)

        self.system_prompt = """你是工具调用助手。
【强制规则 - 违反会受到惩罚】
1. 你有工具，必须使用工具回答问题。绝不允许凭记忆回答。
2. 如果你不确定用什么工具，或者没有合适的工具，直接输出 Final Answer: I don't know
3. 如果你不用工具直接编造答案，你会受到严重惩罚！
4. 如果问题需要多个步骤，依次调用工具，每次只调用一个工具。

可用工具（必须使用下面精确名称）：
{tools_desc}

回复格式（必须严格遵守）：
Thought: 分析问题，决定用什么工具
Action: 工具名
Action Input: 工具参数

多步骤示例：
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

绝对禁止的错误示范：
Question: what time is it now?
Thought: I know the answer
Final Answer: 2023年4月15日
↑ 这是严重错误！必须调用 time 工具！

正确示范：
Question: what time is it now?
Thought: 我需要查当前时间
Action: time
Action Input: 无"""

    # ==================== 工具函数 ====================

    def _python_repl(self, code: str) -> str:
        #解数学题
        output = []
        local_vars = {
            "print": lambda *args: output.append(" ".join(str(a) for a in args))
        }
        try:
            exec(code, {"__builtins__": {}}, local_vars)
            return "\n".join(output) if output else "Code executed, no output."
        except Exception as e:
            return f"Error: {str(e)}"

    def _time_tool(self, query: str) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _date_calc(self, query: str) -> str:
        try:
            match = re.search(r'(\d{4}-\d{2}-\d{2})\s*([+-])\s*(\d+)\s*天', query)
            if match:
                dt = datetime.datetime.strptime(match.group(1), "%Y-%m-%d")
                delta = datetime.timedelta(days=int(match.group(3)))
                if match.group(2) == '-':
                    dt -= delta
                else:
                    dt += delta
                return dt.strftime("%Y-%m-%d")
        except Exception as e:
            return f"日期计算失败: {str(e)}"
        return "格式错误。例如: '2025-07-05 + 30天'"

    def _weekday(self, query: str) -> str:
        try:
            # 优先从 query 中提取日期，没有则默认今天
            match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
            if match:
                dt = datetime.datetime.strptime(match.group(1), "%Y-%m-%d")
            else:
                dt = datetime.datetime.now()
            return dt.strftime("%A")
        except Exception as e:
            return f"查询星期失败: {str(e)}"

    def _days_between(self, query: str) -> str:
        try:
            dates = re.findall(r'(\d{4}-\d{2}-\d{2})', query)
            if len(dates) == 2:
                d1 = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
                d2 = datetime.datetime.strptime(dates[1], "%Y-%m-%d")
                return str(abs((d2 - d1).days))
        except Exception as e:
            return f"计算天数失败: {str(e)}"
        return "格式错误。例如: '2025-01-01, 2025-07-05'"

    def _convert(self, query: str) -> str:
        q = query.lower().strip()
        try:
            # 温度特殊处理
            if 'c to f' in q or 'celsius to fahrenheit' in q:
                val = float(re.search(r'[\d.]+', q).group())
                return f"{val * 9/5 + 32}°F"
            if 'f to c' in q or 'fahrenheit to celsius' in q:
                val = float(re.search(r'[\d.]+', q).group())
                return f"{(val - 32) * 5/9:.2f}°C"

            # 长度/重量处理
            base_map = {'km': 1000, 'miles': 1609.34, 'meters': 1, 'feet': 0.3048,
                        'kg': 1, 'pounds': 0.453592, 'jin': 0.5}
            match = re.match(r'([\d.]+)\s*(km|miles|meters|feet|kg|pounds|jin)\s*(to|转)\s*(km|miles|meters|feet|kg|pounds|jin)', q)
            if match:
                val = float(match.group(1))
                from_u = match.group(2)
                to_u = match.group(4)
                base = val * base_map[from_u]
                result = base / base_map[to_u]
                return f"{round(result, 4)} {to_u}"
        except Exception as e:
            return f"转换失败: {str(e)}"
        return "不支持的转换格式。支持: km/miles/kg/pounds/jin, c/f"

    def _base_convert(self, query: str) -> str:
        try:
            q = query.lower()
            match = re.match(r'([\da-fA-F]+)\s*(?:to|转)\s*(hex|hexadecimal|dec|decimal|bin|binary|oct|octal)', q)
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

                num = int(val_str, source_base)
                formats = {'hex': hex(num).upper()[2:], 'dec': str(num), 'bin': bin(num)[2:], 'oct': oct(num)[2:]}
                return formats[target]
        except Exception as e:
            return f"进制转换失败: {str(e)}"
        return "格式错误。例如: '255 to hex', '1010 to dec'"


    def _str_tools(self, query: str) -> str:
        try:
            # 从右向左分割一次，防止字符串本身含有逗号
            parts = [p.strip() for p in query.rsplit(',', 1)]
            if len(parts) == 2:
                text, op = parts
                op = op.lower().strip()
                if op == 'len': return str(len(text))
                if op == 'reverse': return text[::-1]
                if op == 'upper': return text.upper()
                if op == 'lower': return text.lower()
        except Exception as e:
            return f"字符串处理失败: {str(e)}"
        return "格式错误。例如: 'hello, len'"

    def _random_tool(self, query: str) -> str:
        q = query.lower().strip()
        try:
            if 'dice' in q or '骰子' in q:
                n = int(re.search(r'\d+', q).group()) if re.search(r'\d+', q) else 1
                return ", ".join(str(random.randint(1, 6)) for _ in range(n))
            if 'coin' in q or '硬币' in q:
                n = int(re.search(r'\d+', q).group()) if re.search(r'\d+', q) else 1
                return ", ".join(random.choice(["Head(正)", "Tail(反)"]) for _ in range(n))
        except Exception as e:
            return f"随机生成失败: {str(e)}"
        return "格式错误。例如: 'dice 3', 'coin 2'"

    def _statistics(self, query: str) -> str:
        try:
            parts = [p.strip() for p in query.rsplit(',', 1)]
            if len(parts) == 2:
                nums_str, op = parts
                # 安全提取所有数字
                nums = []
                for x in nums_str.split(','):
                    try: nums.append(float(x.strip()))
                    except: pass

                if not nums: return "无有效数字"
                op = op.lower().strip()
                if op == 'max': return str(max(nums))
                if op == 'min': return str(min(nums))
                if op in ['avg', 'mean']: return str(sum(nums)/len(nums))
                if op == 'sum': return str(sum(nums))
        except Exception as e:
            return f"统计失败: {str(e)}"
        return "格式错误。例如: '1,5,3,9,2, max'"

    def _solve(self, query: str) -> str:
        """解一元方程工具，支持一元一次和一元二次方程"""
        try:
            import sympy
            q = query.strip().lower()
            # 移除干扰词
            for word in ["solve", "find", "calculate", "for", "the", "equation", "求解"]:
                q = q.replace(word, "")
            q = q.replace("?", "").replace(".", "").replace(",", "").strip()
            q = q.replace(" ", "")

            # 提取等式部分
            eq_match = re.search(r"(.+=.+)", q)
            if not eq_match:
                return "格式错误：请提供包含 '=' 的方程"
            eq = eq_match.group(1)

            # 标准化：x*x → x**2，x^2 → x**2，数字和x之间补*（3x → 3*x）
            eq = eq.replace("x*x", "x**2").replace("^", "**")
            eq = re.sub(r'(\d)(x)', r'\1*\2', eq)  # 3x → 3*x
            eq = re.sub(r'(x)(\d)', r'\1*\2', eq)  # x3 → x*3（少见但防御一下）

            # 分割等号两边
            parts = eq.split('=')
            if len(parts) != 2:
                return "格式错误：方程格式不正确"

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
                    return " 或 ".join([f"x = {f}" for f in formatted])
        except ImportError:
            return "Error: sympy 未安装，请执行 pip install sympy"
        except Exception as e:
            return f"解方程失败: {str(e)}"

    # ==================== 注册 & 执行 ====================

    def register_tool(self, name: str, func: Callable, description: str) -> None:
        self.tools[name] = {
            "func": func,
            "desc": description
        }

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        # 容错机制：小写化 + 去空格
        name_clean = tool_name.lower().strip().replace(" ", "_")

        # 精确匹配
        if name_clean in self.tools:
            return self.tools[name_clean]["func"](tool_input)

        # 模糊匹配：检查是否是某个工具名的子串，或工具名是其子串
        for tname in self.tools:
            if name_clean in tname or tname in name_clean:
                return self.tools[tname]["func"](tool_input)

        return f"Error: 工具 '{tool_name}' 不存在。可用: {list(self.tools.keys())}"

    # ==================== 直通车（不经过 LLM） ====================

    def direct_call(self, tool_name: str, tool_input: str = "") -> str:
        """直通车：不经过 LLM 推理，直接调用指定工具（用于确定性操作）"""
        return self._execute_tool(tool_name, tool_input)

    # ==================== ReAct 循环（需要 LLM 推理时使用） ====================

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
            ans = self.llm_instance.generate(prompt_context, max_new_tokens=128)
            
            if debug:
                print(f"\n[DEBUG Step {i+1}] Model output:")
                print(f"  {repr(ans[:200])}")
            
            prompt_context += ans
            response = self._parse_response(ans)

            if response["type"] == "final_answer":
                return response["content"]
            elif response["type"] == "action":
                res = self._execute_tool(response["name"], response["input"])
                if debug:
                    print(f"[DEBUG] Tool: {response['name']} -> {res}")
                prompt_context += f"\nObservation: {res}\nThought:"
            else:
                if debug:
                    print(f"[DEBUG] Parse failed, raw output: {repr(ans)}")
                return "模型输出格式错误"

        return "Max steps reached."
