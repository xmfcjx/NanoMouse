import re
import math

class InputClassifier:
    """输入分类器 - 统一管理所有输入类型判断"""

    def __init__(self):
        # Identity 关键词（查询类）
        self.identity_keys = [
            "who am i", "what is my name", "what's my name",
            "do you know my name", "remember my name", "tell me my name",
        ]
        # 名称黑名单
        self.name_blacklist = {"happy", "fine", "tired", "ok", "good", "sad", "angry"}
        # 名称提取正则
        self.name_patterns = [
            re.compile(r"\bmy name is\s+([A-Za-z]{2,20})\b", re.IGNORECASE),
            re.compile(r"\bi am\s+([A-Za-z]{2,20})\b", re.IGNORECASE),
            re.compile(r"\bi'm\s+([A-Za-z]{2,20})\b", re.IGNORECASE),
            re.compile(r"\b(?:changed my name|change my name) to\s+([A-Za-z]{2,20})\b", re.IGNORECASE),
        ]
        # 数学自然语言 -> 符号映射
        self.math_word_map = {
            "multiplied by": "*", "times": "*", "plus": "+",
            "minus": "-", "divided by": "/", "solve": "",
            "find": "", "calculate": "", "what is": "", "what's": "",
        }

    def _normalize_math_text(self, text):
        t = text.lower()
        for word, symbol in self.math_word_map.items():
            t = t.replace(word, symbol)
        return t

    def is_identity_question(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in self.identity_keys)

    def extract_user_name(self, history):
        for msg in reversed(history):
            if msg.get("role") != "user": continue
            text = msg.get("content", "").strip()
            for p in self.name_patterns:
                m = p.search(text)
                if m:
                    name = m.group(1).strip()
                    if name.lower() not in self.name_blacklist: return name
        return None

    def extract_name_from_current_query(self, query: str):
        for p in self.name_patterns:
            m = p.search(query)
            if m:
                name = m.group(1).strip()
                if name.lower() not in self.name_blacklist: return name
        return None

    def try_arithmetic(self, query: str):
        q_orig = query.strip().lower()
        q_norm = self._normalize_math_text(q_orig)
        for q in [q_orig, q_norm]:
            q = re.sub(r"^(please\s+)?(count|calculate|compute|what is|what's)\s+", "", q).strip()
            q = q.replace("=?", "").replace("= ?", "").replace("?", "").strip()
            if len(q) > 50: continue
            if re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", q):
                try:
                    val = eval(q, {"__builtins__": {}}, {})
                    if isinstance(val, float) and val.is_integer(): return str(int(val))
                    return str(val)
                except: continue
        return None

    def is_equation_question(self, query: str):
        q = query.lower()
        q_norm = self._normalize_math_text(q)
        return ("x" in q and "=" in q) or ("x" in q_norm and "=" in q_norm)

    def solve_equation(self, query: str):
        q = query.strip().lower()
        # 1. 移除干扰词 (增加了 "for")
        for word in ["solve", "find", "calculate", "for"]:
            q = q.replace(word, "")
        # 2. 移除标点
        q = q.replace("?", "").replace(".", "").replace(",", "").strip()

        # 🎯 FIX 1: 绝对关键！必须先去空格，再正则截取
        q = q.replace(" ", "")

        # 3. 截取纯方程
        eq_match = re.match(r"([\dx\+\-\*/\^]+=[\d\+\-\*/\^]+)", q)
        if eq_match:
            q = eq_match.group(1)

        # 🎯 FIX 2: 标准化 x*x -> x^2
        q = q.replace("x*x", "x^2")

        # 格式 1: x^2 = n
        m = re.fullmatch(r"(x\^2)=([+-]?\d+)", q)
        if m:
            n = int(m.group(2))
            if n < 0: return "No real solution."
            r = int(n ** 0.5)
            if r * r == n: return f"x = {r} or x = -{r}" if r != 0 else "x = 0"

        # 格式 2: ax + b = c
        m = re.fullmatch(r"([+-]?\d+)\*x([+-]\d+)=([+-]?\d+)", q)
        if m:
            a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if a != 0:
                x = (c - b) / a
                return str(int(x)) if float(x).is_integer() else str(x)

        # 格式 3: ax = c
        m = re.fullmatch(r"(\d*)x=([+-]?\d+)", q)
        if m:
            coef = int(m.group(1)) if m.group(1) else 1
            val = int(m.group(2))
            if coef != 0:
                res = val / coef
                return str(int(res)) if res.is_integer() else str(res)

        # 🎯 FIX 3: 通用二次方程兜底
        result = self._solve_quadratic(q)
        if result: return result

        return None

    def _solve_quadratic(self, eq_str):
        """求解 ax^2 + bx + c = 0"""
        if '=0' not in eq_str: return None
        left = eq_str.split('=')[0].strip()
        if not left: return None

        left = left.replace('-', '+-')
        terms = [t.strip() for t in left.split('+') if t.strip()]
        a, b, c = 0, 0, 0
        for term in terms:
            if 'x^2' in term:
                coef = term.replace('x^2', '').strip()
                a = 1 if coef in ('', '+') else (-1 if coef == '-' else int(coef))
            elif 'x' in term:
                coef = term.replace('x', '').strip()
                b = 1 if coef in ('', '+') else (-1 if coef == '-' else int(coef))
            else:
                c = int(term)
        if a == 0: return None

        discriminant = b * b - 4 * a * c
        if discriminant < 0: return "No real solution."
        sqrt_d = math.sqrt(discriminant)
        x1, x2 = (-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a)
        fmt = lambda v: str(int(v)) if v == int(v) else str(round(v, 2))
        return f"x = {fmt(x1)}" if abs(x1 - x2) < 1e-9 else f"x = {fmt(x1)} or x = {fmt(x2)}"

    def extract_inline_context(self, query: str):
        text = query.strip()
        pattern = re.compile(r"Context\s*:\s*(.*?)\s*Q\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)
        m = pattern.match(text)
        return (m.group(1).strip(), m.group(2).strip()) if m else ("", text)

    def classify(self, query: str, history=None):
        if history is None: history = []
        
        current_name = self.extract_name_from_current_query(query)
        if current_name:
            if self.is_identity_question(query):
                return {"type": "identity", "subtype": "question", "value": f"You are {current_name}.", "handled": True}
            return {"type": "identity", "subtype": "set_name", "value": f"Nice to meet you, {current_name}.", "handled": True}

        if self.is_identity_question(query):
            name = self.extract_user_name(history)
            if name:
                return {"type": "identity", "subtype": "question", "value": f"You are {name}.", "handled": True}
            return {"type": "rag", "subtype": "", "value": {"context": "", "question": query}, "handled": False, "fallback_reason": "identity_not_found"}

        if self.is_equation_question(query):
            result = self.solve_equation(query)
            if result:
                return {"type": "tool", "subtype": "equation", "value": result, "handled": True}

        result = self.try_arithmetic(query)
        if result is not None:
            return {"type": "tool", "subtype": "arithmetic", "value": result, "handled": True}

        inline_context, actual_question = self.extract_inline_context(query)
        return {"type": "rag", "subtype": "", "value": {"context": inline_context, "question": actual_question}, "handled": False}
