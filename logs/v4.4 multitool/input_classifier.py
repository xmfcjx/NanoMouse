import re

class InputClassifier:
    """输入分类器 - 纯路由器，不再包含任何业务计算逻辑"""
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
        # 数学自然语言 -> 符号映射（仅用于辅助判断）
        self.math_word_map = {
            "multiplied by": "*", "times": "*", "plus": "+", "minus": "-", "divided by": "/",
            "solve": "", "find": "", "calculate": "", "what is": "", "what's": "",
        }

    # ==================== 判断方法 ====================

    def is_time_question(self, query: str) -> str:
        """返回具体工具子类型字符串，None 表示不是时间类"""
        q = query.lower()
        if any(k in q for k in ["星期", "星期几", "what day", "weekday"]):
            return "weekday"
        if any(k in q for k in ["几点", "what time", "current time"]):
            return "time"
        if any(k in q for k in ["几号", "日期", "what date", "today", "今天", "几月几日"]):
            return "date_today"
        if any(k in q for k in ["相差几天", "days between", "隔了几天"]):
            return "days_between"
        return None

    def _normalize_math_text(self, text):
        t = text.lower()
        for word, symbol in self.math_word_map.items():
            t = t.replace(word, symbol)
        return t

    def is_identity_question(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in self.identity_keys)

    def is_bot_question(self, query: str) -> bool:
        """拦截询问 AI 自身的问题"""
        q = query.lower()
        return any(k in q for k in ["your name", "who are you", "你是谁", "你叫什么"])

    def extract_user_name(self, history):
        for msg in reversed(history):
            if msg.get("role") != "user": continue
            text = msg.get("content", "").strip()
            for p in self.name_patterns:
                m = p.search(text)
                if m:
                    name = m.group(1).strip()
                    if name.lower() not in self.name_blacklist:
                        return name
        return None

    def is_arithmetic_question(self, query: str) -> bool:
        """判断是否为纯数学算术表达式（只判断，不计算）"""
        q_orig = query.strip().lower()
        q_norm = self._normalize_math_text(q_orig)
        for q in [q_orig, q_norm]:
            q = re.sub(r"^(please\s+)?(count|calculate|compute|what is|what's)\s+", "", q).strip()
            q = q.replace("=?", "").replace("= ?", "").replace("?", "").strip()
            if len(q) > 50: continue
            if re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", q):
                return True
        return False

    def extract_arithmetic_expr(self, query: str) -> str:
        """提取纯算术表达式（交给 Agent 的 calc 工具执行）"""
        q_orig = query.strip().lower()
        q_norm = self._normalize_math_text(q_orig)
        for q in [q_orig, q_norm]:
            q = re.sub(r"^(please\s+)?(count|calculate|compute|what is|what's)\s+", "", q).strip()
            q = q.replace("=?", "").replace("= ?", "").replace("?", "").strip()
            if len(q) > 50: continue
            if re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", q):
                return q
        return ""

    def is_equation_question(self, query: str):
        """判断是否为方程题（包含 x 和 =）"""
        q = query.lower()
        q_norm = self._normalize_math_text(q)
        return ("x" in q and "=" in q) or ("x" in q_norm and "=" in q_norm)

    def is_multi_tool_question(self, query: str) -> bool:
        """判断是否为多工具问题（需要串联多个工具）"""
        q = query.lower()
        
        # 必须包含 "then" 才算多工具问题
        if "then" not in q:
            return False
        
        multi_tool_patterns = [
            r'\bthen\b.*\b(to|convert|bin|hex|binary|hexadecimal)\b',
            r'\bthen\b.*\b(weekday|day|time)\b',
        ]
        
        for pattern in multi_tool_patterns:
            if re.search(pattern, q):
                return True
        return False
    
    def is_base_convert_question(self, query: str) -> bool:
        """判断是否为进制转换问题（单工具）"""
        q = query.lower()
        patterns = [
            r'\d+\s*to\s*(bin|binary|hex|hexadecimal|oct|octal|dec|decimal)',
            r'convert\s+\d+\s*to\s*(bin|binary|hex|hexadecimal|oct|octal|dec|decimal)',
        ]
        for pattern in patterns:
            if re.search(pattern, q):
                return True
        return False

    def extract_inline_context(self, query: str):
        text = query.strip()
        pattern = re.compile(r"Context\s*:\s*(.*?)\s*Q\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)
        m = pattern.match(text)
        return (m.group(1).strip(), m.group(2).strip()) if m else ("", text)

    # ==================== 主路由 ====================
    def classify(self, query: str, history=None):
        if history is None: history = []

        # ================= 1. 身份处理 =================
        # 内联提取当前query里的名字
        current_name = None
        for p in self.name_patterns:
            m = p.search(query)
            if m:
                name = m.group(1).strip()
                if name.lower() not in self.name_blacklist:
                    current_name = name
                    break

        if current_name:
            if self.is_identity_question(query):
                return {"type": "identity", "subtype": "question", "value": f"You are {current_name}.", "handled": True}
            return {"type": "identity", "subtype": "set_name", "value": f"Nice to meet you, {current_name}.", "handled": True}

        if self.is_identity_question(query):
            name = self.extract_user_name(history)
            if name:
                return {"type": "identity", "subtype": "question", "value": f"You are {name}.", "handled": True}
            return {"type": "identity", "subtype": "question", "value": "I don't know your name yet.", "handled": True}

        # 🎯 拦截询问 AI 自身的问题，直接硬回复，不丢给 RAG
        if self.is_bot_question(query):
            return {"type": "identity", "subtype": "bot", "value": "I am an AI assistant.", "handled": True}

        # ================= 2. 时间类（细分工具） =================
        time_subtype = self.is_time_question(query)
        if time_subtype:
            return {"type": "agent", "subtype": time_subtype, "value": query, "handled": False}

        # ================= 3. 方程（交给 Agent 的 solve 工具） =================
        if self.is_equation_question(query):
            return {"type": "agent", "subtype": "equation", "value": query, "handled": False}

        # ================= 4. 算术（交给 Agent 的 calc 工具） =================
        if self.is_arithmetic_question(query):
            return {"type": "agent", "subtype": "arithmetic", "value": query, "handled": False}

        # ================= 5. 进制转换（单工具） =================
        if self.is_base_convert_question(query):
            return {"type": "agent", "subtype": "base_convert", "value": query, "handled": False}

        # ================= 6. 多工具问题（走 ReAct 循环） =================
        if self.is_multi_tool_question(query):
            return {"type": "agent", "subtype": "multi_tool", "value": query, "handled": False}

        # ================= 7. RAG 兜底 =================
        inline_context, actual_question = self.extract_inline_context(query)
        return {"type": "rag", "subtype": "", "value": {"context": inline_context, "question": actual_question}, "handled": False}
