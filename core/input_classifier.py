import re

class InputClassifier:
    """输入分类器 - 纯路由器，不再包含任何业务计算逻辑"""
    def __init__(self):
        self.greetings = [
            "hello", "hi", "hey", "hola", "你好", "您好", "早上好", "下午好", "晚上好",
            "good morning", "good afternoon", "good evening", "howdy", "greetings",
        ]
        self.greeting_responses = [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hey! Feel free to ask me anything.",
            "你好！有什么我可以帮助你的吗？",
        ]
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

    def is_weather_question(self, query: str) -> str:
        """判断是否为天气查询，返回城市名或 None"""
        q = query.lower()
        if not any(k in q for k in ["weather", "天气", "温度", "temperature", "气温"]):
            return None

        import re
        patterns = [
            r"weather (?:in|for|at|of) ([a-z\s]+?)(?:\?|$|,|\.)",
            r"([a-z\s]+?) weather",
            r"([a-z\s]+?)的?天气",
            r"天气.*?([a-z]+)",
        ]
        for p in patterns:
            m = re.search(p, q)
            if m:
                city = m.group(1).strip()
                if city and city not in ["the", "a", "is", "what", "how", "about", "today", "now", "current"]:
                    return city.title()

        return None

    def _normalize_math_text(self, text):
        t = text.lower()
        for word, symbol in self.math_word_map.items():
            t = t.replace(word, symbol)
        return t

    def is_greeting(self, query: str) -> bool:
        q = query.lower().strip()
        if len(q.split()) > 4:
            return False
        return any(q == g or q.startswith(g + " ") for g in self.greetings)

    def is_identity_question(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in self.identity_keys)

    def is_bot_question(self, query: str) -> bool:
        """拦截询问 AI 自身的问题"""
        q = query.lower()
        return any(k in q for k in ["your name", "who are you", "你是谁", "你叫什么"])

    def is_preference_statement(self, query: str):
        """识别偏好陈述，返回 (key, value) 或 None"""
        q = query.lower().strip()
        
        patterns = [
            (r"i (?:like|love|enjoy) (.+)", "like"),
            (r"i (?:hate|dislike) (.+)", "hate"),
            (r"my favorite (?:color|colour) is (.+)", "favorite_color"),
            (r"my favorite (?:food) is (.+)", "favorite_food"),
            (r"my favorite (?:sport) is (.+)", "favorite_sport"),
            (r"i (?:also )?like (.+)", "like"),
        ]
        
        for pattern, key in patterns:
            match = re.search(pattern, q)
            if match:
                value = match.group(1).strip()
                return (key, value)
        return None

    def is_preference_question(self, query: str) -> str:
        """识别偏好查询，返回 key 或 None"""
        q = query.lower()
        
        if any(k in q for k in ["what do i like", "what i like", "my preference"]):
            return "like"
        if any(k in q for k in ["what do i hate", "what i hate"]):
            return "hate"
        if any(k in q for k in ["favorite color", "favourite color"]):
            return "favorite_color"
        if any(k in q for k in ["favorite food", "favourite food"]):
            return "favorite_food"
        if any(k in q for k in ["favorite sport", "favourite sport"]):
            return "favorite_sport"
        if any(k in q for k in ["what sports", "what do i play"]):
            return "like"
        if "everything you know about me" in q or "all about me" in q:
            return "all"
        return None

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

    def is_convert_question(self, query: str) -> bool:
        """判断是否为单位转换问题"""
        q = query.lower()
        patterns = [
            r'[\d.]+\s*(km|mile|miles|kg|pound|pounds|jin)\s*(to|转)\s*(km|mile|miles|kg|pound|pounds|jin)',
            r'[\d.]+\s*(c|f)\s*to\s*(c|f)',
            r'convert\s+[\d.]+\s*(km|mile|miles|kg|pound|pounds|c|f)',
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

        import random

        if self.is_greeting(query):
            return {"type": "greeting", "subtype": "", "value": random.choice(self.greeting_responses), "handled": True}

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

        # ================= 2. 偏好处理 =================
        pref_stmt = self.is_preference_statement(query)
        if pref_stmt:
            key, value = pref_stmt
            return {"type": "preference", "subtype": "set", "key": key, "value": value, "handled": False}
        
        pref_q = self.is_preference_question(query)
        if pref_q:
            return {"type": "preference", "subtype": "get", "key": pref_q, "handled": False}

        # ================= 3. 时间类（细分工具） =================
        time_subtype = self.is_time_question(query)
        if time_subtype:
            return {"type": "agent", "subtype": time_subtype, "value": query, "handled": False}

        weather_city = self.is_weather_question(query)
        if weather_city:
            return {"type": "agent", "subtype": "weather", "value": weather_city, "handled": False}

        # ================= 4. 方程（交给 Agent 的 solve 工具） =================
        if self.is_equation_question(query):
            return {"type": "agent", "subtype": "equation", "value": query, "handled": False}

        # ================= 5. 算术（交给 Agent 的 calc 工具） =================
        if self.is_arithmetic_question(query):
            return {"type": "agent", "subtype": "arithmetic", "value": query, "handled": False}

        # ================= 6. 进制转换（单工具） =================
        if self.is_base_convert_question(query):
            return {"type": "agent", "subtype": "base_convert", "value": query, "handled": False}

        # ================= 6.5 单位转换 =================
        if self.is_convert_question(query):
            return {"type": "agent", "subtype": "convert", "value": query, "handled": False}

        # ================= 7. 多工具问题（走 ReAct 循环） =================
        if self.is_multi_tool_question(query):
            return {"type": "agent", "subtype": "multi_tool", "value": query, "handled": False}

        if len(query.split()) <= 2:
            q = query.lower().strip()
            if q in ["hello", "hi", "hey", "hola", "你好", "您好"]:
                return {"type": "greeting", "subtype": "", "value": "Hello! How can I help you today?", "handled": True}
            return {"type": "chat", "subtype": "short", "value": "Could you provide more details about your question?", "handled": True}

        inline_context, actual_question = self.extract_inline_context(query)
        return {"type": "rag", "subtype": "", "value": {"context": inline_context, "question": actual_question}, "handled": False}
