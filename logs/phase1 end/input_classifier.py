"""
输入类型判别类（完整版）
用于分类用户输入：Identity / Arithmetic / Equation / RAG
支持当前轮次名字提取 + 规则降级机制
"""
import re


class InputClassifier:
    """输入分类器 - 统一管理所有输入类型判断"""
    
    def __init__(self):
        # Identity 关键词
        self.identity_keys = [
            "who am i",
            "what is my name",
            "what's my name",
            "do you know my name",
            "remember my name",
            "tell me my name",
        ]
        
        # 名称黑名单（避免误判形容词）
        self.name_blacklist = {"happy", "fine", "tired", "ok", "good", "sad", "angry"}
        
        # 名称提取正则
        self.name_patterns = [
            re.compile(r"\bmy name is\s+([A-Za-z]{2,20})\b", re.IGNORECASE),
            re.compile(r"\bi am\s+([A-Za-z]{2,20})\b", re.IGNORECASE),
            re.compile(r"\bi'm\s+([A-Za-z]{2,20})\b", re.IGNORECASE),
        ]
    
    # =========================
    # 1️⃣ Identity 检测
    # =========================
    def is_identity_question(self, query: str) -> bool:
        """判断是否为身份询问"""
        q = query.lower()
        return any(k in q for k in self.identity_keys)
    
    def extract_user_name(self, history):
        """
        从历史中提取用户名
        :param history: [{"role": "user/assistant", "content": "..."}]
        :return: 用户名或 None
        """
        for msg in reversed(history):
            if msg.get("role") != "user":
                continue
            text = msg.get("content", "").strip()
            for p in self.name_patterns:
                m = p.search(text)
                if m:
                    name = m.group(1).strip()
                    if name.lower() not in self.name_blacklist:
                        return name
        return None
    
    def extract_name_from_current_query(self, query: str):
        """
        从当前输入中提取用户名（用于 "I am Tom. Who am I?" 场景）
        :param query: 当前用户输入
        :return: 用户名或 None
        """
        for p in self.name_patterns:
            m = p.search(query)
            if m:
                name = m.group(1).strip()
                if name.lower() not in self.name_blacklist:
                    return name
        return None
    
    # =========================
    # 2️⃣ 算术检测
    # =========================
    def try_arithmetic(self, query: str):
        """
        检测并计算简单算术表达式
        :return: 计算结果字符串 或 None
        """
        q = query.strip().lower()
        # 去除常见前缀
        q = re.sub(r"^(please\s+)?(count|calculate|compute|what is|what's)\s+", "", q).strip()
        q = q.replace("=?", "").replace("= ?", "").replace("?", "").strip()
        
        # 长度限制
        if len(q) > 50:
            return None
        
        # 只包含数字和运算符（安全白名单）
        if re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", q):
            try:
                val = eval(q, {"__builtins__": {}}, {})
                if isinstance(val, float) and val.is_integer():
                    return str(int(val))
                return str(val)
            except:
                return None
        return None
    
    # =========================
    # 3️⃣ 方程检测
    # =========================
    def is_equation_question(self, query: str):
        """判断是否为方程问题"""
        q = query.lower()
        return "x" in q and "=" in q
    
    def solve_equation(self, query: str):
        """
        求解简单方程
        支持格式：
        - ax + b = c
        - x^2 = n 或 x*x = n
        :return: 解字符串 或 None
        """
        q = query.strip().lower().replace(" ", "")
        
        # ax + b = c
        m = re.fullmatch(r"([+-]?\d+)\*x([+-]\d+)=([+-]?\d+)", q)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            c = int(m.group(3))
            if a != 0:
                x = (c - b) / a
                if float(x).is_integer():
                    return str(int(x))
                return str(x)
        
        # x^2 = n 或 x*x = n
        m = re.fullmatch(r"(x\*x|x\^2)=([+-]?\d+)", q)
        if m:
            n = int(m.group(2))
            if n < 0:
                return "No real solution."
            r = int(n ** 0.5)
            if r * r == n:
                return f"x = {r} or x = -{r}" if r != 0 else "x = 0"
            return "No integer solution."
        
        return None
    
    # =========================
    # 4️⃣ Inline Context 提取
    # =========================
    def extract_inline_context(self, query: str):
        """
        提取内联上下文和问题
        格式: Context: ... Q: ...
        :return: (context, question)
        """
        text = query.strip()
        pattern = re.compile(
            r"Context\s*:\s*(.*?)\s*Q\s*:\s*(.+)$",
            re.IGNORECASE | re.DOTALL
        )
        m = pattern.match(text)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return "", text
    
    # =========================
    # 🎯 主分类函数（核心修改）
    # =========================
        # =========================
    # 🎯 主分类函数（完整修复版）
    # =========================
    def classify(self, query: str, history=None):
        """
        主分类器
        :param query: 用户输入
        :param history: 对话历史 [{"role": "user/assistant", "content": "..."}]
        :return: {
            "type": "identity" | "equation" | "arithmetic" | "rag",
            "value": 处理结果或提取的数据,
            "handled": True/False,  # 是否已处理完成
            "fallback_reason": str  # 可选，降级原因
        }
        """
        if history is None:
            history = []
        
        # =========================
        # 1️⃣ Identity（修复：当前输入优先）
        # =========================
        if self.is_identity_question(query):
            # 优先从当前输入中提取（允许用户改名/重新介绍）
            name = self.extract_name_from_current_query(query)
            
            # 如果当前输入没有名字，再去历史里找
            if not name:
                name = self.extract_user_name(history)
            
            if name:
                return {
                    "type": "identity",
                    "value": f"You are {name}.",
                    "handled": True
                }
            else:
                # 都没有，降级 RAG
                return {
                    "type": "rag",
                    "value": {"context": "", "question": query},
                    "handled": False,
                    "fallback_reason": "identity_not_found"
                }
        
        # =========================
        # 2️⃣ 方程（确定性工具）
        # =========================
        if self.is_equation_question(query):
            result = self.solve_equation(query)
            return {
                "type": "equation",
                "value": result if result else "I don't know.",
                "handled": True
            }
        
        # =========================
        # 3️⃣ 算术（确定性工具）
        # =========================
        result = self.try_arithmetic(query)
        if result is not None:
            return {
                "type": "arithmetic",
                "value": result,
                "handled": True
            }
        
        # =========================
        # 4️⃣ RAG（开放域问答）
        # =========================
        inline_context, actual_question = self.extract_inline_context(query)
        return {
            "type": "rag",
            "value": {
                "context": inline_context,
                "question": actual_question
            },
            "handled": False
        }
