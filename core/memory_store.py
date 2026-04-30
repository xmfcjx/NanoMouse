"""
统一记忆系统
正则优先 + LLM 兜底
"""
import json
import re

class MemoryStore:
    def __init__(self, llm=None, embedding_model=None):
        self.llm = llm
        self.embedding_model = embedding_model
        self.facts = []
        self.name = None
    
    def extract_fact(self, query: str) -> dict:
        q = query.lower().strip()
        
        if q.endswith('?') or q.startswith('what') or q.startswith('who') or q.startswith('do you'):
            if any(k in q for k in ['my name', 'who am i', 'who i am']):
                return {"action": "get", "key": "name"}
            if any(k in q for k in ['what do you know', 'about me', 'tell me about']):
                return {"action": "get", "key": "all"}
            return {"action": "none"}
        
        set_patterns = [
            (r"my name is ([a-z]+)", "name"),
            (r"i am ([a-z]+)$", "name"),
            (r"i'm ([a-z]+)$", "name"),
            (r"call me ([a-z]+)", "name"),
            (r"i am (?:playing|doing|watching|reading|eating) (.+)", "activity"),
            (r"i (?:like|love|enjoy) (.+)", "likes"),
            (r"i (?:hate|dislike) (.+)", "hates"),
            (r"my favorite (.+?) is (.+)", "favorite"),
            (r"i have (?:a |an )?(.+)", "has"),
            (r"i work (?:at|for|in) (.+)", "work"),
            (r"i live in (.+)", "location"),
            (r"i'm (\d+) years? old", "age"),
            (r"i am (\d+) years? old", "age"),
            (r"my (?:job|occupation) is (.+)", "job"),
            (r"my major is (.+)", "major"),
            (r"i study (.+)", "studies"),
            (r"i (?:go to|attend) (.+)", "school"),
            (r"i am a (.+)", "status"),
            (r"i play (.+)", "plays"),
            (r"i want to (.+)", "wants"),
        ]
        
        for pattern, key in set_patterns:
            m = re.search(pattern, q)
            if m:
                value = m.group(1).strip()
                if key == "name":
                    blacklist = ["happy", "fine", "tired", "ok", "good", "sad", "sure", "sorry",
                                 "a student", "a teacher", "playing", "working", "studying",
                                 "going", "doing", "reading", "writing", "eating", "sleeping",
                                 "running", "walking", "talking", "thinking", "watching"]
                    if value.lower() in blacklist:
                        continue
                if key == "favorite":
                    return {"action": "set", "key": f"favorite_{m.group(1)}", "value": m.group(2)}
                return {"action": "set", "key": key, "value": value}
        
        if self.llm:
            return self._extract_with_llm(query)
        
        return {"action": "none"}
    
    def _extract_with_llm(self, query: str) -> dict:
        prompt = f"""Extract personal info from the message. Return JSON only.

Message: "{query}"

Valid keys: name, location, work, likes, hates, age, favorite_color, favorite_food, has, studies, job, wants, plays, major, status, school

Examples:
- "I'm called Alice" -> {{"action": "set", "key": "name", "value": "Alice"}}
- "I live in Tokyo" -> {{"action": "set", "key": "location", "value": "Tokyo"}}
- "I work at Google" -> {{"action": "set", "key": "work", "value": "Google"}}
- "I like music" -> {{"action": "set", "key": "likes", "value": "music"}}
- "I'm 25 years old" -> {{"action": "set", "key": "age", "value": "25"}}
- "What is my name?" -> {{"action": "get", "key": "name"}}
- "Who am I?" -> {{"action": "get", "key": "all"}}
- "Hello world" -> {{"action": "none"}}

JSON:"""
        
        try:
            response = self.llm.generate(prompt).strip()
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"[MemoryStore] LLM extraction failed: {e}")
        return {"action": "none"}
    
    def process(self, query: str) -> tuple:
        result = self.extract_fact(query)
        action = result.get("action", "none")
        
        if action == "set":
            key = result.get("key", "")
            value = result.get("value", "")
            self._add_fact(key, value)
            if key == "name":
                return f"Nice to meet you, {value}!", "memory", "set"
            display_key = "possessions" if key == "has" else key
            return f"Got it! I'll remember that your {display_key} {('are' if key == 'has' else 'is')} {value}.", "memory", "set"
        
        elif action == "get":
            facts = self.get_all_facts()
            if facts:
                lines = []
                for k, v in facts:
                    if k == "has":
                        lines.append(f"You have {v}.")
                    elif k == "name":
                        lines.append(f"Your name is {v}.")
                    else:
                        lines.append(f"Your {k} is {v}.")
                return "Here's what I know about you: " + " ".join(lines), "memory", "get"
            return "I don't know anything about you yet.", "memory", "get"
        
        return None, None, None
    
    def _add_fact(self, key: str, value: str):
        if key == "name":
            self.name = value
            return
        
        for i, (k, _) in enumerate(self.facts):
            if k == key:
                self.facts[i] = (key, value)
                return
        
        self.facts.append((key, value))
    
    def get_all_facts(self) -> list:
        result = []
        if self.name:
            result.append(("name", self.name))
        result.extend(self.facts)
        return result
    
    def clear(self):
        self.facts = []
        self.name = None
