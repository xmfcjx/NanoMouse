import re
from typing import Dict, Callable, Optional

class ReActAgent:
    def __init__(self, llm_instance: object, max_steps: int = 5):
        self.llm_instance=llm_instance
        self.max_steps=max_steps
        self.tools={}

        self.register_tool(
            name="Python_REPL", 
            func=self._python_repl, 
            description="用于执行Python数学表达式。输入必须是合法的Python代码字符串，如果是计算结果，请务必使用 print() 输出。例如：'print(3 * 8)'"
        )
        #注册一个专门用于解数学题的tool
        self.system_prompt = """你是一个逻辑推理助手。你必须严格按照以下格式回复：
Thought: 我需要思考的问题
Action: 工具名称
Action Input: 工具的输入参数
Observation: (工具返回的结果会在这里显示)
Thought: 我现在知道最终答案了
Final Answer: 最终答案的具体内容

可用工具：
{tools_desc}

注意：如果没有可用的工具，或者不需要工具，直接输出 Final Answer。"""
    
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

    def register_tool(self, name: str, func: Callable, description: str) -> None:
        self.tools[name] = {
            "func": func,
            "desc": description
        }

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        if self.tools.get(tool_name) is None:
            return "Error: Tool not found."
        try:
            return self.tools[tool_name]["func"](tool_input)
        except Exception as e:
            return f"tool_input is error: {str(e)}"

    def _parse_response(self, text: str) -> Dict[str, Optional[str]]:
        final_answer=re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE).search(text)
        if final_answer is not None:
            return {"type": "final_answer", "content": final_answer.group(1)}
        
        action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
        input_match = re.search(r"Action Input:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if action_match and input_match:
            return {"type": "action", "name": action_match.group(1),"input":input_match.group(1)}
        
        return {"type": "error", "content": None}

    def run(self, question: str) -> str:
        tools_desc = "\n".join([f"- {name}: {info['desc']}" for name, info in self.tools.items()])
        # 拼接完整的 prompt
        prompt_context = self.system_prompt.format(tools_desc=tools_desc)
        prompt_context += f"\nQuestion: {question}\nThought:"
        for i in range(self.max_steps):
            ans = self.llm_instance.generate(prompt_context,max_new_tokens=128) 
            prompt_context+=ans
            response=self._parse_response(ans)
            if response["type"]=="final_answer":
                return response["content"]
            elif response["type"]=="action":
                res=self._execute_tool(response["name"], response["input"])
                prompt_context += f"\nObservation: {res}\nThought:"
            else :
                return "模型输出格式错误"
        return "Max steps reached."