from collections import deque
class Memory:
    def __init__(self, max_history=20):
        self.head = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]
        self.history = deque(maxlen=max_history)

    def add_user_message(self, message):
        self.history.append({
            "role": "user",
            "content": message
        })

    def add_assistant_message(self, message):
        self.history.append({
            "role": "assistant",
            "content": message
        })

    def get_messages(self):
        return self.head + list(self.history)
    #Conversation History:
    # [{'role': 'user', 'content': 'my name is tom'}, ...]
    
    def clear(self):
        self.history.clear()
