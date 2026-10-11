class OpenAIEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_answer(self, query: str, context: str) -> str:
        return f"Simulated answer to: '{query}' based on context: '{context[:100]}...'"