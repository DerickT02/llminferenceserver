from llama_cpp import Llama

class InferenceEngine:
    def __init__(self, model_path: str, n_ctx: int = 2048):
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        out = self.llm(prompt, max_tokens=max_tokens)
        return out["choices"][0]["text"]