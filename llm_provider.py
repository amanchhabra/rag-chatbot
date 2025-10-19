import os
from flask import session
from openai import AuthenticationError, OpenAI
import requests

class LLMProvider:
    def __init__(self, provider="openai"):
        self.provider = provider.lower()
        self.api_key = (
            session.get(f"{self.provider}_key") or
            os.getenv(f"{self.provider.upper()}_API_KEY")
        )

        if self.provider == "openai":
            self.client = OpenAI(api_key=self.api_key)

    def embed(self, texts):
        try:    
            if self.provider == "openai":
                resp = self.client.embeddings.create(
                    model="text-embedding-3-small", input=texts
                )
                return [d.embedding for d in resp.data]

            elif self.provider == "ollama":
                # Example: Ollama API (local models)
                out = []
                for text in texts:
                    r = requests.post(
                        "http://localhost:11434/api/embeddings",
                        json={"model": "nomic-embed-text", "prompt": text}
                    )
                    out.append(r.json()["embedding"])
                return out

            elif self.provider == "anthropic":
                # Anthropic doesn’t provide embeddings → fallback / raise
                raise NotImplementedError("Anthropic embeddings not supported")

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except AuthenticationError:
            raise ValueError("Invalid OpenAI API key. Please check your settings.") 

    def chat(self, messages, model=None):
        try:
            if self.provider == "openai":
                return self.client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=messages
                ).choices[0].message.content

            elif self.provider == "ollama":
                r = requests.post(
                    "http://localhost:11434/api/chat",
                    json={"model": model or "llama2", "messages": messages}
                )
                return r.json()["message"]["content"]

            elif self.provider == "anthropic":
                r = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": self.api_key},
                    json={"model": model or "claude-3-opus", "messages": messages}
                )
                return r.json()["content"][0]["text"]

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except AuthenticationError:
            raise ValueError("Invalid OpenAI API key. Please check your settings.")    
