
# -*- coding: utf-8 -*-
"""
Pluggable model client.
- LocalEchoClient: returns deterministic placeholders (for offline demo)
- OpenAIClient: template code (commented) to call OpenAI responses API
You can implement your own provider by subclassing BaseClient.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseClient(ABC):
    @abstractmethod
    def run(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        ...
"""
class LocalEchoClient(BaseClient):
    def run(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # Minimal offline stub: echoes a trivial CoNLL-U-ish parse or returns sentence back
        mode = kwargs.get("mode", "parse")
        if mode == "parse":
            # naive tab lines for tokens split by space
            import re
            import itertools
            import unicodedata as ud
            # extract the sentence after '문장:'
            text = user_prompt.split("문장:", 1)[-1].strip()
            tokens = text.strip().rstrip(".").split()
            rows = []
            for i, tok in enumerate(tokens, start=1):
                rows.append(f"{i}\t{tok}\t{tok}\tX\t0\tdep")
            return "\n".join(rows)
        else:
            # rewrite mode: just return the original sentence (identity baseline)
            lines = [ln for ln in user_prompt.splitlines() if ln.startswith("문장:")]
            sent = lines[0].split("문장:",1)[-1].strip() if lines else user_prompt
            return sent
"""
# Example OpenAI implementation (uncomment and fill your API key to use)
class OpenAIClient(BaseClient):
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        from openai import OpenAI
        self.client = OpenAI(api_key="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")
        self.model = model
    def run(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt},
            ],
            temperature=kwargs.get("temperature", 0.2),
            top_p=kwargs.get("top_p", 0.9),
        )
        return resp.choices[0].message.content.strip()
