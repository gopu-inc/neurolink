"""Vérifie le découpage des balises Bund (<|thinking|> / <|actual_response|>) côté serveur."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurolite.server import text_events, merge_options  # noqa: E402


class FakeModel:
    def __init__(self, text):
        self.text = text

    def stream(self, prompt, max_tokens=256, **kw):
        for ch in self.text[:max_tokens]:
            yield ch


SAMPLE = ("<|thinking|>\nIl faut additionner deux nombres.\n</|thinking|>\n"
          "<|actual_response|>\n```python\ndef add(a, b):\n    return a + b\n```\n"
          "</|actual_response|>\n<|end|>\n<|user|>\nbruit")
opts = merge_options({}, {"num_predict": 4000})
events = list(text_events(FakeModel(SAMPLE), "", opts))
thinking = "".join(t for k, t in events if k == "thinking")
content = "".join(t for k, t in events if k == "content")
done = [t for k, t in events if k == "done"]
assert "Il faut additionner" in thinking, thinking
assert "<|thinking|>" not in thinking and "</|thinking|>" not in thinking
assert "def add(a, b)" in content, content
assert "<|" not in content and "|>" not in content, content
assert "bruit" not in content and "bruit" not in thinking
assert done == ["stop"], done
print("thinking :", repr(thinking.strip()))
print("content  :", repr(content.strip()))
print("TEST BUND OK")
