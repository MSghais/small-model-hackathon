# agent

Hermes-style skill agent on top of the local `inference` backends.

```python
from agent.runner import AgentRunner
from inference.factory import get_backend

runner = AgentRunner()
result = runner.run_education_pptx(
    topic="Photosynthesis",
    grade="6",
    slide_count=5,
    model_key="minicpm5-1b",
    backend=get_backend("minicpm5-1b"),
)
```
