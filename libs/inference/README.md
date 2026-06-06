# inference

Swappable local inference backends (`llama_cpp` default, `transformers` optional extra).

```python
from inference.factory import get_backend

backend = get_backend()
backend.load()
reply = backend.chat([{"role": "user", "content": "Hello!"}])
```
