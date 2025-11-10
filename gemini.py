```python
from strands import Agent
from strands.models.gemini import GeminiModel
from strands_tools import calculator
model = GeminiModel(
    # **model_config
    model_id="gemini-2.5-flash",
    params={
        # some sample model parameters 
        "temperature": 0.7,
        "max_output_tokens": 2048,
        "top_p": 0.9,
        "top_k": 40
    }
)

```