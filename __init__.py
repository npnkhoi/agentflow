from agentflow.pipeline import Pipeline, Stage
from agentflow.processors import LLMProcessor, CopyProcessor
from agentflow.models import OpenAILLM, GeminiVLM
from agentflow.typing import SampleOutput, RefinedOutput, CountOutput

# Pre-register built-in output types
Pipeline.register_type("SampleOutput", SampleOutput)
Pipeline.register_type("RefinedOutput", RefinedOutput)
Pipeline.register_type("CountOutput", CountOutput)

# Pre-register built-in processors
Pipeline.register_processor("LLMProcessor", LLMProcessor)
Pipeline.register_processor("CopyProcessor", CopyProcessor)

# Pre-register built-in model backends
Pipeline.register_model_backend("openai", OpenAILLM)
Pipeline.register_model_backend("gemini", GeminiVLM)
