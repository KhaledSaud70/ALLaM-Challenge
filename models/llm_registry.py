from typing import Any, Dict, List, Optional, Type
from class_registry import ClassRegistry
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI



class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)

    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, llm_provider: str, model_name: str, **kwargs):
        """
        Fetches an LLM class instance based on the provider and model name.
        
        Args:
            llm_provider: The name of the provider ('custom', 'anthropic', 'openai', etc.)
            model_name: The name of the model or key.
            kwargs: Additional arguments for model initialization.
        """
        if llm_provider == "custom":
            # Handle custom registered models (ALLaM and FakeChatModel)
            llm_class = cls.registry.get(model_name, **kwargs)
            if llm_class is None:
                raise ValueError(f"Custom LLM '{model_name}' not found in registry.")
            return llm_class(**kwargs)
        elif llm_provider == "anthropic":
            return ChatAnthropic(model=model_name, **kwargs)
        elif llm_provider == "openai":
            return ChatOpenAI(model=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported provider '{llm_provider}'.")