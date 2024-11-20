from typing import Optional
from class_registry import ClassRegistry


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)

    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None, **kwargs):
        if model_name is None:
            model_name = "allam-13b"
        return cls.registry.get(model_name, **kwargs)
