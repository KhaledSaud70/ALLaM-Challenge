from typing import Type
from class_registry import ClassRegistry

from langchain_core.prompts import PromptTemplate


class PromptTemplateRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)

    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, name: str, *args, **kwargs) -> PromptTemplate:
        return cls.registry.get(name, *args, **kwargs)

    @classmethod
    def get_class(cls, name: str) -> Type:
        return cls.registry.get_class(name)
