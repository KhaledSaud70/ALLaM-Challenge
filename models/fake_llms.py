from itertools import cycle
from typing import List, Optional

from langchain.schema import AIMessage
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel, FakeMessagesListChatModel

from .llm_registry import LLMRegistry


@LLMRegistry.register("FakeChatModel")
class FakeChatModel(GenericFakeChatModel):
    def __init__(self, messages: Optional[List[str]] = None, **kwargs):
        if messages is None:
            messages = ["This is a fake response."]
        ai_messages = [AIMessage(content=msg) for msg in messages]
        super().__init__(messages=cycle(ai_messages), **kwargs)


@LLMRegistry.register("FakeMessagesListChatModel")
class FakeMessagesListChatModel(FakeMessagesListChatModel):
    def __init__(self, messages: Optional[List[str]] = None, **kwargs):
        if messages is None:
            messages = ["This is a fake response from list."]
        super().__init__(responses=[AIMessage(content=msg) for msg in messages], **kwargs)


@LLMRegistry.register("FakeListChatModel")
class FakeListChatModel(FakeListChatModel):
    def __init__(self, messages: Optional[List[str]] = None, **kwargs):
        if messages is None:
            messages = ["This is a fake list chat model response."]
        super().__init__(responses=messages, **kwargs)


if __name__ == "__main__":
    llm = FakeChatModel(messages=[("human", "Hi"), ("human", "Good bye")])
    print(llm.invoke(("human", "Khaled")))
    print(llm.invoke(("human", "Khaled")))
