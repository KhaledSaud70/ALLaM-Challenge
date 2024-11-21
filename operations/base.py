from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, Field

from models import LLMRegistry

AgentState = TypeVar("AgentState", bound=BaseModel)


class Operation(ABC, BaseModel, Generic[AgentState]):
    llm_provider: str = Field(default="custom")
    llm_name: str = Field(default="FakeChatModel")
    llm_params: dict = Field(default_factory=dict)
    llm_registry: LLMRegistry = Field(default_factory=LLMRegistry)
    system_prompt: str

    class Config:
        arbitrary_types_allowed = True

    def execute(self, state: AgentState) -> AgentState:
        """Synchronous execution method"""
        messages = self._prepare_messages(state)
        response = self.invoke(messages)
        return self.process_response(response, state)

    async def aexecute(self, state: AgentState) -> AgentState:
        """Asynchronous execution method"""
        messages = self._prepare_messages(state)
        response = await self.ainvoke(messages)
        return self.process_response(response, state)

    def _prepare_messages(self, state: AgentState) -> List[BaseMessage]:
        """Prepare messages for LLM invocation"""
        messages = self.get_messages(state)
        if self.system_prompt:
            if not isinstance(messages, list):
                messages = [messages]
            formatted_system_prompt = self.format_system_prompt(state)
            messages = [SystemMessage(content=formatted_system_prompt)] + messages
        return self.filter_messages(messages)

    def format_system_prompt(self, state: AgentState) -> str:
        """Optional method to format system prompt with state variables.
        Can be overridden by child classes if they need custom formatting."""
        return self.system_prompt
    
    def _get_llm(self) -> Any:
        return self.llm_registry.get(llm_provider=self.llm_provider, model_name=self.llm_name, **self.llm_params)

    @abstractmethod
    def get_messages(self, state: AgentState) -> List[BaseMessage]:
        """Abstract method to get messages from state"""
        pass

    @abstractmethod
    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """Synchronous abstract method for the LLM call logic"""
        pass

    @abstractmethod
    async def ainvoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """Asynchronous abstract method for the LLM call logic"""
        pass

    @abstractmethod
    def process_response(self, response: str, state: AgentState) -> Dict[str, Any]:
        """Process the LLM's response into a structured format."""
        pass

    def filter_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Filter conversation history messages."""
        return messages
