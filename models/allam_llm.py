import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_ibm import ChatWatsonx

from .llm_registry import LLMRegistry

load_dotenv()


@LLMRegistry.register("allam-13b")
class ALLaM:
    DEFAULT_MODEL_ID = "sdaia/allam-1-13b-instruct"
    DEFAULT_URL = "https://eu-de.ml.cloud.ibm.com"
    DEFAULT_PARAMETERS = {
        "decoding_method": "sample",
        "max_new_tokens": 500,
        "min_new_tokens": 1,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 1,
    }

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.api_key = os.getenv("WATSONX_APIKEY")

        self.parameters = {**self.DEFAULT_PARAMETERS, **(parameters or {})}

        self.llm = ChatWatsonx(
            model_id=self.DEFAULT_MODEL_ID,
            url=self.DEFAULT_URL,
            project_id=self.project_id,
            api_key=self.api_key,
            params=self.parameters,
            **kwargs,
        )

    def _format_messages(self, messages: List[Union[Dict[str, str], BaseMessage]]) -> List[BaseMessage]:
        """
        Format input messages for the ALLaM model.

        Args:
            messages (List[Union[Dict[str, str], Any]]): List of message dictionaries (OpenAI format) or LangChain message objects.

        Returns:
            List[BaseMessage]: Formatted messages.
        """

        if not isinstance(messages, list):
            raise TypeError("Messages should be a list.")

        def format_message(content: str, role: str) -> BaseMessage:
            if role == "system":
                return SystemMessage(content=f"<s> [INST]<<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "assistant":
                return AIMessage(content=f"{content} </s><s> [INST] ")
            elif role in {"human", "user"}:
                return HumanMessage(content=f"{content} [/INST] ")
            else:
                raise ValueError(f"Unsupported role: {role}")

        formatted_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                formatted_messages.append(format_message(message.content, "system"))
            elif isinstance(message, AIMessage):
                formatted_messages.append(format_message(message.content, "assistant"))
            elif isinstance(message, HumanMessage):
                formatted_messages.append(format_message(message.content, "human"))
            elif isinstance(message, dict):
                role = message.get("role", "").lower()
                content = message.get("content", "")
                formatted_messages.append(format_message(content, role))
            else:
                raise ValueError(f"Unsupported message format: {type(message)}")

        return formatted_messages

    def invoke(self, messages: List[Union[Dict[str, str], BaseMessage]]) -> AIMessage:
        formatted_messages = self._format_messages(messages)
        return self.llm.invoke(formatted_messages)

    async def ainvoke(self, messages: List[Union[Dict[str, str], BaseMessage]]) -> AIMessage:
        formatted_messages = self._format_messages(messages)
        return await self.llm.ainvoke(formatted_messages)

    def stream(self, messages: List[Union[Dict[str, str], BaseMessage]]) -> AIMessageChunk:
        formatted_messages = self._format_messages(messages)
        for chunk in self.llm.stream(formatted_messages):
            yield chunk

    async def astream(self, messages: List[Union[Dict[str, str], BaseMessage]]) -> AIMessageChunk:
        formatted_messages = self._format_messages(messages)
        async for chunk in self.llm.astream(formatted_messages):
            yield chunk


if __name__ == "__main__":
    custom_params = {"temperature": 0.3, "max_new_tokens": 100}
    allam = ALLaM(parameters=custom_params)
