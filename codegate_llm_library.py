import json
import logging
import requests
from typing import Any, Dict, List, Mapping, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pydantic import Field

# LangChain imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure session with retry logic
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST", "GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)


def _parse_sse_line(line: bytes) -> Optional[Dict[str, Any]]:
    if line.startswith(b'data:'):
        data_bytes = line[len(b'data:'):].strip()
        if data_bytes == b'[DONE]':
            return {"done": True}
        if data_bytes:
            try:
                data_str = data_bytes.decode('utf-8')
                return json.loads(data_str)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Could not parse SSE line: {e}, Raw: {data_bytes!r}")
    return None


class CodegateLLM(LLM):
    codegate_url: str = Field(...)
    workspace_name: str = Field(...)
    model_name: str = Field(...)
    provider: str = Field(...)
    temperature: float = Field(default=0.7)
    api_key: Optional[str] = Field(default=None)
    application_name: Optional[str] = Field(default=None)
    timeout: int = Field(default=60)
    @property
    def _llm_type(self) -> str:
        return "codegate"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        headers = {
            "X-Workspace": self.workspace_name,
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.application_name:
            headers["X-Application-Name"] = self.application_name

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": kwargs.get("model", self.model_name),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        full_response = ""
        target_url = f"{self.codegate_url}/v1/mux/chat/completions"

        try:
            response = session.post(target_url, headers=headers, json=payload, stream=True, timeout=self.timeout)
            response.raise_for_status()

            for line_bytes in response.iter_lines():
                if line_bytes:
                    parsed = _parse_sse_line(line_bytes)
                    if not parsed:
                        continue
                    if parsed.get("done"):
                        break
                    if "choices" in parsed and parsed["choices"]:
                        delta = parsed["choices"][0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            full_response += token
                            if run_manager:
                                run_manager.on_llm_new_token(token)
            return full_response

        except Exception as e:
            logger.error(f"Error during _call to Codegate LLM: {e}", exc_info=True)
            if run_manager:
                run_manager.on_llm_error(e)
            raise

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "workspace_name": self.workspace_name,
            "model_name": self.model_name,
            "provider": self.provider,
            "temperature": self.temperature,
        }


class CodegateChatModel(BaseChatModel):
    codegate_url: str = Field(...)
    workspace_name: str = Field(...)
    model_name: str = Field(...)
    provider: str = Field(...)
    temperature: float = Field(default=0.7)
    api_key: Optional[str] = Field(default=None)
    application_name: Optional[str] = Field(default=None)
    timeout: int = Field(default=60)

    @property
    def _llm_type(self) -> str:
        return "codegate_chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        headers = {
            "X-Workspace": self.workspace_name,
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.application_name:
            headers["X-Application-Name"] = self.application_name

        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                role = getattr(msg, "type", "user")
            formatted.append({"role": role, "content": msg.content})

        payload = {
            "messages": formatted,
            "model": kwargs.get("model", self.model_name),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        full_response = ""
        target_url = f"{self.codegate_url}/v1/mux/chat/completions"

        try:
            response = session.post(target_url, headers=headers, json=payload, stream=True, timeout=self.timeout)
            response.raise_for_status()

            for line_bytes in response.iter_lines():
                if line_bytes:
                    parsed = _parse_sse_line(line_bytes)
                    if not parsed:
                        continue
                    if parsed.get("done"):
                        break
                    if "choices" in parsed and parsed["choices"]:
                        delta = parsed["choices"][0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            full_response += token
                            if run_manager:
                                run_manager.on_llm_new_token(token)

            message = AIMessage(content=full_response)
            generation = ChatGeneration(message=message, generation_info=None)
            return ChatResult(generations=[generation], llm_output={})

        except Exception as e:
            logger.error(f"Error during _generate to Codegate ChatModel: {e}", exc_info=True)
            if run_manager:
                run_manager.on_llm_error(e)
            message = AIMessage(content=f"Error: {e}")
            generation = ChatGeneration(message=message, generation_info={"finish_reason": "error"})
            return ChatResult(generations=[generation], llm_output={})

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "workspace_name": self.workspace_name,
            "model_name": self.model_name,
            "provider": self.provider,
            "temperature": self.temperature,
        }


class CodeGateWorkspace:
    def __init__(self, codegate_url: str, workspace_name: str):
        self.codegate_url = codegate_url
        self.workspace_name = workspace_name

    def setup(self, custom_instructions: str = "", muxing_rules: Optional[List[Dict[str, Any]]] = None,
              default_provider: str = "openai", default_model: str = "gpt-3.5-turbo", print_output: bool = True) -> bool:
        try:
            api_base = f"{self.codegate_url.rstrip('/')}/api/v1/workspaces"
            response = session.get(api_base)
            response.raise_for_status()
            workspaces = response.json().get("workspaces", [])
            workspace = next((ws for ws in workspaces if ws.get("name") == self.workspace_name), None)

            if workspace:
                if print_output:
                    logger.info(f"Workspace '{self.workspace_name}' already exists.")
                return True

            if not muxing_rules:
                muxing_rules = [{"pattern": "*", "provider_name": default_provider, "model": default_model}]

            workspace_data = {
                "name": self.workspace_name,
                "config": {"custom_instructions": custom_instructions, "muxing_rules": muxing_rules}
            }

            create_response = session.post(api_base, json=workspace_data)
            create_response.raise_for_status()

            if print_output:
                logger.info(f"Created workspace '{self.workspace_name}' successfully.")
            return True

        except Exception as e:
            logger.error(f"Error setting up workspace '{self.workspace_name}': {e}", exc_info=True)
            return False
