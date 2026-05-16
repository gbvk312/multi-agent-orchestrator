import asyncio
import inspect
import logging
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from .config import OrchestratorConfig

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Raised when the agent encounters an unrecoverable error."""


class BaseAgent:
    """Base class for an intelligent agent powered by Gemini.

    Supports automatic tool execution: when the model requests a function
    call, the agent executes the matching tool and feeds the result back,
    repeating until the model produces a final text response.

    All tuneable parameters fall back to ``OrchestratorConfig`` defaults
    when not supplied explicitly.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str | None = None,
        tools: list[Callable] | None = None,
        max_retries: int | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        max_tool_rounds: int | None = None,
        config: OrchestratorConfig | None = None,
    ):
        # Config provides defaults; explicit params take precedence
        self._config = config or OrchestratorConfig.from_env()

        self.name = name
        self.system_prompt = system_prompt
        self.model = model or self._config.default_model
        self.tools = tools or []
        self.max_retries = max_retries if max_retries is not None else self._config.max_retries
        self.timeout = timeout if timeout is not None else self._config.agent_timeout
        self.temperature = temperature if temperature is not None else self._config.temperature
        self.max_tool_rounds = max_tool_rounds if max_tool_rounds is not None else self._config.max_tool_rounds

        # Build a lookup map for tool execution
        self._tool_map: dict[str, Callable] = {fn.__name__: fn for fn in self.tools}

        # Validate API key early — fail fast instead of at first API call
        self._api_key = api_key or self._config.gemini_api_key
        if not self._api_key:
            raise AgentError("GEMINI_API_KEY environment variable or api_key parameter is required")

        # Initialize Gemini Client
        self.client = genai.Client(api_key=self._api_key)

    async def process(self, query: str, history: list[dict[str, Any]]) -> str:
        """Processes a query with the given context history.

        This is an async method that supports automatic tool execution.
        If the model returns a function call, the agent will execute the
        tool, feed the result back, and continue until a text response
        is generated (up to ``max_tool_rounds``).

        Raises AgentError if the processing exceeds the configured timeout.
        """
        try:
            async with asyncio.timeout(self.timeout):
                return await self._process_inner(query, history)
        except TimeoutError as exc:
            raise AgentError(f"[{self.name}] Processing timed out after {self.timeout}s") from exc

    async def _process_inner(self, query: str, history: list[dict[str, Any]]) -> str:
        """Inner processing logic, separated for timeout wrapping."""
        # Convert history to google-genai content format
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

        # Append the current query
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=query)]))

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=self.tools if self.tools else None,
            temperature=self.temperature,
        )

        for _round_num in range(self.max_tool_rounds):
            response = await self._call_model_with_retry(contents, config)

            # If the model returns a text response, we're done
            if not response.function_calls:
                return response.text or ""

            # Execute each function call and build results
            tool_results = []
            for call in response.function_calls:
                result = await self._execute_tool(call)
                tool_results.append(result)

            # Add the model's function call response and our results to the conversation
            contents.append(response.candidates[0].content)

            # Build function response parts
            fn_response_parts = []
            for call, result in zip(response.function_calls, tool_results, strict=True):
                fn_response_parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=fn_response_parts))

        # If we exhausted rounds, return what we have
        return (
            f"[{self.name}] Reached maximum tool execution rounds"
            f" ({self.max_tool_rounds}). Last response may be incomplete."
        )

    async def _call_model_with_retry(self, contents, config):
        """Calls the Gemini model with retry logic for transient errors."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                return response
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Retry on rate limits (429) and server errors (5xx)
                if "429" in error_str or "resource exhausted" in error_str or "500" in error_str:
                    wait_time = 2**attempt
                    logger.warning(
                        "[%s] Attempt %d failed (%s), retrying in %ds...",
                        self.name,
                        attempt + 1,
                        e,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise AgentError(f"[{self.name}] Model call failed: {e}") from e

        raise AgentError(
            f"[{self.name}] All {self.max_retries} retries exhausted. Last error: {last_error}"
        ) from last_error

    async def _execute_tool(self, call) -> str:
        """Executes a single tool call and returns the result as a string."""
        tool_fn = self._tool_map.get(call.name)
        if not tool_fn:
            return f"Error: Unknown tool '{call.name}'. Available tools: {list(self._tool_map.keys())}"

        try:
            args = dict(call.args) if call.args else {}
            # Support both sync and async tool functions
            if inspect.iscoroutinefunction(tool_fn):
                result = await tool_fn(**args)
            else:
                result = await asyncio.to_thread(tool_fn, **args)
            return str(result)
        except Exception as e:
            logger.error("[%s] Tool '%s' failed: %s", self.name, call.name, e)
            return f"Error executing tool '{call.name}': {e}"

    def __repr__(self) -> str:
        return f"BaseAgent(name={self.name!r}, model={self.model!r}, tools={len(self.tools)})"
