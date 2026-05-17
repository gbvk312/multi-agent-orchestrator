import asyncio
import concurrent.futures
import functools
import inspect
import logging
import random
from collections.abc import AsyncGenerator, Callable
from typing import Any

from google import genai
from google.genai import types

from .config import OrchestratorConfig
from .events import AgentFinishEvent, AgentStartEvent, EventHandler, ToolCallEvent, ToolResultEvent

logger = logging.getLogger(__name__)


def _is_async_callable(obj: Any) -> bool:
    """Robustly checks if a callable is an asynchronous function or object."""
    if inspect.iscoroutinefunction(obj):
        return True
    if callable(obj):
        try:
            if inspect.iscoroutinefunction(obj.__call__):
                return True
        except AttributeError:
            pass
    if hasattr(obj, "__wrapped__"):
        return _is_async_callable(obj.__wrapped__)
    if isinstance(obj, functools.partial):
        return _is_async_callable(obj.func)
    return False


async def _resolve_prompt(
    prompt: str | Callable[..., Any],
    query: str,
) -> str:
    """Resolve a system prompt that may be a string, sync callable, or async callable."""
    if not callable(prompt):
        return prompt
    if _is_async_callable(prompt):
        try:
            return str(await prompt(query))
        except TypeError:
            return str(await prompt())
    try:
        return str(prompt(query))
    except TypeError:
        return str(prompt())


class AgentError(Exception):
    """Raised when the agent encounters an unrecoverable error."""


class AgentHandoff(Exception):
    """Raised by a tool to transfer control to another agent."""

    def __init__(self, target_agent: str, message: str = ""):
        self.target_agent = target_agent
        self.message = message
        super().__init__(f"Handoff to {target_agent}: {message}")


class HumanApprovalRequired(Exception):
    """Raised by a tool to pause execution for human approval."""

    def __init__(self, tool_name: str, tool_args: dict[str, Any], message: str = ""):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.message = message
        super().__init__(f"Human approval required for tool '{tool_name}': {message}")


class BaseAgent:
    """Base class for an intelligent agent powered by Gemini.

    Supports automatic tool execution: when the model requests a function
    call, the agent executes the matching tool and feeds the result back,
    repeating until the model produces a final text response.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str | Callable[..., Any],
        model: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        max_retries: int | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        max_tool_rounds: int | None = None,
        response_schema: Any | None = None,
        config: OrchestratorConfig | None = None,
        executor: concurrent.futures.Executor | None = None,
    ):
        self._config = config or OrchestratorConfig.from_env()

        self.name = name
        self.system_prompt = system_prompt
        self.model = model or self._config.default_model
        self.tools = tools or []
        self.max_retries = max_retries if max_retries is not None else self._config.max_retries
        self.timeout = timeout if timeout is not None else self._config.agent_timeout
        self.temperature = temperature if temperature is not None else self._config.temperature
        self.max_tool_rounds = max_tool_rounds if max_tool_rounds is not None else self._config.max_tool_rounds
        if response_schema is not None and not isinstance(response_schema, (dict, type)):
            raise AgentError(f"response_schema must be a dict or type, got {type(response_schema).__name__}")
        self.response_schema = response_schema
        self.executor = executor

        self._tool_map: dict[str, Callable[..., Any]] = {fn.__name__: fn for fn in self.tools}

        self._api_key = api_key or self._config.gemini_api_key
        if not self._api_key:
            raise AgentError("GEMINI_API_KEY environment variable or api_key parameter is required")

        self.client = genai.Client(api_key=self._api_key)

    async def pre_process(self, query: str, history: list[dict[str, Any]]) -> str:
        """Hook to transform the query before model invocation. Override in subclasses."""
        return query

    async def post_process(self, response: str) -> str:
        """Hook to transform the final response before returning. Override in subclasses."""
        return response

    async def process(
        self,
        query: str,
        history: list[dict[str, Any]],
        session_id: str = "",
        event_handler: EventHandler | None = None,
    ) -> str:
        try:
            async with asyncio.timeout(self.timeout):
                return await self._process_inner(query, history, session_id, event_handler)
        except TimeoutError as exc:
            raise AgentError(f"[{self.name}] Processing timed out after {self.timeout}s") from exc

    async def _await_with_timeout(self, awaitable: Any) -> Any:
        """Apply per-operation timeout to upstream await points."""
        async with asyncio.timeout(self.timeout):
            return await awaitable

    async def _process_inner(
        self,
        query: str,
        history: list[dict[str, Any]],
        session_id: str,
        event_handler: EventHandler | None,
    ) -> str:
        query = await self.pre_process(query, history)

        if event_handler:
            await event_handler.on_event(AgentStartEvent(session_id, self.name, query))

        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=query)]))

        system_instruction = await _resolve_prompt(self.system_prompt, query)

        kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "tools": self.tools if self.tools else None,
            "temperature": self.temperature,
        }
        if self.response_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = self.response_schema
        config = types.GenerateContentConfig(**kwargs)

        for _round_num in range(self.max_tool_rounds):
            response = await self._call_model_with_retry(contents, config)

            if not response.candidates:
                return f"[{self.name}] Generation failed or blocked by safety settings."

            if not response.function_calls:
                text_response = await self.post_process(response.text or "")
                if event_handler:
                    await event_handler.on_event(AgentFinishEvent(session_id, self.name, text_response))
                return text_response

            tool_results = []
            for call in response.function_calls:
                if event_handler and call.name:
                    args_dict = dict(call.args) if call.args else {}
                    event_call = ToolCallEvent(session_id, self.name, call.name, args_dict)
                    await self._await_with_timeout(event_handler.on_event(event_call))

                result = await self._await_with_timeout(self._execute_tool(call))

                if event_handler and call.name:
                    event_result = ToolResultEvent(session_id, self.name, call.name, result)
                    await self._await_with_timeout(event_handler.on_event(event_result))

                tool_results.append(result)

            if response.candidates and response.candidates[0].content:
                contents.append(response.candidates[0].content)

            fn_response_parts = []
            for call, result in zip(response.function_calls, tool_results, strict=True):
                if call.name:
                    fn_response_parts.append(
                        types.Part.from_function_response(name=call.name, response={"result": result})
                    )
            contents.append(types.Content(role="user", parts=fn_response_parts))

        return await self.post_process(f"[{self.name}] Reached maximum tool execution rounds ({self.max_tool_rounds}).")

    async def process_stream(
        self,
        query: str,
        history: list[dict[str, Any]],
        session_id: str = "",
        event_handler: EventHandler | None = None,
    ) -> AsyncGenerator[str, None]:
        try:
            query = await self.pre_process(query, history)

            if event_handler:
                event_start = AgentStartEvent(session_id, self.name, query)
                await self._await_with_timeout(event_handler.on_event(event_start))

            contents = []
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=query)]))

            system_instruction = await _resolve_prompt(self.system_prompt, query)

            kwargs: dict[str, Any] = {
                "system_instruction": system_instruction,
                "tools": self.tools if self.tools else None,
                "temperature": self.temperature,
            }
            if self.response_schema is not None:
                kwargs["response_mime_type"] = "application/json"
                kwargs["response_schema"] = self.response_schema
            config = types.GenerateContentConfig(**kwargs)

            for _round_num in range(self.max_tool_rounds):
                response_stream = await self._await_with_timeout(
                    self.client.aio.models.generate_content_stream(
                        model=self.model,
                        contents=contents,
                        config=config,
                    )
                )

                collected_text = ""
                function_calls = []
                candidates_content = None

                stream_iter = response_stream.__aiter__()
                while True:
                    try:
                        chunk = await self._await_with_timeout(stream_iter.__anext__())
                    except StopAsyncIteration:
                        break

                    if chunk.text:
                        yield chunk.text
                        collected_text += chunk.text
                    if chunk.function_calls:
                        function_calls.extend(chunk.function_calls)

                    if chunk.candidates and chunk.candidates[0].content:
                        if not candidates_content:
                            candidates_content = chunk.candidates[0].content
                        else:
                            if candidates_content.parts is None:
                                candidates_content.parts = []
                            if chunk.candidates[0].content.parts:
                                candidates_content.parts.extend(chunk.candidates[0].content.parts)

                if not function_calls:
                    collected_text = await self.post_process(collected_text)
                    if event_handler:
                        event_finish = AgentFinishEvent(session_id, self.name, collected_text)
                        await self._await_with_timeout(event_handler.on_event(event_finish))
                    return

                tool_results = []
                for call in function_calls:
                    if event_handler and call.name:
                        args_dict = dict(call.args) if call.args else {}
                        event_call = ToolCallEvent(session_id, self.name, call.name, args_dict)
                        await self._await_with_timeout(event_handler.on_event(event_call))

                    result = await self._await_with_timeout(self._execute_tool(call))

                    if event_handler and call.name:
                        event_result = ToolResultEvent(session_id, self.name, call.name, result)
                        await self._await_with_timeout(event_handler.on_event(event_result))
                    tool_results.append(result)

                if candidates_content:
                    contents.append(candidates_content)

                fn_response_parts = []
                for call, result in zip(function_calls, tool_results, strict=True):
                    if call.name:
                        fn_response_parts.append(
                            types.Part.from_function_response(name=call.name, response={"result": result})
                        )
                contents.append(types.Content(role="user", parts=fn_response_parts))

            max_msg = await self.post_process(
                f"[{self.name}] Reached maximum tool execution rounds ({self.max_tool_rounds})."
            )
            yield max_msg
        except TimeoutError as exc:
            raise AgentError(f"[{self.name}] Streaming timed out after {self.timeout}s") from exc

    async def _call_model_with_retry(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "429" in error_str or "resource exhausted" in error_str or "500" in error_str:
                    wait_time = (2**attempt) + random.uniform(0.1, 1.0)
                    logger.warning("[%s] Attempt %d failed, retrying in %.2fs...", self.name, attempt + 1, wait_time)
                    await asyncio.sleep(wait_time)
                else:
                    raise AgentError(f"[{self.name}] Model call failed: {e}") from e

        raise AgentError(f"[{self.name}] All {self.max_retries} retries exhausted.") from last_error

    async def _execute_tool(self, call: types.FunctionCall) -> str:
        if not call.name:
            return "Error: Unknown tool call without a name."

        tool_fn = self._tool_map.get(call.name)
        if not tool_fn:
            return f"Error: Unknown tool '{call.name}'. Available tools: {list(self._tool_map.keys())}"

        try:
            args = dict(call.args) if call.args else {}
            if _is_async_callable(tool_fn):
                result = await tool_fn(**args)
            else:
                if self.executor is not None:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(self.executor, functools.partial(tool_fn, **args))
                else:
                    result = await asyncio.to_thread(tool_fn, **args)
            return str(result)
        except AgentHandoff:
            raise
        except HumanApprovalRequired:
            raise
        except Exception as e:
            logger.error("[%s] Tool '%s' failed: %s", self.name, call.name, e)
            return f"Error executing tool '{call.name}': {e}"

    def __repr__(self) -> str:
        return f"BaseAgent(name={self.name!r}, model={self.model!r}, tools={len(self.tools)})"
