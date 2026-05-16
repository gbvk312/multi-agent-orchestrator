import asyncio
import inspect
import logging
import os
from typing import List, Callable, Optional, Any, Dict
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Maximum number of tool-call round-trips before forcing a text response
_MAX_TOOL_ROUNDS = 5


class AgentError(Exception):
    """Raised when the agent encounters an unrecoverable error."""


class BaseAgent:
    """Base class for an intelligent agent powered by Gemini.

    Supports automatic tool execution: when the model requests a function
    call, the agent executes the matching tool and feeds the result back,
    repeating until the model produces a final text response.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "gemini-2.5-flash",
        tools: Optional[List[Callable]] = None,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools or []
        self.max_retries = max_retries
        self.timeout = timeout

        # Build a lookup map for tool execution
        self._tool_map: Dict[str, Callable] = {fn.__name__: fn for fn in self.tools}

        # Validate API key early — fail fast instead of at first API call
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise AgentError("GEMINI_API_KEY environment variable or api_key parameter is required")

        # Initialize Gemini Client
        self.client = genai.Client(api_key=self._api_key)

    async def process(self, query: str, history: List[Dict[str, Any]]) -> str:
        """Processes a query with the given context history.

        This is an async method that supports automatic tool execution.
        If the model returns a function call, the agent will execute the
        tool, feed the result back, and continue until a text response
        is generated (up to _MAX_TOOL_ROUNDS).

        Raises AgentError if the processing exceeds the configured timeout.
        """
        try:
            async with asyncio.timeout(self.timeout):
                return await self._process_inner(query, history)
        except TimeoutError:
            raise AgentError(f"[{self.name}] Processing timed out after {self.timeout}s")

    async def _process_inner(self, query: str, history: List[Dict[str, Any]]) -> str:
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
            temperature=0.2,
        )

        for round_num in range(_MAX_TOOL_ROUNDS):
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
            for call, result in zip(response.function_calls, tool_results):
                fn_response_parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=fn_response_parts))

        # If we exhausted rounds, return what we have
        return f"[{self.name}] Reached maximum tool execution rounds ({_MAX_TOOL_ROUNDS}). Last response may be incomplete."

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
                    logger.warning(f"[{self.name}] Attempt {attempt + 1} failed ({e}), retrying in {wait_time}s...")
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
            logger.error(f"[{self.name}] Tool '{call.name}' failed: {e}")
            return f"Error executing tool '{call.name}': {e}"

    def __repr__(self) -> str:
        return f"BaseAgent(name={self.name!r}, model={self.model!r}, tools={len(self.tools)})"
