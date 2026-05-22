import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from google import genai
from google.genai import types

from .agent import AgentError, AgentHandoff, BaseAgent, HumanApprovalRequired, _resolve_prompt
from .config import OrchestratorConfig
from .events import (
    EventHandler,
    OrchestratorErrorEvent,
    OrchestratorFinishEvent,
    OrchestratorHandoffEvent,
    OrchestratorRouteEvent,
    OrchestratorStartEvent,
)
from .memory import MemoryManager

logger = logging.getLogger(__name__)


class OrchestratorError(Exception):
    """Raised when the orchestrator encounters an unrecoverable error."""

    def __init__(self, message: str = ""):
        super().__init__(message)


class Orchestrator:
    """Central router that delegates tasks to specialized agents."""

    def __init__(
        self,
        memory_manager: MemoryManager | None = None,
        model: str | None = None,
        api_key: str | None = None,
        config: OrchestratorConfig | None = None,
        event_handler: EventHandler | None = None,
        default_fallback_agent: str | None = None,
        routing_handler: Callable[[str, dict[str, BaseAgent]], Awaitable[str]] | None = None,
        routing_system_instruction: str | None = None,
    ):
        self._config = config or OrchestratorConfig.from_env()

        self.agents: dict[str, BaseAgent] = {}
        self.memory = memory_manager or MemoryManager(max_history=self._config.max_history)
        self.model = model or self._config.default_model
        self.event_handler = event_handler
        self.default_fallback_agent = default_fallback_agent
        self.routing_handler = routing_handler
        self.routing_system_instruction = routing_system_instruction or self._config.routing_system_instruction

        self._api_key = api_key or self._config.gemini_api_key
        if not self._api_key:
            raise OrchestratorError("GEMINI_API_KEY environment variable or api_key parameter is required")
        self.client = genai.Client(api_key=self._api_key)

    def register_agent(self, agent: BaseAgent) -> None:
        if agent.name in self.agents:
            logger.warning("Overwriting existing agent: %s", agent.name)
        self.agents[agent.name] = agent

    def unregister_agent(self, name: str) -> bool:
        return self.agents.pop(name, None) is not None

    async def _route_request(self, query: str, session_id: str = "", context: dict[str, Any] | None = None) -> str:
        if not self.agents:
            raise OrchestratorError("No agents registered with the orchestrator.")
        agent_names = list(self.agents.keys())
        if len(agent_names) == 1:
            return agent_names[0]

        if self.routing_handler is not None:
            try:
                selected_agent = await self.routing_handler(query, self.agents)
                if selected_agent in self.agents:
                    return selected_agent
                logger.warning("Custom routing handler returned unregistered agent: %s", selected_agent)
            except Exception as e:
                logger.warning("Custom routing handler failed: %s", e)

        agent_desc_list = []
        for name, agent in self.agents.items():
            prompt = agent.system_prompt
            try:
                resolved = await _resolve_prompt(prompt, query, session_id=session_id, context=context)
            except Exception:
                resolved = str(prompt)
            agent_desc_list.append(f"- {name}: {resolved[:100]}...")
        agent_descriptions = "\n".join(agent_desc_list)
        routing_system_instruction = self.routing_system_instruction or (
            "You are a routing supervisor. Based on the user's query, "
            "you must decide which agent is best suited to handle the request."
        )
        routing_prompt = (
            f'{routing_system_instruction}\n\nAvailable agents:\n{agent_descriptions}\n\nUser Query: "{query}"'
        )
        schema = types.Schema(type=types.Type.STRING, enum=agent_names)
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=routing_prompt,
                config=types.GenerateContentConfig(
                    temperature=self._config.routing_temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            if response.text:
                try:
                    raw = json.loads(response.text)
                    selected_agent = raw if isinstance(raw, str) else str(raw).strip('"')
                except json.JSONDecodeError:
                    selected_agent = str(response.text.strip().strip('"'))
                if selected_agent in self.agents:
                    return selected_agent
        except Exception as e:
            logger.warning("Routing LLM call failed (%s)", e)

        fallback = self.default_fallback_agent or agent_names[0]
        if fallback not in self.agents:
            fallback = agent_names[0]
        logger.warning("Falling back to agent: %s", fallback)
        return fallback

    async def pre_process(self, query: str) -> str:
        """Hook to transform the user query globally before routing. Override in subclasses."""
        return query

    async def post_process(self, response: str) -> str:
        """Hook to transform the final orchestrator response globally. Override in subclasses."""
        return response

    async def process_request(self, session_id: str, query: str, context: dict[str, Any] | None = None) -> str:
        """Processes a user request by routing it to the appropriate agent."""
        trace_id = uuid.uuid4().hex[:12]
        logger.info("[%s] Processing request for session=%s", trace_id, session_id)

        query = await self.pre_process(query)

        if self.event_handler:
            await self.event_handler.on_event(OrchestratorStartEvent(session_id, query))

        try:
            target_agent_name = await self._route_request(query, session_id=session_id, context=context)
            if self.event_handler:
                await self.event_handler.on_event(OrchestratorRouteEvent(session_id, query, target_agent_name))

            current_query = query
            history = await self.memory.get_history(session_id)

            final_response_text = ""

            # Loop to handle agent handoffs (bounded to prevent infinite cycles)
            handoff_count = 0
            while True:
                handoff_count += 1
                if handoff_count > self._config.max_handoffs:
                    final_response_text += f"Error: Exceeded max handoff limit ({self._config.max_handoffs})."
                    logger.error("[%s] Handoff loop detected after %d handoffs", trace_id, self._config.max_handoffs)
                    break

                target_agent = self.agents.get(target_agent_name)
                if not target_agent:
                    final_response_text = f"Error: Agent '{target_agent_name}' not found."
                    break

                logger.info("[%s] Routing to -> %s", trace_id, target_agent_name)

                try:
                    response_text = await target_agent.process(
                        current_query,
                        history,
                        session_id=session_id,
                        event_handler=self.event_handler,
                        memory=self.memory,
                        context=context,
                    )
                    final_response_text += response_text
                    break  # Done processing

                except AgentHandoff as handoff:
                    logger.info("[%s] Agent %s handing off to %s", trace_id, target_agent_name, handoff.target_agent)
                    if self.event_handler:
                        await self.event_handler.on_event(
                            OrchestratorHandoffEvent(
                                session_id,
                                source_agent=target_agent_name,
                                target_agent=handoff.target_agent,
                                message=handoff.message,
                            )
                        )
                    if handoff.message:
                        history.append(
                            {"role": "model", "content": f"Handing off to {handoff.target_agent}: {handoff.message}"}
                        )
                        current_query = handoff.message
                    target_agent_name = handoff.target_agent

                except HumanApprovalRequired as approval:
                    final_response_text = (
                        f"Execution paused. Human approval required for tool '{approval.tool_name}' "
                        f"with args {approval.tool_args}. Message: {approval.message}"
                    )
                    break

                except AgentError as e:
                    if self._config.propagate_errors:
                        raise
                    final_response_text = f"Error from {target_agent_name}: {e}"
                    logger.error("[%s] %s", trace_id, final_response_text)
                    break

            final_response_text = await self.post_process(final_response_text)

            await self.memory.add_message(session_id, "user", query)
            await self.memory.add_message(session_id, "model", final_response_text)

            if self.event_handler:
                await self.event_handler.on_event(OrchestratorFinishEvent(session_id, final_response_text))

            return final_response_text

        except Exception as e:
            if self.event_handler:
                await self.event_handler.on_event(OrchestratorErrorEvent(session_id, e))
            raise

    async def process_request_stream(
        self, session_id: str, query: str, context: dict[str, Any] | None = None
    ) -> AsyncGenerator[str, None]:
        """Processes a user request and streams the response.

        Note:
            Any post-processing via ``post_process()`` is applied *only* to the
            accumulated response stored in the session's memory at the end of the stream.
            The yielded chunks are streamed directly from the agent and are raw/unprocessed.
        """
        trace_id = uuid.uuid4().hex[:12]
        logger.info("[%s] Streaming request for session=%s", trace_id, session_id)

        query = await self.pre_process(query)

        if self.event_handler:
            await self.event_handler.on_event(OrchestratorStartEvent(session_id, query))

        try:
            target_agent_name = await self._route_request(query, session_id=session_id, context=context)
            if self.event_handler:
                await self.event_handler.on_event(OrchestratorRouteEvent(session_id, query, target_agent_name))

            current_query = query
            history = await self.memory.get_history(session_id)

            final_response_text = ""

            handoff_count = 0
            while True:
                handoff_count += 1
                if handoff_count > self._config.max_handoffs:
                    err_msg = f"Error: Exceeded max handoff limit ({self._config.max_handoffs})."
                    yield err_msg
                    final_response_text += err_msg
                    logger.error("[%s] Handoff loop detected after %d handoffs", trace_id, self._config.max_handoffs)
                    break

                target_agent = self.agents.get(target_agent_name)
                if not target_agent:
                    err_msg = f"Error: Agent '{target_agent_name}' not found."
                    yield err_msg
                    final_response_text += err_msg
                    break

                logger.info("[%s] Routing to -> %s", trace_id, target_agent_name)

                try:
                    async for chunk in target_agent.process_stream(
                        current_query,
                        history,
                        session_id=session_id,
                        event_handler=self.event_handler,
                        memory=self.memory,
                        context=context,
                    ):
                        yield chunk
                        final_response_text += chunk
                    break

                except AgentHandoff as handoff:
                    logger.info("[%s] Agent %s handing off to %s", trace_id, target_agent_name, handoff.target_agent)
                    if self.event_handler:
                        await self.event_handler.on_event(
                            OrchestratorHandoffEvent(
                                session_id,
                                source_agent=target_agent_name,
                                target_agent=handoff.target_agent,
                                message=handoff.message,
                            )
                        )
                    if handoff.message:
                        history.append(
                            {"role": "model", "content": f"Handing off to {handoff.target_agent}: {handoff.message}"}
                        )
                        current_query = handoff.message
                    target_agent_name = handoff.target_agent

                except HumanApprovalRequired as approval:
                    msg = (
                        f"\nExecution paused. Human approval required for tool "
                        f"'{approval.tool_name}'. Message: {approval.message}"
                    )
                    yield msg
                    final_response_text += msg
                    break

                except AgentError as e:
                    if self._config.propagate_errors:
                        raise
                    msg = f"\nError from {target_agent_name}: {e}"
                    yield msg
                    final_response_text += msg
                    logger.error("[%s] %s", trace_id, msg)
                    break

            final_response_text = await self.post_process(final_response_text)

            await self.memory.add_message(session_id, "user", query)
            await self.memory.add_message(session_id, "model", final_response_text)

            if self.event_handler:
                await self.event_handler.on_event(OrchestratorFinishEvent(session_id, final_response_text))

        except Exception as e:
            if self.event_handler:
                await self.event_handler.on_event(OrchestratorErrorEvent(session_id, e))
            raise

    async def close(self) -> None:
        """Cleanly close the memory backend."""
        await self.memory.close()

    async def __aenter__(self) -> "Orchestrator":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def chain(
        self,
        session_id: str,
        query: str,
        sequence: list[str],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Execute agents sequentially in a pipeline."""
        try:
            if not sequence:
                raise OrchestratorError("Sequence of agents cannot be empty.")

            query = await self.pre_process(query)

            if self.event_handler:
                await self.event_handler.on_event(OrchestratorStartEvent(session_id, query))

            current_query = query
            final_response = ""
            for agent_name in sequence:
                if agent_name not in self.agents:
                    raise OrchestratorError(f"Unknown agent in sequence: {agent_name}")

                if self.event_handler:
                    await self.event_handler.on_event(OrchestratorRouteEvent(session_id, current_query, agent_name))

                agent = self.agents[agent_name]
                history = await self.memory.get_history(session_id)

                try:
                    response = await agent.process(
                        current_query,
                        history,
                        session_id=session_id,
                        event_handler=self.event_handler,
                        memory=self.memory,
                        context=context,
                    )

                    # Append to history so next agent sees it
                    await self.memory.add_message(session_id, "user", current_query)
                    await self.memory.add_message(session_id, "model", f"[{agent_name} output]: {response}")

                    # Output of current agent becomes input for next
                    current_query = (
                        f"Here is the output from the previous step ({agent_name}). "
                        f"Please continue the workflow: {response}"
                    )
                    final_response = response

                except Exception as e:
                    logger.error("Error in chain at agent %s: %s", agent_name, e)
                    if self.event_handler:
                        await self.event_handler.on_event(OrchestratorErrorEvent(session_id, e))
                    if self._config.propagate_errors:
                        raise
                    return f"Chain failed at {agent_name}: {e}"

            final_response = await self.post_process(final_response)

            if self.event_handler:
                await self.event_handler.on_event(OrchestratorFinishEvent(session_id, final_response))

            return final_response

        except Exception as e:
            if self.event_handler:
                await self.event_handler.on_event(OrchestratorErrorEvent(session_id, e))
            raise

    async def fan_out(
        self,
        session_id: str,
        query: str,
        agent_names: list[str] | None = None,
        max_concurrency: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Execute multiple agents in parallel and return all results."""
        trace_id = uuid.uuid4().hex[:12]
        targets = agent_names or list(self.agents.keys())
        unknown = [n for n in targets if n not in self.agents]
        if unknown:
            raise OrchestratorError(f"Unknown agents: {unknown}")

        query = await self.pre_process(query)

        if self.event_handler:
            await self.event_handler.on_event(OrchestratorStartEvent(session_id, query))

        history = await self.memory.get_history(session_id)
        logger.info("[%s] Fan-out to %d agents: %s", trace_id, len(targets), targets)

        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None

        async def _run_agent(name: str) -> tuple[str, str]:
            if semaphore:
                async with semaphore:
                    return await _run_agent_inner(name)
            return await _run_agent_inner(name)

        async def _run_agent_inner(name: str) -> tuple[str, str]:
            try:
                result = await self.agents[name].process(
                    query,
                    list(history),
                    session_id=session_id,
                    event_handler=self.event_handler,
                    memory=self.memory,
                    context=context,
                )
                return name, result
            except Exception as e:
                logger.error("[%s] Agent %s failed: %s", trace_id, name, e)
                return name, f"Error from {name}: {e}"

        results_list = await asyncio.gather(*[_run_agent(n) for n in targets])
        results = dict(results_list)

        await self.memory.add_message(session_id, "user", query)
        summary_parts = [f"[{name}]: {resp}" for name, resp in results.items()]
        combined_response = "\n\n".join(summary_parts)
        combined_response = await self.post_process(combined_response)
        await self.memory.add_message(session_id, "model", combined_response)

        if self.event_handler:
            await self.event_handler.on_event(OrchestratorFinishEvent(session_id, combined_response))

        return results

    def __repr__(self) -> str:
        return f"Orchestrator(model={self.model!r}, agents={list(self.agents.keys())})"
