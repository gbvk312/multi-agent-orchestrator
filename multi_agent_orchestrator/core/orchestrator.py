import asyncio
import logging
import uuid

from google import genai
from google.genai import types

from .agent import AgentError, BaseAgent
from .config import OrchestratorConfig
from .memory import MemoryManager

logger = logging.getLogger(__name__)


class OrchestratorError(Exception):
    """Raised when the orchestrator encounters an unrecoverable error."""


class Orchestrator:
    """Central router that delegates tasks to specialized agents.

    The orchestrator uses an LLM-based router to analyze user intent
    and select the most appropriate agent. All processing is async.

    Accepts an optional ``OrchestratorConfig``; when omitted the config
    is built from environment variables via ``OrchestratorConfig.from_env()``.
    """

    def __init__(
        self,
        memory_manager: MemoryManager | None = None,
        model: str | None = None,
        api_key: str | None = None,
        config: OrchestratorConfig | None = None,
    ):
        self._config = config or OrchestratorConfig.from_env()

        self.agents: dict[str, BaseAgent] = {}
        self.memory = memory_manager or MemoryManager(max_history=self._config.max_history)
        self.model = model or self._config.default_model

        # Validate API key early
        self._api_key = api_key or self._config.gemini_api_key
        if not self._api_key:
            raise OrchestratorError("GEMINI_API_KEY environment variable or api_key parameter is required")
        self.client = genai.Client(api_key=self._api_key)

    def register_agent(self, agent: BaseAgent) -> None:
        """Registers an agent with the orchestrator."""
        if agent.name in self.agents:
            logger.warning("Overwriting existing agent: %s", agent.name)
        self.agents[agent.name] = agent

    def unregister_agent(self, name: str) -> bool:
        """Removes a registered agent. Returns True if the agent existed."""
        return self.agents.pop(name, None) is not None

    async def _route_request(self, query: str) -> str:
        """Determines which agent should handle the request."""
        if not self.agents:
            raise OrchestratorError("No agents registered with the orchestrator.")

        agent_names = list(self.agents.keys())

        # If only one agent, skip routing
        if len(agent_names) == 1:
            return agent_names[0]

        agent_descriptions = "\n".join(
            [
                f"- {name}: {agent.system_prompt[:100]}{'...' if len(agent.system_prompt) > 100 else ''}"
                for name, agent in self.agents.items()
            ]
        )

        routing_prompt = f"""
        You are a routing supervisor. Based on the user's query, \
you must decide which agent is best suited to handle the request.

        Available agents:
        {agent_descriptions}

        User Query: "{query}"

        Respond ONLY with the exact name of the selected agent. If no agent is suitable, respond with "Default".
        """

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=routing_prompt,
                config=types.GenerateContentConfig(temperature=self._config.routing_temperature),
            )

            if response.text:
                selected_agent = response.text.strip()
                if selected_agent in self.agents:
                    return selected_agent
        except Exception as e:
            logger.warning("Routing LLM call failed (%s), falling back to first agent.", e)

        # Fallback to the first registered agent
        return agent_names[0]

    async def process_request(self, session_id: str, query: str) -> str:
        """Processes a user request by routing it to the appropriate agent."""
        trace_id = uuid.uuid4().hex[:12]
        logger.info("[%s] Processing request for session=%s", trace_id, session_id)

        # 1. Determine which agent should handle this
        target_agent_name = await self._route_request(query)
        target_agent = self.agents[target_agent_name]

        # 2. Get session history BEFORE adding current query
        history = await self.memory.get_history(session_id)

        # 3. Delegate to the agent
        logger.info("[%s] Routing to -> %s", trace_id, target_agent_name)

        try:
            response_text = await target_agent.process(query, history)
        except AgentError as e:
            response_text = f"Error from {target_agent_name}: {e}"
            logger.error("[%s] %s", trace_id, response_text)

        # 4. Save BOTH messages to memory AFTER processing
        await self.memory.add_message(session_id, "user", query)
        await self.memory.add_message(session_id, "model", response_text)

        return response_text

    async def fan_out(
        self,
        session_id: str,
        query: str,
        agent_names: list[str] | None = None,
    ) -> dict[str, str]:
        """Execute multiple agents in parallel and return all results.

        Args:
            session_id: Session identifier for context retrieval.
            query: The user query to process.
            agent_names: Optional list of agent names to run. Defaults to all.

        Returns:
            Dict mapping agent names to their responses.
        """
        trace_id = uuid.uuid4().hex[:12]
        targets = agent_names or list(self.agents.keys())

        # Validate all target agents exist
        unknown = [n for n in targets if n not in self.agents]
        if unknown:
            raise OrchestratorError(f"Unknown agents: {unknown}")

        history = await self.memory.get_history(session_id)
        logger.info("[%s] Fan-out to %d agents: %s", trace_id, len(targets), targets)

        async def _run_agent(name: str) -> tuple[str, str]:
            try:
                result = await self.agents[name].process(query, history)
                return name, result
            except AgentError as e:
                logger.error("[%s] Agent %s failed: %s", trace_id, name, e)
                return name, f"Error from {name}: {e}"

        results_list = await asyncio.gather(*[_run_agent(n) for n in targets])
        results = dict(results_list)

        # Persist the query and a combined summary to memory
        await self.memory.add_message(session_id, "user", query)
        summary_parts = [f"[{name}]: {resp}" for name, resp in results.items()]
        await self.memory.add_message(session_id, "model", "\n\n".join(summary_parts))

        return results

    def __repr__(self) -> str:
        return f"Orchestrator(model={self.model!r}, agents={list(self.agents.keys())})"
