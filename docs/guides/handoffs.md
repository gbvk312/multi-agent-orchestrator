# Agent Handoffs & Human-in-the-Loop

This guide covers two powerful control-flow mechanisms: **Agent Handoffs** for transferring work between agents, and **Human-in-the-Loop (HITL)** for pausing execution to request human approval.

## Agent Handoffs

An agent can transfer control to another agent by raising `AgentHandoff` inside a tool function. The orchestrator catches this, updates the context, and routes the request to the target agent.

### Example

```python
from multi_agent_orchestrator import AgentHandoff, BaseAgent, Orchestrator

def escalate_to_human_agent():
    """Escalate complex issues to a human support agent."""
    raise AgentHandoff(
        target_agent="HumanSupport",
        message="Customer needs billing help beyond my capability."
    )

bot_agent = BaseAgent(
    name="ChatBot",
    system_prompt="You are a first-line support bot. Use escalate_to_human_agent if the issue is too complex.",
    tools=[escalate_to_human_agent],
)

human_agent = BaseAgent(
    name="HumanSupport",
    system_prompt="You are a human support specialist. Help with complex billing issues.",
)

orchestrator = Orchestrator()
orchestrator.register_agent(bot_agent)
orchestrator.register_agent(human_agent)
```

### Handoff Loop Protection

The orchestrator limits consecutive handoffs to prevent infinite cycles (e.g., Agent A → Agent B → Agent A → ...). Configure the limit via `max_handoffs` in `OrchestratorConfig` (default: 5).

```python
from multi_agent_orchestrator import OrchestratorConfig, Orchestrator

config = OrchestratorConfig(max_handoffs=3)
orchestrator = Orchestrator(config=config)
```

---

## Human-in-the-Loop (HITL)

For high-stakes operations (e.g., deleting data, sending emails), a tool can pause execution by raising `HumanApprovalRequired`. The orchestrator returns a structured pause message instead of continuing.

### Example

```python
from multi_agent_orchestrator import HumanApprovalRequired

def delete_user_account(user_id: str) -> str:
    """Delete a user account. Requires human approval."""
    raise HumanApprovalRequired(
        tool_name="delete_user_account",
        tool_args={"user_id": user_id},
        message=f"Confirm deletion of user {user_id}?",
    )
```

When this tool is called, the orchestrator returns:

```
Execution paused. Human approval required for tool 'delete_user_account'
with args {'user_id': '12345'}. Message: Confirm deletion of user 12345?
```

Your application can then present this to a human operator and re-submit the request after approval.

---

## Lifecycle Hooks

Agents support `pre_process` and `post_process` hooks for custom query/response transformation:

```python
class AuditAgent(BaseAgent):
    async def pre_process(self, query, history):
        # Log or transform the query before it reaches the model
        return f"[AUDIT] {query}"

    async def post_process(self, response):
        # Sanitize or format the response before returning
        return response.strip()
```
