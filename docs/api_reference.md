# API Reference

This page contains the API reference for the core components of the Multi-Agent Orchestrator.

## Orchestrator

::: multi_agent_orchestrator.core.orchestrator.Orchestrator

::: multi_agent_orchestrator.core.orchestrator.OrchestratorError

## BaseAgent

::: multi_agent_orchestrator.core.agent.BaseAgent

## Exceptions

::: multi_agent_orchestrator.core.agent.AgentError

::: multi_agent_orchestrator.core.agent.AgentHandoff

::: multi_agent_orchestrator.core.agent.HumanApprovalRequired

## Configuration

::: multi_agent_orchestrator.core.config.OrchestratorConfig

## Memory Management

### MemoryManager

::: multi_agent_orchestrator.core.memory.MemoryManager

### MemoryBackend (Interface)

::: multi_agent_orchestrator.core.memory.MemoryBackend

### Backends

#### RedisMemoryBackend

::: multi_agent_orchestrator.core.redis_backend.RedisMemoryBackend

#### SQLiteMemoryBackend

::: multi_agent_orchestrator.core.sqlite_backend.SQLiteMemoryBackend

## Event System

### EventHandler

::: multi_agent_orchestrator.core.events.EventHandler

### Event Types

::: multi_agent_orchestrator.core.events.OrchestratorStartEvent

::: multi_agent_orchestrator.core.events.OrchestratorRouteEvent

::: multi_agent_orchestrator.core.events.OrchestratorHandoffEvent

::: multi_agent_orchestrator.core.events.OrchestratorFinishEvent

::: multi_agent_orchestrator.core.events.OrchestratorErrorEvent

::: multi_agent_orchestrator.core.events.AgentStartEvent

::: multi_agent_orchestrator.core.events.AgentFinishEvent

::: multi_agent_orchestrator.core.events.ToolCallEvent

::: multi_agent_orchestrator.core.events.ToolResultEvent
