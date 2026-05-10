import os
from typing import Dict, Any, Optional
from google import genai
from google.genai import types

from .agent import BaseAgent
from .memory import MemoryManager

class Orchestrator:
    """Central router that delegates tasks to specialized agents."""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.agents: Dict[str, BaseAgent] = {}
        self.memory = memory_manager or MemoryManager()
        self.client = genai.Client()
        self.model = "gemini-2.5-flash"
        
    def register_agent(self, agent: BaseAgent):
        """Registers an agent with the orchestrator."""
        self.agents[agent.name] = agent
        
    def _route_request(self, query: str) -> str:
        """Determines which agent should handle the request."""
        if not self.agents:
            raise ValueError("No agents registered with the orchestrator.")
            
        agent_names = list(self.agents.keys())
        agent_descriptions = "\n".join([f"- {name}: {agent.system_prompt[:100]}..." for name, agent in self.agents.items()])
        
        routing_prompt = f"""
        You are a routing supervisor. Based on the user's query, you must decide which agent is best suited to handle the request.
        
        Available agents:
        {agent_descriptions}
        
        User Query: "{query}"
        
        Respond ONLY with the exact name of the selected agent. If no agent is suitable, respond with "Default".
        """
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=routing_prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        
        selected_agent = response.text.strip()
        if selected_agent in self.agents:
            return selected_agent
            
        # Fallback to the first registered agent if routing fails
        return agent_names[0]

    def process_request(self, session_id: str, query: str) -> str:
        """Processes a user request by routing it to the appropriate agent."""
        
        # 1. Determine which agent should handle this
        target_agent_name = self._route_request(query)
        target_agent = self.agents[target_agent_name]
        
        # 2. Get session history
        history = self.memory.get_history(session_id)
        
        # 3. Add current query to memory
        self.memory.add_message(session_id, "user", query)
        
        # 4. Delegate to the agent
        print(f"[{self.__class__.__name__}] Routing to -> {target_agent_name}")
        response_text = target_agent.process(query, history)
        
        # 5. Save response to memory
        self.memory.add_message(session_id, "model", response_text)
        
        return response_text
