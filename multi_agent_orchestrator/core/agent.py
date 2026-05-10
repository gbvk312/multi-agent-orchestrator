import os
from typing import List, Callable, Optional, Any, Dict
from google import genai
from google.genai import types

class BaseAgent:
    """Base class for an intelligent agent powered by Gemini."""
    
    def __init__(
        self, 
        name: str, 
        system_prompt: str,
        model: str = "gemini-2.5-flash",
        tools: Optional[List[Callable]] = None
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools or []
        
        # Initialize Gemini Client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # We assume it's in the environment if not passed explicitly, but throw if missing
            pass
        self.client = genai.Client()

    def process(self, query: str, history: List[Dict[str, Any]]) -> str:
        """Processes a query with the given context history."""
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
            temperature=0.2
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        
        # Note: If tools are triggered, this basic implementation doesn't handle automatic 
        # multi-turn tool execution internally yet. It returns the text or tool call intent.
        
        if response.function_calls:
            # For demonstration, if it calls a tool we return the call info
            # In a full framework, we would execute the tool and feed it back.
            call = response.function_calls[0]
            return f"[{self.name}] Suggested tool call: {call.name} with args {call.args}"
            
        return response.text
