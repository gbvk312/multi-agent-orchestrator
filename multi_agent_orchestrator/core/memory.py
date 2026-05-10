from typing import List, Dict, Any

class MemoryManager:
    """Manages conversation context across different agents."""
    
    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}

    def add_message(self, session_id: str, role: str, content: str):
        """Adds a message to the session's history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({
            "role": role,
            "content": content
        })

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves the history for a given session."""
        return self.sessions.get(session_id, [])

    def clear(self, session_id: str):
        """Clears the history for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
