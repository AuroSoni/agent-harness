from abc import ABC, abstractmethod
from agent_base.core.config import AgentConfig
from agent_base.core.messages import Message

class Agent(ABC):
    
    def __init__(self, ):
        """
        Initialize the agent config.
        """
        
    @abstractmethod
    def run():
        """
        Run the agent.
        """
        
    @abstractmethod
    def resume_run_with_tool_results():
        """
        Resume the agent run with the frontend tool results.
        """