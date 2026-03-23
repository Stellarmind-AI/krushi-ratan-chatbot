"""
Abstract Base Class for LLM Providers.
Ensures all LLM providers implement the same interface for provider-agnostic design.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.models.chat_models import LLMMessage, LLMResponse, ToolCall


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        """
        Initialize LLM provider.
        
        Args:
            api_key: API key for the provider
            model: Model name/identifier
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.model = model
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
        self.config = kwargs
    
    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            tools: Optional list of tools/functions for tool calling
            **kwargs: Additional provider-specific parameters
        
        Returns:
            LLMResponse: Normalized response object
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion from the LLM.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        
        Yields:
            str: Token chunks as they are generated
        """
        pass
    
    @abstractmethod
    def parse_tool_response(self, response: Any) -> Optional[List[ToolCall]]:
        """
        Parse tool calls from provider-specific response format.
        Normalizes to universal ToolCall format.
        
        Args:
            response: Raw response from provider
        
        Returns:
            List of ToolCall objects or None if no tool calls
        """
        pass
    
    @abstractmethod
    def format_messages(self, messages: List[LLMMessage]) -> Any:
        """
        Format messages to provider-specific format.
        
        Args:
            messages: Universal message format
        
        Returns:
            Provider-specific message format
        """
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Input text
        
        Returns:
            Estimated token count
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is available and API key is valid.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return self.provider_name
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(model={self.model})"
