"""
Groq LLM Provider Implementation.
Primary LLM provider using Groq's API.
"""

import json
from typing import List, Dict, Any, Optional, AsyncGenerator
from groq import AsyncGroq
from app.services.llm.base import BaseLLMProvider
from app.models.chat_models import LLMMessage, LLMResponse, ToolCall
from app.core.logger import get_llm_logger

logger = get_llm_logger()


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider implementation."""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        """Initialize Groq provider."""
        super().__init__(api_key, model, **kwargs)
        self.client = AsyncGroq(api_key=api_key)
        self.provider_name = "groq"
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Groq API."""
        try:
            # Format messages to Groq format
            formatted_messages = self.format_messages(messages)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": temperature,
            }
            
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            
            # Add tools if provided
            if tools:
                request_params["tools"] = self._format_tools(tools)
                request_params["tool_choice"] = "auto"
            
            logger.llm_call(
                provider=self.provider_name,
                model=self.model,
                tokens=max_tokens
            )
            
            # Make API call
            response = await self.client.chat.completions.create(**request_params)
            
            # Parse response
            message = response.choices[0].message
            content = message.content or ""
            
            # Parse tool calls if present
            tool_calls = None
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = self.parse_tool_response(message.tool_calls)
            
            # Get token usage
            tokens_used = None
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
            
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                model=self.model,
                provider=self.provider_name,
                tokens_used=tokens_used,
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            logger.error_with_context(e, {
                "provider": self.provider_name,
                "model": self.model,
                "action": "generate"
            })
            raise
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion using Groq API."""
        try:
            formatted_messages = self.format_messages(messages)
            
            request_params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": temperature,
                "stream": True
            }
            
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            
            logger.llm_call(
                provider=self.provider_name,
                model=self.model,
                tokens=max_tokens
            )
            
            stream = await self.client.chat.completions.create(**request_params)
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content
                        
        except Exception as e:
            logger.error_with_context(e, {
                "provider": self.provider_name,
                "model": self.model,
                "action": "generate_stream"
            })
            raise
    
    def parse_tool_response(self, tool_calls_raw: Any) -> Optional[List[ToolCall]]:
        """Parse Groq tool calls to universal format."""
        if not tool_calls_raw:
            return None
        
        tool_calls = []
        for tool_call in tool_calls_raw:
            try:
                function = tool_call.function
                tool_name = function.name
                
                # Parse arguments (they come as JSON string)
                tool_args = {}
                if function.arguments:
                    try:
                        tool_args = json.loads(function.arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments: {function.arguments}")
                        tool_args = {}
                
                tool_calls.append(ToolCall(
                    tool_name=tool_name,
                    tool_args=tool_args
                ))
                
            except Exception as e:
                logger.error_with_context(e, {
                    "action": "parse_tool_response",
                    "tool_call": str(tool_call)
                })
                continue
        
        return tool_calls if tool_calls else None
    
    def format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Format messages to Groq API format."""
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    
    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools to Groq function calling format."""
        formatted_tools = []
        
        for tool in tools:
            formatted_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
            }
            formatted_tools.append(formatted_tool)
        
        return formatted_tools
    
    async def count_tokens(self, text: str) -> int:
        """
        Estimate token count for Groq.
        Using rough approximation: ~4 characters per token.
        """
        return len(text) // 4
    
    async def health_check(self) -> bool:
        """Check Groq API health."""
        try:
            test_messages = [
                LLMMessage(role="user", content="Hello")
            ]
            
            response = await self.generate(
                messages=test_messages,
                max_tokens=5,
                temperature=0.0
            )
            
            return response.content is not None
            
        except Exception as e:
            logger.error_with_context(e, {
                "provider": self.provider_name,
                "action": "health_check"
            })
            return False
