"""
LLM Manager with Rate Limiting, Retry Logic, and Automatic Fallback.
Handles provider switching, rate limits, and ensures robust LLM operations.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.services.llm.base import BaseLLMProvider
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.openai_provider import OpenAIProvider
from app.models.chat_models import LLMMessage, LLMResponse
from app.core.config import settings
from app.core.logger import get_llm_logger

logger = get_llm_logger()


class RateLimitException(Exception):
    """Custom exception for rate limit errors."""
    pass


class LLMManager:
    """
    Manages multiple LLM providers with rate limiting and fallback.
    Ensures robust operation even under high load.
    """
    
    def __init__(self):
        """Initialize LLM manager with providers and rate limiters."""
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.rate_limiters: Dict[str, AsyncLimiter] = {}
        self.current_provider = "groq"
        
        # Initialize providers
        self._initialize_providers()
        
        # Initialize rate limiters
        self._initialize_rate_limiters()
        
        logger.info(
            "🚀 LLM Manager initialized",
            providers=list(self.providers.keys()),
            primary=self.current_provider
        )
    
    def _initialize_providers(self):
        """Initialize available LLM providers."""
        # Initialize Groq (primary)
        if settings.has_groq_key:
            try:
                self.providers["groq"] = GroqProvider(
                    api_key=settings.GROQ_API_KEY,
                    model=settings.GROQ_MODEL
                )
                logger.info("✅ Groq provider initialized", model=settings.GROQ_MODEL)
            except Exception as e:
                logger.error_with_context(e, {"provider": "groq", "action": "initialize"})
        else:
            logger.warning("⚠️  Groq API key not found")
        
        # Initialize OpenAI (optional fallback — skip silently if key is placeholder/missing)
        openai_key = settings.OPENAI_API_KEY or ""
        is_placeholder = openai_key in ("", "your_openai_api_key_here") or openai_key.startswith("your_")
        if settings.has_openai_key and not is_placeholder:
            try:
                self.providers["openai"] = OpenAIProvider(
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.OPENAI_MODEL
                )
                logger.info("✅ OpenAI provider initialized (fallback)", model=settings.OPENAI_MODEL)
            except Exception as e:
                logger.error_with_context(e, {"provider": "openai", "action": "initialize"})
        else:
            logger.info("ℹ️  OpenAI not configured — using Groq only (this is fine)")
        
        if not self.providers:
            raise ValueError("No LLM providers available! Please configure API keys.")
        
        # Set primary provider
        if "groq" in self.providers:
            self.current_provider = "groq"
        elif "openai" in self.providers:
            self.current_provider = "openai"
        else:
            self.current_provider = list(self.providers.keys())[0]
    
    def _initialize_rate_limiters(self):
        """Initialize rate limiters for each provider."""
        if "groq" in self.providers:
            self.rate_limiters["groq"] = AsyncLimiter(
                max_rate=settings.GROQ_RPM_LIMIT,
                time_period=60
            )
        
        if "openai" in self.providers:
            self.rate_limiters["openai"] = AsyncLimiter(
                max_rate=settings.OPENAI_RPM_LIMIT,
                time_period=60
            )
    
    async def _acquire_rate_limit(self, provider_name: str):
        """Acquire rate limit token before making request."""
        if provider_name in self.rate_limiters:
            limiter = self.rate_limiters[provider_name]
            try:
                async with limiter:
                    pass
            except Exception as e:
                logger.rate_limit_hit(provider=provider_name, retry_after=1.0)
                raise RateLimitException(f"Rate limit hit for {provider_name}")
    
    def _get_provider(self, provider_name: Optional[str] = None) -> BaseLLMProvider:
        """Get LLM provider by name."""
        name = provider_name or self.current_provider
        
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not available")
        
        return self.providers[name]
    
    def _get_fallback_provider(self) -> Optional[str]:
        """Get fallback provider name."""
        available = list(self.providers.keys())
        
        if self.current_provider in available:
            available.remove(self.current_provider)
        
        return available[0] if available else None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RateLimitException)
    )
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        use_fallback: bool = True,
        provider_name: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion with automatic rate limiting and fallback.
        
        Args:
            provider_name: Override which provider to use for this call.
                           None = use self.current_provider (default behaviour).
                           "openai" = force OpenAI for this call only (SQL generation).
                           "groq"   = force Groq for this call only.
        """
        provider_name = provider_name or self.current_provider
        
        try:
            await self._acquire_rate_limit(provider_name)
            
            provider = self._get_provider(provider_name)
            response = await provider.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                **kwargs
            )
            
            return response
            
        except RateLimitException:
            raise
            
        except Exception as e:
            logger.error_with_context(e, {
                "provider": provider_name,
                "action": "generate",
                "use_fallback": use_fallback
            })
            
            if use_fallback:
                fallback_name = self._get_fallback_provider()
                
                if fallback_name:
                    logger.fallback_trigger(
                        from_provider=provider_name,
                        to_provider=fallback_name,
                        reason=str(e)[:100]
                    )
                    
                    try:
                        original_provider = self.current_provider
                        self.current_provider = fallback_name
                        
                        result = await self.generate(
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            tools=tools,
                            use_fallback=False,
                            provider_name=fallback_name,
                            **kwargs
                        )
                        
                        self.current_provider = original_provider
                        
                        return result
                        
                    except Exception as fallback_error:
                        logger.error_with_context(fallback_error, {
                            "provider": fallback_name,
                            "action": "generate_fallback"
                        })
                        raise
            
            raise
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion with rate limiting."""
        provider_name = self.current_provider
        
        try:
            await self._acquire_rate_limit(provider_name)
            
            provider = self._get_provider(provider_name)
            
            async for chunk in provider.generate_stream(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                yield chunk
                
        except Exception as e:
            logger.error_with_context(e, {
                "provider": provider_name,
                "action": "generate_stream"
            })
            raise
    
    async def health_check_all(self) -> Dict[str, bool]:
        """
        Check health of all providers in parallel.
        OpenAI errors are logged at DEBUG level only (invalid key is expected in dev).
        """
        import asyncio

        async def check_one(name: str, provider):
            try:
                is_healthy = await provider.health_check()
                if is_healthy:
                    logger.info(f"✅ {name.upper()} health check passed")
                else:
                    # Only warn for providers we actually expect to work
                    if name == "openai" and not settings.has_openai_key:
                        pass  # silently skip — openai is optional
                    elif name == "openai":
                        logger.debug(f"OpenAI health check failed (key may be invalid)")
                    else:
                        logger.warning(f"⚠️  {name.upper()} health check failed")
                return name, is_healthy
            except Exception as e:
                if name == "openai":
                    # Don't spam logs with 401 errors when openai key is wrong/missing
                    logger.debug(f"OpenAI health check exception (suppressed): {type(e).__name__}")
                else:
                    logger.error_with_context(e, {"provider": name, "action": "health_check"})
                return name, False

        results = await asyncio.gather(
            *[check_one(name, provider) for name, provider in self.providers.items()]
        )
        return dict(results)


    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())
    
    def set_primary_provider(self, provider_name: str):
        """Set primary provider."""
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        
        old_provider = self.current_provider
        self.current_provider = provider_name
        
        logger.info(
            "🔄 Primary provider changed",
            from_provider=old_provider,
            to_provider=provider_name
        )


# Global LLM manager instance
llm_manager = LLMManager()


def get_llm_manager() -> LLMManager:
    """Get LLM manager instance."""
    return llm_manager