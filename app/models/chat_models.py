"""
Pydantic models for request/response validation in the AI Chatbot.
Ensures type safety and data validation across the application.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime


# ================================
# WEBSOCKET MESSAGE MODELS
# ================================

class WebSocketMessage(BaseModel):
    """Base model for WebSocket messages."""
    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class TextInputMessage(WebSocketMessage):
    """Text input message from client."""
    type: Literal["text_input"] = "text_input"
    text: str = Field(..., min_length=1, max_length=5000, description="User text input")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class AudioInputMessage(WebSocketMessage):
    """Audio input message from client."""
    type: Literal["audio_input"] = "audio_input"
    audio_data: str = Field(..., description="Base64 encoded audio data")
    audio_format: str = Field(default="webm", description="Audio format (webm, wav, mp3)")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class ControlMessage(WebSocketMessage):
    """Control message (e.g., clear history, stop generation)."""
    type: Literal["control"] = "control"
    action: Literal["clear_history", "stop_generation"] = Field(..., description="Control action")


class TextOutputMessage(WebSocketMessage):
    """Text output message to client (streaming)."""
    type: Literal["text_output"] = "text_output"
    text: str = Field(..., description="Generated text (word or sentence)")
    is_complete: bool = Field(default=False, description="Whether this is the final chunk")
    full_text: Optional[str] = Field(None, description="Full text when complete")


class AudioOutputMessage(WebSocketMessage):
    """Audio output message to client."""
    type: Literal["audio_output"] = "audio_output"
    audio_data: str = Field(..., description="Base64 encoded audio data")
    audio_format: str = Field(default="mp3", description="Audio format")
    is_complete: bool = Field(default=False, description="Whether this is the final audio chunk")


class StatusMessage(WebSocketMessage):
    """Status update message."""
    type: Literal["status"] = "status"
    status: Literal["thinking", "generating_sql", "executing_query", "generating_answer", "complete", "error"] = Field(
        ..., description="Current processing status"
    )
    details: Optional[str] = Field(None, description="Additional status details")


class ErrorMessage(WebSocketMessage):
    """Error message to client."""
    type: Literal["error"] = "error"
    error: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(None, description="Error type/category")


# ================================
# LLM MODELS
# ================================

class LLMMessage(BaseModel):
    """LLM message format (universal across providers)."""
    role: Literal["system", "user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")


class ToolDefinition(BaseModel):
    """Tool definition for LLM function calling."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters schema")


class ToolCall(BaseModel):
    """Tool call result from LLM."""
    tool_name: str = Field(..., description="Name of the tool to call")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class LLMResponse(BaseModel):
    """Normalized LLM response across providers."""
    content: str = Field(..., description="Response content")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls if any")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider name (groq/openai)")
    tokens_used: Optional[int] = Field(None, description="Total tokens used")
    finish_reason: Optional[str] = Field(None, description="Finish reason")


# ================================
# AGENT MODELS
# ================================

class ToolSelectionRequest(BaseModel):
    """Request for tool selection (1st LLM call)."""
    user_query: str = Field(..., description="User's question")
    condensed_schema: Dict[str, Any] = Field(..., description="Condensed database schema")
    available_tools: List[str] = Field(..., description="List of available tool names")


class ToolSelectionResponse(BaseModel):
    """Response from tool selection."""
    selected_tools: List[str] = Field(..., description="Selected tool names")
    reasoning: Optional[str] = Field(None, description="Reasoning for selection")


class QueryGenerationRequest(BaseModel):
    """Request for SQL query generation (2nd LLM call)."""
    user_query: str = Field(..., description="User's question")
    tool_schemas: List[Dict[str, Any]] = Field(..., description="Full schemas for selected tools")


class QueryGenerationResponse(BaseModel):
    """Response from query generation."""
    queries: List[Dict[str, str]] = Field(..., description="Generated SQL queries with table names")
    
    @validator("queries")
    def validate_queries(cls, v):
        """Ensure each query has table_name and sql."""
        for query in v:
            if "table_name" not in query or "sql" not in query:
                raise ValueError("Each query must have 'table_name' and 'sql' keys")
        return v


class QueryResult(BaseModel):
    """Result from query execution."""
    table_name: str = Field(..., description="Table queried")
    sql: str = Field(..., description="SQL query executed")
    rows: List[Dict[str, Any]] = Field(..., description="Query results")
    row_count: int = Field(..., description="Number of rows returned")
    execution_time: float = Field(..., description="Query execution time in seconds")


class AnswerGenerationRequest(BaseModel):
    """Request for final answer generation (3rd LLM call)."""
    user_query: str = Field(..., description="User's question")
    query_results: List[QueryResult] = Field(..., description="Results from all queries")


class AnswerGenerationResponse(BaseModel):
    """Response from answer generation."""
    answer: str = Field(..., description="Natural language answer")
    sources: Optional[List[str]] = Field(None, description="Tables used as sources")


# ================================
# DATABASE MODELS
# ================================

class TableSchema(BaseModel):
    """Full table schema definition."""
    table_name: str = Field(..., description="Table name")
    columns: List[Dict[str, Any]] = Field(..., description="Column definitions")
    relationships: Optional[List[str]] = Field(None, description="Foreign key relationships")
    engine: Optional[str] = Field(None, description="Storage engine")


class CondensedSchema(BaseModel):
    """Condensed database schema (table names and columns only)."""
    database_name: str = Field(..., description="Database name")
    total_tables: int = Field(..., description="Total number of tables")
    tables: List[Dict[str, Any]] = Field(..., description="List of tables with column names")


# ================================
# AUDIO PROCESSING MODELS
# ================================

class AudioTranscription(BaseModel):
    """Audio transcription result."""
    text: str = Field(..., description="Transcribed text")
    language: Optional[str] = Field(None, description="Detected language")
    confidence: Optional[float] = Field(None, description="Confidence score")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")


class TTSRequest(BaseModel):
    """Text-to-speech request."""
    text: str = Field(..., min_length=1, description="Text to convert to speech")
    language: str = Field(default="en", description="Language code")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")


class TTSResponse(BaseModel):
    """Text-to-speech response."""
    audio_data: str = Field(..., description="Base64 encoded audio")
    audio_format: str = Field(..., description="Audio format (mp3, wav)")
    duration: float = Field(..., description="Audio duration in seconds")


# ================================
# CHAT HISTORY MODELS
# ================================

class ChatMessage(BaseModel):
    """Single chat message in history."""
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    audio_url: Optional[str] = Field(None, description="Audio URL if available")


class ChatHistory(BaseModel):
    """Chat conversation history."""
    session_id: str = Field(..., description="Session identifier")
    messages: List[ChatMessage] = Field(default_factory=list, description="List of messages")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")


# ================================
# HEALTH CHECK MODELS
# ================================

class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "unhealthy"] = Field(..., description="Overall health status")
    database: bool = Field(..., description="Database connection status")
    llm_primary: bool = Field(..., description="Primary LLM availability")
    llm_fallback: bool = Field(..., description="Fallback LLM availability")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
