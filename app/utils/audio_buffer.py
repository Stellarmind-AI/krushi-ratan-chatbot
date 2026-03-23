"""
Audio Buffer Utility for Smooth Voice Streaming.
Implements buffering to prevent choppy audio playback.
"""

import asyncio
from typing import List, Optional
from collections import deque
from app.core.logger import get_audio_logger

logger = get_audio_logger()


class AudioBuffer:
    """
    Audio buffer for smooth streaming.
    Buffers words/chunks before sending to prevent choppy playback.
    """
    
    def __init__(
        self,
        buffer_size: int = 10,
        min_chunk_words: int = 5,
        pause_on_punctuation: bool = True
    ):
        """
        Initialize audio buffer.
        
        Args:
            buffer_size: Maximum number of chunks to buffer
            min_chunk_words: Minimum words to accumulate before sending
            pause_on_punctuation: Add natural pauses at punctuation
        """
        self.buffer_size = buffer_size
        self.min_chunk_words = min_chunk_words
        self.pause_on_punctuation = pause_on_punctuation
        
        self.buffer: deque = deque(maxlen=buffer_size)
        self.word_accumulator: List[str] = []
        
        # Punctuation that triggers a chunk
        self.chunk_punctuation = {'.', '!', '?', '।'}  # Including Devanagari danda
        
        # Punctuation that adds a pause
        self.pause_punctuation = {'.', '!', '?', ';', ':', '।'}
        
        logger.debug(
            "Audio buffer initialized",
            buffer_size=buffer_size,
            min_words=min_chunk_words
        )
    
    def add_word(self, word: str) -> Optional[str]:
        """
        Add a word to the buffer.
        Returns a chunk if ready to send.
        
        Args:
            word: Word to add
        
        Returns:
            Chunk of text if ready, None otherwise
        """
        if not word:
            return None
        
        self.word_accumulator.append(word)
        
        # Check if we should create a chunk
        should_chunk = False
        
        # Chunk on sentence-ending punctuation
        if any(punct in word for punct in self.chunk_punctuation):
            should_chunk = True
        # Or if we've accumulated enough words
        elif len(self.word_accumulator) >= self.min_chunk_words:
            should_chunk = True
        
        if should_chunk:
            chunk = ' '.join(self.word_accumulator)
            self.word_accumulator = []
            
            # Add to buffer
            self.buffer.append(chunk)
            
            return chunk
        
        return None
    
    def add_text(self, text: str) -> List[str]:
        """
        Add text and split into buffered chunks.
        
        Args:
            text: Text to add
        
        Returns:
            List of chunks ready to send
        """
        chunks = []
        words = text.split()
        
        for word in words:
            chunk = self.add_word(word)
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def flush(self) -> Optional[str]:
        """
        Flush remaining words as a final chunk.
        
        Returns:
            Final chunk if any words remain
        """
        if self.word_accumulator:
            chunk = ' '.join(self.word_accumulator)
            self.word_accumulator = []
            self.buffer.append(chunk)
            return chunk
        return None
    
    def get_all_chunks(self) -> List[str]:
        """
        Get all buffered chunks.
        
        Returns:
            List of all chunks in buffer
        """
        return list(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.word_accumulator = []
    
    def add_pause(self, text: str) -> str:
        """
        Add natural pauses to text at punctuation marks.
        
        Args:
            text: Input text
        
        Returns:
            Text with pause markers (SSML-like)
        """
        if not self.pause_on_punctuation:
            return text
        
        # Add pauses after punctuation
        for punct in self.pause_punctuation:
            # Short pause (250ms) after commas
            if punct == ',':
                text = text.replace(punct, f'{punct} <break time="250ms"/> ')
            # Medium pause (400ms) after semicolons and colons
            elif punct in {';', ':'}:
                text = text.replace(punct, f'{punct} <break time="400ms"/> ')
            # Long pause (600ms) after sentence endings
            else:
                text = text.replace(punct, f'{punct} <break time="600ms"/> ')
        
        return text


class StreamingAudioBuffer:
    """
    Streaming audio buffer for real-time audio generation.
    Manages the flow between text generation and audio synthesis.
    """
    
    def __init__(self, tts_provider, buffer_words: int = 10):
        """
        Initialize streaming audio buffer.
        
        Args:
            tts_provider: TTS provider instance
            buffer_words: Number of words to buffer before synthesis
        """
        self.tts_provider = tts_provider
        self.buffer_words = buffer_words
        self.text_queue: asyncio.Queue = asyncio.Queue()
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        
        logger.debug("Streaming audio buffer initialized", buffer_words=buffer_words)
    
    async def add_text(self, text: str):
        """Add text to the queue for audio synthesis."""
        await self.text_queue.put(text)
    
    async def mark_complete(self):
        """Mark text generation as complete."""
        await self.text_queue.put(None)  # Sentinel value
    
    async def start_processing(self):
        """Start processing text and generating audio."""
        self.is_running = True
        
        word_buffer = []
        
        while self.is_running:
            try:
                # Get text from queue (with timeout)
                text = await asyncio.wait_for(
                    self.text_queue.get(),
                    timeout=0.5
                )
                
                # Check for completion sentinel
                if text is None:
                    # Flush remaining words
                    if word_buffer:
                        buffered_text = ' '.join(word_buffer)
                        audio_response = await self.tts_provider.synthesize(buffered_text)
                        await self.audio_queue.put(audio_response)
                    
                    # Mark audio generation complete
                    await self.audio_queue.put(None)
                    break
                
                # Add words to buffer
                words = text.split()
                word_buffer.extend(words)
                
                # If buffer is large enough, synthesize
                if len(word_buffer) >= self.buffer_words:
                    buffered_text = ' '.join(word_buffer)
                    word_buffer = []
                    
                    # Generate audio
                    audio_response = await self.tts_provider.synthesize(buffered_text)
                    await self.audio_queue.put(audio_response)
                    
            except asyncio.TimeoutError:
                # No text received, continue waiting
                continue
            except Exception as e:
                logger.error_with_context(e, {"action": "process_text_to_audio"})
                await self.audio_queue.put(None)
                break
    
    async def get_audio(self) -> Optional[any]:
        """Get next audio chunk from the queue."""
        return await self.audio_queue.get()
    
    def stop(self):
        """Stop processing."""
        self.is_running = False
