#!/usr/bin/env python3
"""
Cross-platform PDF and EPUB text extraction tool

Installation:
1. Python packages (all platforms):
   pip install pymupdf pdfplumber pypdf pdfminer.six pytesseract pdf2image kraken easyocr paddleocr python-doctr ocrmypdf camelot-py opencv-python importlib-metadata tqdm

   Additional EPUB dependencies:
   pip install ebooklib beautifulsoup4 html2text mobi

2. System dependencies:
   Windows:
   - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
   - Ghostscript: https://ghostscript.com/releases/gsdnld.html
   - Poppler: https://github.com/oschwartz10612/poppler-windows/releases/

   macOS:
   brew install tesseract poppler ghostscript

   Linux:
   sudo apt-get install tesseract-ocr poppler-utils ghostscript
"""
import warnings
# Suppress common warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')

import os
import re
import sys
import logging
import argparse
from typing import Optional, List, Dict, Any,  Union, Tuple, Callable
import textwrap
import pkg_resources
import traceback
from contextlib import contextmanager
import shutil
import platform
import subprocess
import tempfile
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import signal
import threading
from datetime import datetime
from types import MappingProxyType
import time

# Thread-local storage for LLM clients
thread_local = threading.local()

# Limit concurrent connections to Ollama - prevents overwhelming the server
ollama_semaphore = threading.Semaphore(2)  # Allow only 2 concurrent connections
llm_semaphore = threading.Semaphore(2)

# Global lock for thread-safe file operations
file_lock = threading.Lock()

# Define a shutdown flag for graceful termination
shutdown_flag = threading.Event()

# Constants for LLM model
#MODEL_NAME = "cas/spaetzle-v85-7b" 
MODEL_NAME = "cas/llama-3.2-3b-instruct:latest"
# Can also use "cas/llama-3.1-8b-instruct" or other Ollama models


# Import OpenAI client for Ollama
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI client library is not installed. Please install it using 'pip install openai'.")
    print("Required for --sort functionality")

class timeout:
    """Context manager for timeout"""
    def __init__(self, seconds):
        self.seconds = seconds
        self.timer = None

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(f"Operation timed out after {self.seconds} seconds")

    def __enter__(self):
        if self.seconds > 0:
            self.timer = signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        if self.seconds > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.timer)
            
class ProgressProxy:
    """Proxy for tqdm progress bar that can update description with engine info"""
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar
        self.current_engine = None
        
    def update(self, n: int = 1, engine: Optional[str] = None):
        """Update progress and optionally show current engine"""
        if engine and engine != self.current_engine:
            self.current_engine = engine
            self.progress_bar.set_description(f"Extracting text [{engine}]")
        self.progress_bar.update(n)


class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """
        Send chat completion request to the LLM provider
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            timeout: Timeout in seconds
            
        Returns:
            Dict with response content
        """
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Either pass as api_key or set OPENAI_API_KEY environment variable")
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            if not hasattr(thread_local, "openai_client"):
                thread_local.openai_client = OpenAI(api_key=self.api_key)
            return thread_local.openai_client
        except ImportError:
            raise ImportError("OpenAI client library is required. Install with 'pip install openai'")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """Send chat completion request to OpenAI"""
        client = self._init_client()
        
        with llm_semaphore:
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                
                # Convert response to a standard format
                return {
                    "id": getattr(response, "id", "unknown"),
                    "content": response.choices[0].message.content.strip(),
                    "finish_reason": response.choices[0].finish_reason,
                    "model": self.model_name
                }
            except Exception as e:
                logging.error(f"OpenAI request failed: {str(e)}")
                raise

class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API provider"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        # HF can work without API key for some models
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """Send chat completion request to HuggingFace Inference API"""
        try:
            from huggingface_hub import InferenceClient
            
            # Format messages into HF-compatible prompt
            prompt = self._format_messages_for_hf(messages)
            
            # Initialize client with or without token
            client = InferenceClient(token=self.api_key) if self.api_key else InferenceClient()
            
            with llm_semaphore:
                response = client.text_generation(
                    prompt,
                    model=self.model_name,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    do_sample=True,
                    timeout=timeout
                )
                
                return {
                    "id": "hf-inference",
                    "content": str(response).strip(),
                    "finish_reason": "stop",
                    "model": self.model_name
                }
                
        except Exception as e:
            logging.error(f"HuggingFace inference error: {str(e)}")
            raise
    
    def _format_messages_for_hf(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages for HuggingFace text generation"""
        formatted_prompt = ""
        
        # Add system message if present
        system_msgs = [msg for msg in messages if msg["role"] == "system"]
        if system_msgs:
            formatted_prompt = f"<s>[INST] {system_msgs[0]['content']} [/INST]</s>\n\n"
        
        # Add user/assistant messages
        for msg in [m for m in messages if m["role"] != "system"]:
            if msg["role"] == "user":
                formatted_prompt += f"<s>[INST] {msg['content']} [/INST]"
            elif msg["role"] == "assistant":
                formatted_prompt += f" {msg['content']}</s>\n"
        
        # Handle last message if it's from user
        if messages[-1]["role"] == "user":
            formatted_prompt += "</s>"
            
        return formatted_prompt

class CohereProvider(LLMProvider):
    """Cohere API provider"""
    
    def __init__(self, model_name: str = "command-r-plus", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key required. Either pass as api_key or set COHERE_API_KEY environment variable")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """Send chat completion request to Cohere"""
        try:
            import cohere
            
            with llm_semaphore:
                try:
                    # Try ClientV2 first
                    client = cohere.ClientV2(self.api_key)
                    cohere_messages = self._format_messages_for_cohere(messages)
                    
                    response = client.chat(
                        model=self.model_name,
                        messages=cohere_messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    return {
                        "id": "cohere-v2",
                        "content": response.message.content[0].text,
                        "finish_reason": "stop",
                        "model": self.model_name
                    }
                    
                except Exception as e:
                    logging.warning(f"Cohere V2 failed, trying V1: {e}")
                    
                    # Fallback to V1
                    client = cohere.Client(self.api_key)
                    # Get the last user message
                    last_user_msg = next((msg["content"] for msg in reversed(messages) 
                                        if msg["role"] == "user"), "")
                    
                    response = client.chat(
                        message=last_user_msg,
                        model=self.model_name,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    return {
                        "id": "cohere-v1",
                        "content": response.text,
                        "finish_reason": "stop",
                        "model": self.model_name
                    }
                    
        except ImportError:
            raise ImportError("Cohere client library is required. Install with 'pip install cohere'")
        except Exception as e:
            logging.error(f"Cohere request failed: {str(e)}")
            raise
    
    def _format_messages_for_cohere(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for Cohere API"""
        cohere_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                cohere_messages.append({"role": "SYSTEM", "content": msg["content"]})
            elif msg["role"] == "user":
                cohere_messages.append({"role": "USER", "content": msg["content"]})
            elif msg["role"] == "assistant":
                cohere_messages.append({"role": "CHATBOT", "content": msg["content"]})
                
        return cohere_messages

class GLHFProvider(LLMProvider):
    """GLHF API provider (uses OpenAI-compatible API)"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("GLHF_API_KEY")
        if not self.api_key:
            raise ValueError("GLHF API key required. Either pass as api_key or set GLHF_API_KEY environment variable")
        self._init_client()
    
    def _init_client(self):
        """Initialize GLHF client (using OpenAI client with custom base URL)"""
        try:
            from openai import OpenAI
            if not hasattr(thread_local, "glhf_client"):
                thread_local.glhf_client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://glhf.chat/api/openai/v1"
                )
            return thread_local.glhf_client
        except ImportError:
            raise ImportError("OpenAI client library is required for GLHF. Install with 'pip install openai'")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """Send chat completion request to GLHF"""
        client = self._init_client()
        
        # Format model id for GLHF
        model_id = f"hf:{self.model_name}" if not self.model_name.startswith("hf:") else self.model_name
        
        with llm_semaphore:
            try:
                # GLHF works best with streaming
                response_chunks = []
                
                completion = client.chat.completions.create(
                    stream=True,
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        response_chunks.append(chunk.choices[0].delta.content)
                        
                full_response = "".join(response_chunks)
                
                return {
                    "id": "glhf",
                    "content": full_response,
                    "finish_reason": "stop",
                    "model": self.model_name
                }
                
            except Exception as e:
                logging.error(f"GLHF request failed: {str(e)}")
                raise

class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local LLMs"""
    
    def __init__(self, model_name: str = "cas/llama-3.2-3b-instruct:latest", 
                base_url: str = "http://localhost:11434/v1/"):
        super().__init__(model_name)
        self.base_url = base_url
        # Use OpenAI client with Ollama
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client for Ollama"""
        try:
            from openai import OpenAI
            if not hasattr(thread_local, "ollama_client"):
                thread_local.ollama_client = OpenAI(
                    base_url=self.base_url,
                    api_key="ollama"  # Placeholder value for Ollama
                )
            return thread_local.ollama_client
        except ImportError:
            raise ImportError("OpenAI client library is required. Install with 'pip install openai'")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """Send chat completion request to Ollama"""
        client = self._init_client()
        
        with llm_semaphore:
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=messages,
                    timeout=timeout
                )
                
                # Convert response to a standard format
                return {
                    "id": getattr(response, "id", "unknown"),
                    "content": response.choices[0].message.content.strip(),
                    "finish_reason": response.choices[0].finish_reason,
                    "model": self.model_name
                }
            except Exception as e:
                logging.error(f"Ollama request failed: {str(e)}")
                raise


class GroqProvider(LLMProvider):
    """Groq LLM provider for cloud LLMs"""
    
    def __init__(self, model_name: str = "llama3-70b-8192", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required. Either pass as api_key or set GROQ_API_KEY environment variable")
        self._init_client()
    
    def _init_client(self):
        """Initialize Groq client"""
        try:
            from groq import Groq
            if not hasattr(thread_local, "groq_client"):
                thread_local.groq_client = Groq(api_key=self.api_key)
            return thread_local.groq_client
        except ImportError:
            raise ImportError("Groq client library is required. Install with 'pip install groq'")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """Send chat completion request to Groq"""
        client = self._init_client()
        
        with llm_semaphore:
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                
                # Convert response to a standard format
                return {
                    "id": getattr(response, "id", "unknown"),
                    "content": response.choices[0].message.content.strip(),
                    "finish_reason": response.choices[0].finish_reason,
                    "model": self.model_name
                }
            except Exception as e:
                logging.error(f"Groq request failed: {str(e)}")
                raise


class PoeProvider(LLMProvider):
    """Poe.com LLM provider"""
    
    def __init__(self, model_name: str = "claude-3-opus", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("POE_API_KEY")
        if not self.api_key:
            raise ValueError("Poe API key required. Either pass as api_key or set POE_API_KEY environment variable")
        self.api_url = "https://api.poe.com/api/chat"
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """Send chat completion request to Poe.com"""
        
        # Format messages for Poe API
        formatted_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                # Poe doesn't support system messages directly,
                # prepend to first user message instead
                continue
            formatted_messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        # Add system message to the first user message if present
        system_messages = [msg for msg in messages if msg['role'] == 'system']
        if system_messages and formatted_messages:
            for user_msg in formatted_messages:
                if user_msg['role'] == 'user':
                    user_msg['content'] = f"[System Instructions: {system_messages[0]['content']}]\n\n{user_msg['content']}"
                    break
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        with llm_semaphore:
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()
                response_data = response.json()
                
                # Extract and return content in a standard format
                return {
                    "id": response_data.get("id", "unknown"),
                    "content": response_data.get("choices", [{"message": {"content": ""}}])[0]["message"]["content"].strip(),
                    "finish_reason": response_data.get("choices", [{"finish_reason": "unknown"}])[0]["finish_reason"],
                    "model": self.model_name
                }
            except requests.RequestException as e:
                logging.error(f"Poe request failed: {str(e)}")
                raise


def get_llm_provider(provider_type: str = "ollama", 
                    model_name: Optional[str] = None,
                    api_key: Optional[str] = None) -> LLMProvider:
    """
    Factory function to get an LLM provider based on type
    """
    provider_type = provider_type.lower()
    
    # Default model names for each provider
    default_models = {
        "ollama": "cas/spaetzle-v85-7b",
        "groq": "llama-3.1-70b-versatile",
        "openai": "gpt-3.5-turbo",
        "cohere": "command-r-plus",
        "huggingface": "mistralai/Mistral-7B-Instruct-v0.3",
        "glhf": "mistralai/Mistral-7B-Instruct-v0.3"
    }
    
    # Use default model if none specified
    if not model_name:
        model_name = default_models.get(provider_type, default_models["ollama"])
    
    if provider_type == "ollama":
        return OllamaProvider(model_name=model_name)
    elif provider_type == "groq":
        return GroqProvider(model_name=model_name, api_key=api_key)
    elif provider_type == "openai":
        return OpenAIProvider(model_name=model_name, api_key=api_key)
    elif provider_type == "cohere":
        return CohereProvider(model_name=model_name, api_key=api_key)
    elif provider_type == "huggingface":
        return HuggingFaceProvider(model_name=model_name, api_key=api_key)
    elif provider_type == "glhf":
        return GLHFProvider(model_name=model_name, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


def send_to_llm(text: str, filename: str, provider: Union[str, LLMProvider], 
              model_name: Optional[str] = None,
              api_key: Optional[str] = None,
              max_attempts: int = 5,
              verbose: bool = False) -> str:
    """
    Extract metadata from text using an LLM provider
    
    Args:
        text: Text to analyze
        filename: Filename for context
        provider: Provider type string ('ollama', 'groq', 'poe') or LLMProvider instance
        model_name: Optional model name to use
        api_key: Optional API key for cloud providers
        max_attempts: Maximum retry attempts
        verbose: Whether to print debug information
        
    Returns:
        str: The formatted metadata response
    """
    if isinstance(provider, (int, float)):
        logging.error(f"Invalid provider parameter: {provider} (type: {type(provider)}). Expected provider string or LLMProvider instance.")
        return ""
        
    # Get or validate provider
    if isinstance(provider, str):
        try:
            llm_provider = get_llm_provider(provider, model_name, api_key)
        except ImportError as e:
            logging.error(f"Failed to initialize {provider} provider: {str(e)}")
            # Fall back to Ollama if available
            try:
                llm_provider = get_llm_provider("ollama")
                logging.info(f"Falling back to Ollama provider")
            except ImportError:
                raise ImportError("No LLM providers available. Install at least one client library.")
    else:
        llm_provider = provider
    
    base_retry_wait = 2  # Base wait time in seconds
    
    # Prepare different prompt templates to try if earlier ones fail
    prompt_templates = [
        # First attempt - simple structured format
        (
            f"Extract the author name (lastname surname) of the main author (ignore other authors), "
            f"year of publication, title, and language from the following text, considering the filename '{os.path.basename(filename)}' "
            f"which may contain clues. I need the output **only** in the following format with no additional text or explanations: \n"
            f"<TITLE>The publication title</TITLE>\n<YEAR>2023</YEAR>\n<AUTHOR>Lastname Firstname</AUTHOR>\n<LANGUAGE>en</LANGUAGE>\n\n"
        ),
        # Second attempt - emphasize exact format
        (
            f"I need to extract metadata from a document. Please give me ONLY these four tags with the information, and nothing else:\n"
            f"<TITLE>The exact title</TITLE>\n<YEAR>The publication year (4 digits)</YEAR>\n<AUTHOR>The author's name in 'Lastname Firstname' format</AUTHOR>\n<LANGUAGE>The language code</LANGUAGE>\n\n"
            f"Document filename: {os.path.basename(filename)}\n"
        ),
        # Third attempt - even more explicit
        (
            f"You are a metadata extraction tool. Extract these fields from the text:\n"
            f"1. TITLE (the full publication title)\n"
            f"2. YEAR (the 4-digit publication year, use 'Unknown' if not found)\n"
            f"3. AUTHOR (format as 'Lastname Firstname', preserve any commas)\n"
            f"4. LANGUAGE (the 2-letter language code, e.g., 'en', 'de', 'fr')\n\n"
            f"Format your response EXACTLY like this with no other text:\n"
            f"<TITLE>The title</TITLE>\n<YEAR>2023</YEAR>\n<AUTHOR>Smith John</AUTHOR>\n<LANGUAGE>en</LANGUAGE>\n\n"
        )
    ]
    
    # Try different prompt templates if we encounter format issues
    attempt = 1
    while attempt <= max_attempts:
        # Choose prompt template based on attempt number
        template_index = min(attempt - 1, len(prompt_templates) - 1)
        prompt_template = prompt_templates[template_index]
        
        logging.debug(f"Consulting LLM {provider} on file: {filename} (Attempt: {attempt}, Template: {template_index + 1})")


        # Build the final prompt with text sample
        prompt = prompt_template + f"Here is the document text:\n{text[:3000]}"  # Limit text to avoid token limits
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = llm_provider.chat_completion(
                messages=messages,
                temperature=0.5,  # Reduced temperature for more consistent formatting
                max_tokens=250,
                timeout=120  # 2 minute timeout
            )
            
            output = response["content"]
            if verbose:
                logging.debug(f"Metadata content received from LLM: {output}")
            
            # Validate the response has the expected format
            metadata = parse_metadata(output, verbose=verbose)
            if metadata:
                return output
            else:
                logging.warning(f"Unexpected response format from LLM: {output}")
                # Less aggressive backoff for format issues
                if attempt < max_attempts:
                    wait_time = base_retry_wait * (1.5 ** (attempt - 1))  # Gentler exponential backoff
                    logging.info(f"Retrying with different prompt in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                return output
            
        except Exception as e:
            if "rate_limit" in str(e).lower() or "timeout" in str(e).lower():
                # Use exponential backoff for rate limiting/timeouts
                wait_time = base_retry_wait * (2 ** (attempt - 1))  # Exponential backoff
                logging.info(f"Rate limit or timeout encountered. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                attempt += 1
                continue
            else:
                logging.error(f"Error communicating with LLM for {filename}: {e}")
                if attempt < max_attempts:
                    wait_time = base_retry_wait * (1.5 ** (attempt - 1))  # Gentler exponential backoff
                    logging.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
            return ""
            
    logging.error(f"Maximum retry attempts reached for LLM request.")
    return ""

class ImportCache:
    """Global cache for imports and their availability"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._modules = {}
            cls._instance._available = {}
        return cls._instance
    
    def is_available(self, module_name: str, submodules: List[str] = None) -> bool:
        """Check if module and optional submodules can be imported"""
        # Import importlib here to ensure it's available
        import importlib.util
        
        cache_key = f"{module_name}:{','.join(submodules or [])}"
        if cache_key not in self._available:
            try:
                # Check main module
                if importlib.util.find_spec(module_name) is None:
                    self._available[cache_key] = False
                    return False
                # Check submodules if specified
                if submodules:
                    for submodule in submodules:
                        full_name = f"{module_name}.{submodule}"
                        if importlib.util.find_spec(full_name) is None:
                            self._available[cache_key] = False
                            return False
                self._available[cache_key] = True
            except Exception:
                self._available[cache_key] = False
        return self._available[cache_key]

    def import_module(self, module_name: str, submodule: str = None) -> Any:
        """Import and cache a module"""
        # Import importlib here to ensure it's available
        import importlib
        
        cache_key = f"{module_name}{f'.{submodule}' if submodule else ''}"
        if cache_key not in self._modules:
            try:
                if submodule:
                    main_module = importlib.import_module(module_name)
                    self._modules[cache_key] = getattr(main_module, submodule)
                else:
                    self._modules[cache_key] = importlib.import_module(module_name)
            except ImportError as e:
                raise ImportError(f"Failed to import {cache_key}: {e}")
        return self._modules[cache_key]

class ExtractionManager:
    """Central manager for text extraction operations"""
    
    def __init__(self, debug: bool = False):
        self._debug = debug
        self._setup_logging(debug)
        self._binaries = self._check_binaries()
        self._versions = self._check_versions()
        self._extractors = {}
        
    def _setup_logging(self, debug: bool):
        """Configure logging with appropriate level and handlers"""
        level = logging.DEBUG if debug else logging.INFO
        logging.getLogger().setLevel(level)

        # Suppress common warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Suppress detailed logs from specific libraries
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('pdf2image').setLevel(logging.WARNING)
        logging.getLogger('pytesseract').setLevel(logging.WARNING)
        logging.getLogger('pdfminer').setLevel(logging.WARNING)
        logging.getLogger('pypdf').setLevel(logging.WARNING)
        logging.getLogger('camelot').setLevel(logging.WARNING)
        logging.getLogger('pymupdf').setLevel(logging.WARNING)
        
        # Additional debug logging suppression for PyPDF
        if not debug:
            logging.getLogger('pypdf').setLevel(logging.ERROR)
        else:
            # Even in debug mode, limit some excessive loggers
            logging.getLogger('pypdf.filters').setLevel(logging.INFO)
            logging.getLogger('pypdf.xref').setLevel(logging.INFO)
            logging.getLogger('pypdf.generic').setLevel(logging.INFO)

    def _check_binaries(self) -> Dict[str, bool]:
        """Check availability of system binaries"""
        binaries = {
            'tesseract': False,
            'pdftoppm': False,
            'gs': False
        }
        
        for binary in binaries:
            try:
                result = subprocess.run(
                    [binary, '--version'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    encoding='utf-8'
                )
                binaries[binary] = True
                logging.debug(f"Found {binary}: {result.stdout.splitlines()[0]}")
            except Exception as e:
                logging.debug(f"Binary {binary} not found: {e}")
        
        return binaries

    def _check_versions(self) -> Dict[str, str]:
        """Get versions of installed Python packages"""
        versions = {}
        packages = [
            'pymupdf', 'pdfplumber', 'pypdf', 'pdfminer.six',
            'pytesseract', 'pdf2image', 'easyocr',
            'paddleocr', 'python-doctr', 'ocrmypdf', 'camelot-py',
            'ebooklib', 'beautifulsoup4', 'html2text', 'kraken'
        ]
        
        for package in packages:
            try:
                version = pkg_resources.get_distribution(package).version
                versions[package] = version
                logging.debug(f"Found {package} version {version}")
            except Exception as e:
                logging.debug(f"Package {package} not found: {e}")
        
        return versions

    def _get_extractor(self, file_path: str) -> Union['PDFExtractor', 'EPUBExtractor']:
        """Get or create appropriate extractor for file type"""
        file_ext = os.path.splitext(file_path)[1].lower()
        cache_key = f"{file_ext}:{file_path}"
        
        if cache_key not in self._extractors:
            if file_ext == '.pdf':
                self._extractors[cache_key] = PDFExtractor(debug=self._debug)
            elif file_ext == '.epub':
                # Create and pass the ImportCache instance
                import_cache = ImportCache()
                self._extractors[cache_key] = EPUBExtractor(import_cache=import_cache, debug=self._debug)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
        
        return self._extractors[cache_key]
    
    def extract(self, input_path: str, 
        output_path: Optional[str] = None,
        method: Optional[str] = None,
        ocr_method: Optional[str] = None,
        password: Optional[str] = None,
        extract_tables: bool = False,
        sort: bool = False,     # with callback for sort processing
        llm_provider = None,      
        rename_script_path: Optional[str] = None,
        **kwargs) -> Union[str, bool]:
        """
        Extract text from a document with progress reporting and error handling
        
        Args:
            input_path: Path to input document
            output_path: Optional path for output text file
            method: Preferred extraction method
            password: Password for encrypted documents
            extract_tables: Whether to extract tables (PDF only)
            sort: Whether to sort files based on content
            openai_client: OpenAI client for Ollama communication
            rename_script_path: Path to write rename commands
            **kwargs: Additional extraction options
            
        Returns:
            Extracted text if output_path is None, else success boolean
        """
        try:
            # Get appropriate extractor
            extractor = self._get_extractor(input_path)
            
            # Configure extraction
            if password:
                extractor.set_password(password)
            
            # Extract text with progress reporting
            with tqdm(desc="Extracting text", disable=not self._debug, unit='pages') as pbar:
                def progress_callback(n: int, engine: Optional[str] = None):
                    if engine:
                        pbar.set_description(f"Extracting text [{engine}]")
                    pbar.update(n)
                
                # Only pass extract_tables parameter to PDFExtractor
                extraction_kwargs = kwargs.copy()
                if isinstance(extractor, PDFExtractor) and extract_tables:
                    extraction_kwargs['extract_tables'] = extract_tables
                
                text = extractor.extract_text(
                    input_path,
                    preferred_method=method,
                    ocr_method=ocr_method,
                    progress_callback=progress_callback,
                    **extraction_kwargs
                )
                
            # Validate and handle output
            if not self._validate_text(text):
                logging.warning("Extracted text may be of low quality")
            
            # Handle sorting if requested
            if sort and llm_provider and rename_script_path and text:
                try:
                    # Get metadata from Ollama server
                    metadata_content = extract_metadata(text, input_path, llm_provider)
                    if metadata_content:
                        metadata = parse_metadata(metadata_content)
                        if metadata:
                            # Process author names
                            author = metadata['author']
                            logging.debug(f"extracted author: {author}")
                            corrected_author = sort_author_names(
                                author_names=author,
                                provider=llm_provider,
                                #temperature=temperature,
                                #max_tokens=max_tokens
                            )
                            logging.debug(f"corrected author: {corrected_author}")
                            metadata['author'] = corrected_author
                            
                            # Get file details
                            title = metadata['title']
                            year = metadata['year']
                            if not year or year == "Unknown":
                                year = "UnknownYear"
                                
                            if not author or not title:
                                logging.warning(f"Missing author or title for {input_path}. Skipping rename.")
                            else:
                                # Create target paths
                                first_author = sanitize_filename(corrected_author)

                                target_dir = os.path.join(os.path.dirname(input_path), first_author)
                                file_extension = os.path.splitext(input_path)[1].lower()
                                new_filename = f"{year} {sanitize_filename(title)}{file_extension}"
                                logging.debug(f"New filename will be: {new_filename}")

                                # Add rename command - Pass output_path to determine the text file location
                                output_dir = os.path.dirname(output_path) if output_path else None
                                add_rename_command(
                                    rename_script_path, 
                                    input_path, 
                                    target_dir, 
                                    new_filename, 
                                    output_dir=output_dir  # Pass output directory for text file location
                                )
                        else:
                            logging.warning(f"Failed to parse metadata for {input_path}")
                    else:
                        logging.warning(f"Failed to get metadata from Ollama server for {input_path}")
                except Exception as sort_e:
                    logging.error(f"Error sorting file {input_path}: {sort_e}")
                        
            # Write output file if path provided
            if output_path:
                # Make sure the directory exists
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                logging.info(f"Writing text file: {output_path}")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                return True
            return text
            
        except Exception as e:
            error_context = self._recover_from_error(e, "extraction")
            if error_context:
                logging.error(f"Extraction failed: {error_context}")
            else:
                logging.error(f"Extraction failed: {str(e)}")
                if self._debug:
                    traceback.print_exc()
            return False if output_path else ""

    @contextmanager
    def _progress_context(self, message: str):
        """Context manager for progress reporting"""
        progress = tqdm(
            desc=message,
            disable=not self._debug,
            unit='pages'
        )
        try:
            yield progress
        finally:
            progress.close()

    def _validate_text(self, text: str, min_length: int = 50) -> bool:
        """Validate extracted text quality"""
        if not text or len(text.strip()) < min_length:
            return False
            
        # Check for garbage content
        garbage_ratio = sum(1 for c in text if not c.isprintable()) / len(text)
        if garbage_ratio > 0.1:
            return False
            
        # Check line lengths
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return False
            
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        if avg_line_length < 20:
            return False
            
        return True

    def _recover_from_error(self, error: Exception, context: str = "") -> Optional[str]:
        """Try to recover from extraction errors"""
        error_str = str(error)
        
        if isinstance(error, MemoryError):
            self._clear_memory()
            return "Memory error occurred - cleared memory"
            
        if "PDF file is encrypted" in error_str:
            return "PDF is encrypted - try providing a password"
            
        if "PDF file is damaged" in error_str:
            return "PDF file appears to be damaged"
            
        if "no text extractable" in error_str.lower():
            return "No extractable text found - try OCR"
            
        if "not enough memory" in error_str.lower():
            self._clear_memory()
            return "Memory allocation failed - try reducing batch size"
            
        if "permission error" in error_str.lower():
            return "Permission denied - check file access"
            
        if "timeout" in error_str.lower():
            return "Operation timed out - try again"
            
        return None

    def _clear_memory(self):
        """Clear memory and cached data"""
        import gc
        gc.collect()
        
        # Clear GPU memory if available
        if self._check_gpu_available():
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except:
                    try:
                        import paddle
                        paddle.device.cuda.empty_cache()
                    except:
                        pass

    def _check_gpu_available(self) -> bool:
        """Enhanced GPU availability check"""
        try:
            import torch
            if torch.cuda.is_available():
                # Check if CUDA initialization works
                try:
                    torch.cuda.init()
                    device = torch.cuda.current_device()
                    capability = torch.cuda.get_device_capability(device)
                    logging.info(f"CUDA device available: {torch.cuda.get_device_name(device)} "
                            f"(Compute {capability[0]}.{capability[1]})")
                    return True
                except Exception as e:
                    logging.warning(f"CUDA available but initialization failed: {e}")
                    return False
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logging.info("MPS (Metal Performance Shaders) device available")
                return True
                
            logging.warning("No GPU detected - using CPU only")
            return False
        except ImportError:
            logging.warning("PyTorch not available - using CPU only")
            return False

class EPUBExtractor:
    """EPUB text extraction with multiple fallback methods"""
    
    def __init__(self, import_cache: ImportCache, debug: bool = False):
        self._import_cache = import_cache
        self._debug = debug
        self._checked_methods = {}
        self._available_methods = None

    @property
    def available_methods(self) -> Dict[str, bool]:
        """Lazy load available methods"""
        if self._available_methods is None:
            self._available_methods = {
                'ebooklib': self._import_cache.is_available('ebooklib'),
                'bs4': self._import_cache.is_available('bs4'),
                'html2text': self._import_cache.is_available('html2text'),
                'zipfile': True  # Basic fallback always available
            }
        return self._available_methods

    def extract_text(self, epub_path: str, preferred_method: Optional[str] = None,
                    progress_callback: Optional[Callable] = None) -> str:
        """
        Extract text with fallback methods and progress reporting
        
        Args:
            epub_path: Path to EPUB file
            preferred_method: Optional preferred extraction method
            progress_callback: Optional callback for progress updates
            
        Returns:
            Extracted text
        """
        methods = ['ebooklib', 'bs4', 'zipfile']
        if preferred_method:
            if preferred_method not in methods:
                raise ValueError(f"Invalid method: {preferred_method}")
            methods.insert(0, methods.pop(methods.index(preferred_method)))

        text = ""
        with tqdm(total=len(methods), desc="Trying extraction methods", unit="method") as method_pbar:
            for method in methods:
                if not self.available_methods.get(method, False):
                    method_pbar.update(1)
                    continue
                    
                try:
                    if progress_callback:
                        progress_callback(0, method)  # Signal start with method name
                    
                    extraction_func = getattr(self, f'extract_with_{method}')
                    text = extraction_func(
                        epub_path,
                        lambda n: progress_callback(n, method) if progress_callback else None
                    )
                    
                    if text and text.strip():
                        method_pbar.update(1)
                        break
                        
                except Exception as e:
                    logging.debug(f"Error with {method}: {e}")
                    
                method_pbar.update(1)

        return text.strip()

    def extract_with_ebooklib(self, epub_path: str, progress_callback=None) -> str:
        """Extract using ebooklib with BeautifulSoup parsing and progress bars"""
        ebooklib = self._import_cache.import_module('ebooklib')
        BeautifulSoup = self._import_cache.import_module('bs4').BeautifulSoup
        
        text_parts = []
        book = None
        
        try:
            with tqdm(desc="Loading EPUB", unit="file") as pbar:
                book = ebooklib.epub.read_epub(epub_path)
                pbar.update(1)
            
            items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
            
            with tqdm(total=len(items), desc="Extracting content", unit="item") as pbar:
                for i, item in enumerate(items):
                    try:
                        content = item.get_content().decode('utf-8')
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Remove unwanted elements
                        for tag in soup(['script', 'style', 'nav']):
                            tag.decompose()
                        
                        # Extract text with layout preservation
                        text = self._process_html_content(soup)
                        if text.strip():
                            text_parts.append(text.strip())
                        
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(1)
                            
                    except Exception as e:
                        logging.debug(f"Item extraction failed: {e}")
                        continue
                    
        finally:
            book = None  # Release memory
            
        return '\n\n'.join(text_parts)

    def _process_html_content(self, soup) -> str:
        """Process HTML content with layout preservation"""
        text_parts = []
        
        # Process headings with progress
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        with tqdm(total=len(headings), desc="Processing headings", unit="heading", leave=False) as pbar:
            for tag in headings:
                text = tag.get_text(strip=True)
                if text:
                    text_parts.append(f"\n{text}\n")
                pbar.update(1)
        
        # Process paragraphs and other block elements with progress
        blocks = soup.find_all(['p', 'div', 'section'])
        with tqdm(total=len(blocks), desc="Processing blocks", unit="block", leave=False) as pbar:
            for tag in blocks:
                text = tag.get_text(strip=True)
                if text:
                    text_parts.append(text)
                pbar.update(1)
        
        return '\n\n'.join(text_parts)

    def extract_with_bs4(self, epub_path: str, progress_callback=None) -> str:
        """Extract using BeautifulSoup with zipfile and progress bars"""
        BeautifulSoup = self._import_cache.import_module('bs4').BeautifulSoup
        html2text = self._import_cache.import_module('html2text').HTML2Text()
        zipfile = self._import_cache.import_module('zipfile')
        
        text_parts = []
        
        try:
            with zipfile.ZipFile(epub_path) as zf:
                # Get HTML files
                html_files = [f for f in zf.namelist() 
                            if f.endswith(('.html', '.xhtml', '.htm'))]
                
                with tqdm(total=len(html_files), desc="Processing HTML files", unit="file") as pbar:
                    for i, html_file in enumerate(html_files):
                        try:
                            content = zf.read(html_file).decode('utf-8')
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Remove unwanted elements
                            for tag in soup(['script', 'style', 'nav']):
                                tag.decompose()
                            
                            # Convert to markdown-style text
                            html2text.ignore_links = True
                            html2text.ignore_images = True
                            text = html2text.handle(str(soup))
                            
                            if text.strip():
                                text_parts.append(text.strip())
                            
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                                
                        except Exception as e:
                            logging.debug(f"File extraction failed: {e}")
                            continue
                            
        except Exception as e:
            logging.error(f"EPUB extraction failed: {e}")
            
        return '\n\n'.join(text_parts)

    def extract_with_zipfile(self, epub_path: str, progress_callback=None) -> str:
        """Basic fallback extraction using zipfile with progress bars"""
        zipfile = self._import_cache.import_module('zipfile')
        import re
        
        text_parts = []
        html_pattern = re.compile(r'<[^>]+>')
        
        try:
            with zipfile.ZipFile(epub_path) as zf:
                html_files = [f for f in zf.namelist() 
                            if f.endswith(('.html', '.xhtml', '.htm'))]
                
                with tqdm(total=len(html_files), desc="Extracting text", unit="file") as pbar:
                    for i, html_file in enumerate(html_files):
                        try:
                            content = zf.read(html_file).decode('utf-8')
                            
                            # Basic HTML cleaning
                            content = re.sub(r'<script.*?</script>', '', content, 
                                           flags=re.DOTALL)
                            content = re.sub(r'<style.*?</style>', '', content, 
                                           flags=re.DOTALL)
                            content = html_pattern.sub(' ', content)
                            
                            # Clean up whitespace
                            content = re.sub(r'\s+', ' ', content).strip()
                            
                            if content:
                                text_parts.append(content)
                            
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                                
                        except Exception as e:
                            logging.debug(f"File extraction failed: {e}")
                            continue
                            
        except Exception as e:
            logging.error(f"EPUB extraction failed: {e}")
            
        return '\n\n'.join(text_parts)

class TableExtractor:
    """PDF table extraction using Camelot"""
    
    def __init__(self, import_cache: ImportCache):
        self._import_cache = import_cache
        self._camelot = None
        
    def extract_tables(self, pdf_path: str) -> List[Any]:
        """Extract tables using multiple methods"""
        if not self._init_camelot():
            return []
            
        tables = []
        methods = [
            ('lattice', {'line_scale': 40}),
            ('stream', {'edge_tol': 500})
        ]
        
        for method, params in methods:
            try:
                current_tables = self._camelot.read_pdf(
                    pdf_path,
                    flavor=method,
                    pages='all',
                    **params
                )
                
                if len(current_tables) > 0:
                    tables.extend(current_tables)
                    
            except Exception as e:
                logging.debug(f"Table extraction failed with {method}: {e}")
                continue
                
        return tables
        
    def _init_camelot(self) -> bool:
        """Initialize Camelot library only when needed."""
        if self._camelot is None:
            try:
                # Lazy import for Camelot (which may in turn import pydot)
                self._camelot = self._import_cache.import_module('camelot')
                return True
            except Exception as e:
                logging.error(f"Failed to initialize Camelot: {e}")
                return False
        return True

    
class PDFExtractor:
    """Enhanced PDF text extraction with lazy loading and multiple fallback methods"""

    TEXT_METHODS = [
        'pymupdf',      # Fast native PDF parsing
        'pdfplumber',   # Good balance of speed and accuracy
        'pypdf',        # Simple but reliable
        'pdfminer',     # Good layout preservation
        'tesseract',    # OCR support
        'doctr',        # Deep learning OCR
        'easyocr',      # Alternative OCR
        'kraken'        # Advanced OCR
    ]
    TABLE_METHODS = ['camelot']
    
    def _setup_windows_paths(self):
        """Add binary paths to system PATH for Windows"""
        if platform.system() == 'Windows':
            # Define paths to check with most specific paths first
            paths_to_check = [
                # Ghostscript - specific versions
                r'C:\Program Files\gs\gs10.04.0\bin',
                r'C:\Program Files\gs\gs10.02.0\bin',
                r'C:\Program Files (x86)\gs\gs10.04.0\bin',
                r'C:\Program Files (x86)\gs\gs10.02.0\bin',
                # Poppler - specific versions
                r'C:\Users\stc\Downloads\code\poppler-24.08.0\Library\bin',
                r'C:\Program Files\poppler-24.02.0\Library\bin',
                # Tesseract
                r'C:\Program Files\Tesseract-OCR',
                r'C:\Program Files (x86)\Tesseract-OCR',
            ]
            
            # Add each existing path to PATH
            paths_added = []
            for path in paths_to_check:
                if os.path.exists(path) and path not in os.environ['PATH']:
                    os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
                    paths_added.append(path)
            
            if paths_added and self._debug:
                logging.debug(f"Added to PATH: {', '.join(paths_added)}")
    
    def _safe_import(self, module_name):
        """Safely import a module with error handling"""
        try:
            # Use importlib directly instead of relying on a global import
            import importlib
            
            # Split module_name to handle submodules (e.g., 'PIL.Image')
            parts = module_name.split('.')
            if len(parts) > 1:
                # For submodules, import the base and then get the attribute
                base_module = importlib.import_module(parts[0])
                current = base_module
                
                # Navigate through the module hierarchy
                for part in parts[1:]:
                    current = getattr(current, part)
                
                return current
            else:
                # Direct import for simple module names
                return importlib.import_module(module_name)
        except ImportError as e:
            if self._debug:
                logging.debug(f"Cannot import {module_name}: {e}")
            return None
        except Exception as e:
            if self._debug:
                logging.debug(f"Error importing {module_name}: {e}")
            return None
    
    @property
    def languages(self):
        """Return list of OCR languages"""
        return ['eng']  # Add more languages as needed
    
    def __init__(self, debug=False):
        """Initialize PDF extractor"""
        self._debug = debug
        self._import_cache = ImportCache()
        self._initialized_methods = set()
        self._password = None
        self._current_doc = None
        self._ocr_initialized = {}
        self._available_methods = None
        self._ocr_failed_methods = set()
        
        # Setup Windows paths first
        self._setup_windows_paths()
        
        # Check system dependencies
        self._binaries = self._check_system_dependencies()
        
        # Check core dependencies first to prioritize stable methods
        self._check_core_dependencies()
        
        # Only check OCR dependencies if debug mode is on or we have binaries
        if debug or self._binaries.get('tesseract', False):
            self._check_ocr_dependencies()
        
        if self._debug:
            available = sorted(list(self._initialized_methods))
            logging.debug(f"Available extraction methods: {', '.join(available)}")
    
    def _check_system_dependencies(self) -> Dict[str, bool]:
        """Check system dependencies with proper binary detection"""
        binaries = {
            'tesseract': False,
            'poppler': False,
            'ghostscript': False
        }
        
        if platform.system() == 'Windows':
            # Check Tesseract (simple executable)
            tesseract_path = shutil.which('tesseract')
            
            # Check Poppler (multiple possible executables)
            pdftoppm_path = None
            for exe in ['pdftoppm', 'pdftoppm.exe']:
                path = shutil.which(exe)
                if path:
                    pdftoppm_path = path
                    break
            
            # Check Ghostscript (multiple possible executables)
            gs_path = None
            for exe in ['gs', 'gswin64c', 'gswin64c.exe', 'gswin32c.exe']:
                path = shutil.which(exe)
                if path:
                    gs_path = path
                    break
            
            # Update binary status
            binaries['tesseract'] = bool(tesseract_path)
            binaries['poppler'] = bool(pdftoppm_path)
            binaries['ghostscript'] = bool(gs_path)
            
            # Log discovered binaries
            if self._debug:
                found_binaries = []
                if tesseract_path:
                    found_binaries.append(f"Tesseract: {tesseract_path}")
                if pdftoppm_path:
                    found_binaries.append(f"Poppler: {pdftoppm_path}")
                if gs_path:
                    found_binaries.append(f"Ghostscript: {gs_path}")
                
                if found_binaries:
                    logging.debug(f"Found binaries: {'; '.join(found_binaries)}")
                
            # Only show warnings for missing binaries once
            if not tesseract_path and 'tesseract' not in self._initialized_methods:
                logging.info("Tesseract not found. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
                
            if not pdftoppm_path and 'poppler' not in self._initialized_methods:
                logging.info("Poppler not found. Download from: https://github.com/oschwartz10612/poppler-windows/releases/")
                
            if not gs_path and 'ghostscript' not in self._initialized_methods:
                logging.info("Ghostscript not found. Download from: https://ghostscript.com/releases/gsdnld.html")
        
        return binaries
    
    def _check_core_dependencies(self):
        """Check core text extraction dependencies"""
        # These are the most reliable methods, so check them first
        try:
            import fitz  # pymupdf
            self._initialized_methods.add('pymupdf')
            if self._debug:
                logging.debug("PyMuPDF (fitz) available")
        except ImportError:
            pass

        try:
            import pdfplumber
            self._initialized_methods.add('pdfplumber')
            if self._debug:
                logging.debug("pdfplumber available")
        except ImportError:
            pass

        try:
            import pypdf
            self._initialized_methods.add('pypdf')
            if self._debug:
                logging.debug("pypdf available")
        except ImportError:
            pass

        try:
            from pdfminer import high_level
            self._initialized_methods.add('pdfminer')
            if self._debug:
                logging.debug("pdfminer available")
        except ImportError:
            pass
    
    def _check_ocr_dependencies(self):
        """Check OCR-related dependencies separately"""
        # OCR-related checks - these are more likely to cause issues
        try:
            import pytesseract
            import pdf2image
            
            # Verify tesseract works by checking the version
            try:
                version = pytesseract.get_tesseract_version()
                self._pytesseract = pytesseract
                self._pdf2image = pdf2image
                self._initialized_methods.add('tesseract')
                if self._debug:
                    logging.debug(f"Found Tesseract version: {version}")
            except Exception as e:
                if self._debug:
                    logging.debug(f"Tesseract check failed: {e}")
        except ImportError:
            if self._debug:
                logging.debug("pytesseract or pdf2image not available")

        # Skip problematic OCR dependencies in normal operation
        # They'll be checked when actually needed
        if self._debug:
            try:
                self._safe_import('doctr')
                self._initialized_methods.add('doctr')
                logging.debug("doctr available")
            except Exception:
                pass

            try:
                self._safe_import('easyocr')
                self._initialized_methods.add('easyocr')
                logging.debug("easyocr available")
            except Exception:
                pass

            try:
                self._safe_import('kraken')
                self._initialized_methods.add('kraken')
                logging.debug("kraken available")
            except Exception:
                pass
    
    def _is_method_available(self, method: str) -> bool:
        """Check if extraction method is available"""
        # For core methods, use cached results
        if method in self._initialized_methods:
            return True
            
        # For OCR methods, check on demand if not already checked
        if method in ['tesseract', 'doctr', 'easyocr', 'kraken'] and method not in self._initialized_methods:
            try:
                if method == 'tesseract':
                    # Only check if tesseract binary exists
                    if not self._binaries.get('tesseract', False):
                        return False
                    
                    # Import required packages
                    import pytesseract
                    import pdf2image
                    
                    if pytesseract and pdf2image:
                        try:
                            pytesseract.get_tesseract_version()
                            self._initialized_methods.add('tesseract')
                            return True
                        except Exception:
                            return False
                    return False
                    
                elif method == 'doctr':
                    # Skip doctr - it's causing numpy compatibility issues
                    return False
                    
                elif method == 'easyocr':
                    # Skip easyocr - it's causing numpy compatibility issues
                    return False
                    
                elif method == 'kraken':
                    # Skip kraken - it's causing compatibility issues
                    return False
                    
            except Exception as e:
                if self._debug:
                    logging.debug(f"Error checking {method}: {e}")
                return False
                
        return False
        
    def set_password(self, password: str):
        """Set password for encrypted PDFs"""
        self._password = password

    @property
    def available_methods(self) -> Dict[str, bool]:
        """Lazy load available methods"""
        if self._available_methods is None:
            self._available_methods = {
                method: self._is_method_available(method)
                for method in self.TEXT_METHODS + self.TABLE_METHODS
            }
        return self._available_methods

    def _is_method_available_old(self, method: str) -> bool:
        """Check method availability with dependencies"""
        if method not in self._checked_methods:
            try:
                if method == 'pymupdf':
                    self._checked_methods[method] = self._import_cache.is_available('fitz')
                elif method == 'pdfplumber':
                    self._checked_methods[method] = self._import_cache.is_available('pdfplumber')
                elif method == 'pypdf':
                    self._checked_methods[method] = self._import_cache.is_available('pypdf')
                elif method == 'pdfminer':
                    self._checked_methods[method] = self._import_cache.is_available('pdfminer.high_level')
                elif method == 'tesseract':
                    self._checked_methods[method] = (
                        self._import_cache.is_available('pytesseract') and
                        self._import_cache.is_available('pdf2image') and
                        self._binaries.get('tesseract', False)
                    )
                elif method == 'kraken':
                    self._checked_methods[method] = self._import_cache.is_available('kraken')
                elif method == 'easyocr':
                    self._checked_methods[method] = self._import_cache.is_available('easyocr')
                elif method == 'doctr':
                    self._checked_methods[method] = self._import_cache.is_available('doctr')
                elif method == 'camelot':
                    self._checked_methods[method] = (
                        self._import_cache.is_available('camelot') and
                        self._binaries.get('gs', False)
                    )
            except Exception as e:
                logging.debug(f"Error checking {method}: {e}")
                self._checked_methods[method] = False
                
        return self._checked_methods[method]

    def extract_text(self, pdf_path: str, preferred_method: Optional[str] = None,
                progress_callback: Optional[Callable] = None, **kwargs) -> str:
        """Extract text with optimized fallback methods"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
                
        # Start with fast methods first
        methods = ['pymupdf', 'pdfplumber', 'pypdf', 'pdfminer']  # Fast and reliable methods
        
        # Define OCR methods
        ocr_methods = ['tesseract', 'doctr', 'easyocr', 'kraken']
        
        # If a specific preferred method is provided and it's an OCR method,
        # move it to the front of the OCR methods list
        if preferred_method and preferred_method in ocr_methods:
            # Remove it from its current position if present
            if preferred_method in ocr_methods:
                ocr_methods.remove(preferred_method)
            # Add it to the front of OCR methods
            ocr_methods.insert(0, preferred_method)
            if self._debug:
                logging.info(f"Prioritizing OCR method: {preferred_method}")
        
        text_parts = []
        current_method = None

        try:
            # Try fast methods first
            for method in methods:
                if method not in self._initialized_methods:
                    continue
                    
                try:
                    current_method = method
                    if self._debug:
                        logging.info(f"Trying extraction with {method}...")
                    
                    if progress_callback:
                        progress_callback(0, method)
                    
                    extraction_func = getattr(self, f'extract_with_{method}')
                    
                    # Extract text
                    try:
                        text = extraction_func(
                            pdf_path,
                            lambda n: progress_callback(n, method) if progress_callback else None
                        )
                    except KeyboardInterrupt:
                        print("\nExtraction interrupted by user.")
                        raise
                    
                    if text and text.strip():
                        text_parts.append(text.strip())
                        quality = self._assess_text_quality(text)
                        if self._debug:
                            logging.info(f"Text quality with {method}: {quality:.2f}")
                        if quality > 0.7:  # Good enough quality
                            if self._debug:
                                logging.info(f"Got good quality text from {method}, stopping here")
                            break
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    if self._debug:
                        logging.error(f"Error with {method}: {e}")
                    continue
                finally:
                    if progress_callback and current_method == method:
                        progress_callback(1, None)

            # Only try OCR as a last resort
            if not text_parts and self._might_need_ocr(pdf_path):
                if self._debug:
                    logging.info("Initial extraction insufficient, trying OCR methods...")
                
                for method in ocr_methods:
                    if method not in self._initialized_methods and not self._is_method_available(method):
                        continue
                        
                    try:
                        current_method = method
                        if self._debug:
                            logging.info(f"Trying OCR with {method}...")
                        
                        if progress_callback:
                            progress_callback(0, method)
                            
                        # Initialize OCR method
                        if not self._init_ocr(method):
                            if self._debug:
                                logging.debug(f"Skipping {method} - initialization failed")
                            continue
                            
                        extraction_func = getattr(self, f'extract_with_{method}')
                        
                        # Extract text
                        try:
                            text = extraction_func(
                                pdf_path,
                                lambda n: progress_callback(n, method) if progress_callback else None
                            )
                        except KeyboardInterrupt:
                            print("\nOCR processing interrupted by user.")
                            raise
                            
                        if text and text.strip():
                            text_parts.append(text.strip())
                            if self._debug:
                                logging.info(f"Got text from {method}")
                            break  # One successful OCR method is enough
                            
                    except KeyboardInterrupt:
                        logging.info("Extraction process interrupted.")
                        raise
                    except Exception as e:
                        if self._debug:
                            logging.error(f"Error with {method}: {e}")
                        self._ocr_failed_methods.add(method)
                        continue
                    finally:
                        if progress_callback and current_method == method:
                            progress_callback(1, None)

        finally:
            self._cleanup()

        return "\n\n".join(text_parts).strip()

    def _might_need_ocr(self, pdf_path: str) -> bool:
        """Quick check if PDF might need OCR"""
        try:
            if 'pymupdf' in self._initialized_methods:
                import fitz
                doc = fitz.open(pdf_path)
                try:
                    # Check first 3 pages or all pages if less
                    pages_to_check = min(3, len(doc))
                    total_text = 0
                    
                    for i in range(pages_to_check):
                        text = doc[i].get_text()
                        total_text += len(text.strip())
                        
                    # If average text per page is very low, might need OCR
                    return (total_text / pages_to_check) < 100
                    
                finally:
                    doc.close()
        except Exception:
            pass
            
        return True  # Default to yes if we can't check

    def _assess_text_quality(self, text: str) -> float:
        """Assess extracted text quality"""
        if not text:
            return 0.0
            
        score = 0.0
        text = text.strip()
        
        # Basic text characteristics
        words = text.split()
        if not words:
            return 0.0
            
        # Check word lengths
        avg_word_len = sum(len(w) for w in words) / len(words)
        if 3 <= avg_word_len <= 10:
            score += 0.3
            
        # Check for reasonable text structure
        lines = text.split('\n')
        if lines:
            # Check line lengths
            avg_line_len = sum(len(l.strip()) for l in lines) / len(lines)
            if 30 <= avg_line_len <= 100:
                score += 0.3
                
        # Check for paragraph structure
        if '\n\n' in text:
            score += 0.2
            
        # Check character distribution
        alpha_count = sum(c.isalpha() for c in text)
        if len(text) > 0:
            alpha_ratio = alpha_count / len(text)
            if 0.6 <= alpha_ratio <= 0.9:
                score += 0.2
                
        return min(max(score, 0.0), 1.0)

    def _is_scanned_pdf(self, pdf_path: str) -> bool:
        """Quick check if PDF appears to be scanned"""
        try:
            # Try quick text extraction with pymupdf
            import fitz
            doc = fitz.open(pdf_path)
            first_page = doc[0]
            text = first_page.get_text()
            doc.close()
            
            # If first page has very little text, likely scanned
            return len(text.strip()) < 100
            
        except Exception:
            return False

    def _needs_further_processing(self, text: str) -> bool:
        """Check if text needs additional processing methods"""
        # Check text quality
        words = text.split()
        if len(words) < 100:  # Too short - try other methods
            return True
            
        # Check for common OCR/extraction artifacts
        artifacts = ['�', '□', '■', '○', '●', '¶']
        artifact_count = sum(text.count(a) for a in artifacts)
        if artifact_count > len(text) * 0.01:  # More than 1% artifacts
            return True
            
        # Check for layout issues
        lines = text.splitlines()
        if not lines:
            return True
            
        # Check for suspiciously short lines
        short_lines = sum(1 for line in lines if len(line.strip()) < 20)
        if short_lines > len(lines) * 0.5:  # More than 50% short lines
            return True
            
        # Check for reasonable paragraph structure
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:  # No clear paragraph breaks
            return True
            
        return False

    def _cleanup(self):
        """Clean up resources"""
        if self._current_doc:
            try:
                self._current_doc.close()
            except:
                pass
            self._current_doc = None
            
        import gc
        gc.collect()

    def _clear_memory(self):
        """Clear memory after processing"""
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
    def extract_with_pymupdf(self, pdf_path: str, progress_callback=None) -> str:
        try:
            fitz = self._import_cache.import_module('fitz')  # Use ImportCache
            text_parts = []
            
            doc = fitz.open(pdf_path)
            self._current_doc = doc
            
            if doc.needs_pass:
                if not self._password or not doc.authenticate(self._password):
                    raise ValueError("Invalid PDF password")
            
            total_pages = len(doc)
            with tqdm(total=total_pages, desc="PyMuPDF extraction", unit="pages") as pbar:
                for page_num in range(total_pages):
                    try:
                        page = doc[page_num]
                        # Try different extraction strategies
                        page_text = page.get_text("text", sort=True)
                        if not page_text.strip():
                            # Fallback to dict extraction for complex layouts
                            page_text = page.get_text("dict")
                            if isinstance(page_text, dict):
                                page_text = self._process_text_dict(page_text)
                        
                        if page_text.strip():
                            text_parts.append(page_text.strip())
                        
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(1)
                            
                    except Exception as e:
                        logging.debug(f"Page {page_num + 1} extraction failed: {e}")
                        continue
                    finally:
                        page = None
            
            return "\n\n".join(text_parts)
            
        finally:
            if self._current_doc:
                try:
                    self._current_doc.close()
                except:
                    pass
                self._current_doc = None


    def _process_text_dict(self, text_dict: Dict) -> str:
        """Process PyMuPDF dict format text"""
        text_parts = []
        try:
            for block in text_dict.get('blocks', []):
                if 'lines' in block:
                    for line in block['lines']:
                        line_text = ' '.join(
                            span.get('text', '') 
                            for span in line.get('spans', [])
                        )
                        if line_text.strip():
                            text_parts.append(line_text)
        except Exception as e:
            logging.debug(f"Text dict processing failed: {e}")
        return '\n'.join(text_parts)

    def extract_with_pdfplumber(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using pdfplumber with layout preservation and progress bar"""
        pdfplumber = self._import_cache.import_module('pdfplumber')
        text_parts = []
        
        try:
            with pdfplumber.open(pdf_path, password=self._password) as pdf:
                self._current_doc = pdf
                total_pages = len(pdf.pages)
                
                with tqdm(total=total_pages, desc="pdfplumber extraction", unit="pages") as pbar:
                    for page in pdf.pages:
                        try:
                            # Extract with layout settings
                            words = page.extract_words(
                                keep_blank_chars=True,
                                use_text_flow=True,
                                horizontal_ltr=True
                            )
                            
                            if words:
                                # Group words into lines
                                lines = self._group_words_into_lines(words)
                                text_parts.append('\n'.join(' '.join(line) for line in lines))
                            else:
                                # Fallback to basic extraction
                                text = page.extract_text()
                                if text.strip():
                                    text_parts.append(text.strip())
                            
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                                
                        except Exception as e:
                            logging.debug(f"Page extraction failed: {e}")
                            continue
            
            return '\n\n'.join(text_parts)
            
        finally:
            self._current_doc = None
            
    def _group_words_into_lines(self, words: List[Dict]) -> List[List[str]]:
        """Group words into lines based on positions"""
        if not words:
            return []
        
        # Sort by top position and x position
        words.sort(key=lambda w: (round(w['top']), w['x0']))
        
        lines = []
        current_line = []
        last_top = round(words[0]['top'])
        
        for word in words:
            current_top = round(word['top'])
            if abs(current_top - last_top) > 3:  # Line break threshold
                if current_line:
                    lines.append(current_line)
                current_line = []
                last_top = current_top
            current_line.append(word['text'])
        
        if current_line:
            lines.append(current_line)
        
        return lines

    def extract_with_pypdf(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using pypdf with encryption support and progress bar"""
        pypdf = self._import_cache.import_module('pypdf')
        
        # Disable debug logging from pypdf
        logging.getLogger('pypdf').setLevel(logging.WARNING)
        
        text_parts = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                self._current_doc = reader
                
                if reader.is_encrypted:
                    if not reader.decrypt(self._password or ""):
                        raise ValueError("PDF is encrypted and requires a valid password")
                
                total_pages = len(reader.pages)
                with tqdm(total=total_pages, desc="pypdf extraction", unit="pages") as pbar:
                    for page in reader.pages:
                        try:
                            text = page.extract_text()
                            if text.strip():
                                text_parts.append(text.strip())
                            
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                                
                        except Exception as e:
                            logging.debug(f"Page extraction failed: {e}")
                            continue
            
            return '\n\n'.join(text_parts)
            
        finally:
            self._current_doc = None

    def extract_with_pdfminer(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using pdfminer with layout analysis"""
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        from io import StringIO
        
        try:
            output = StringIO()
            with open(pdf_path, 'rb') as file:
                # Configure layout parameters
                laparams = LAParams(
                    line_margin=0.5,
                    word_margin=0.1,
                    char_margin=2.0,
                    boxes_flow=0.5,
                    detect_vertical=True
                )
                
                extract_text_to_fp(
                    file, output,
                    laparams=laparams,
                    password=self._password,
                    codec='utf-8'
                )
                
                text = output.getvalue()
                
                # Post-process the extracted text
                if text.strip():
                    lines = text.splitlines()
                    processed_lines = []
                    current_para = []
                    
                    for line in lines:
                        line = line.strip()
                        if not line and current_para:
                            processed_lines.append(' '.join(current_para))
                            current_para = []
                        elif line:
                            current_para.append(line)
                    
                    if current_para:
                        processed_lines.append(' '.join(current_para))
                    
                    return '\n\n'.join(processed_lines)
            
            return ""
            
        finally:
            if 'output' in locals():
                output.close()
    
    def extract_with_doctr(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using docTR with proper API handling."""
        if not self._init_ocr('doctr'):
            return ""
            
        try:
            # Use proper docTR imports and API
            doctr = self._doctr
            
            # Load document
            with tqdm(desc="Loading document", unit="file") as pbar:
                try:
                    # Load PDF document
                    doc = doctr.io.DocumentFile.from_pdf(pdf_path)
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"DocTR document loading failed: {e}")
                    return ""
            
            # Initialize predictor if needed
            if not hasattr(self, '_doctr_predictor') or self._doctr_predictor is None:
                try:
                    # Initialize OCR predictor with default pretrained model
                    self._doctr_predictor = doctr.models.ocr_predictor(pretrained=True)
                    logging.info("DocTR OCR predictor initialized")
                except Exception as e:
                    logging.error(f"Failed to initialize DocTR predictor: {e}")
                    return ""
            
            # Process document with predictor
            try:
                # Use the predictor on the whole document at once
                result = self._doctr_predictor(doc)
                
                # Extract text from result with proper error handling
                text_parts = []
                
                # Fix: Safely access text content
                try:
                    for page_idx, page in enumerate(result.pages):
                        # Safe access to text property
                        try:
                            page_export = page.export()
                            if isinstance(page_export, dict) and "text" in page_export:
                                page_text = page_export["text"]
                                if page_text:
                                    text_parts.append(page_text)
                            else:
                                logging.warning(f"DocTR: Missing text in page {page_idx} export")
                        except Exception as page_e:
                            logging.warning(f"DocTR: Error exporting page {page_idx}: {page_e}")
                            continue
                except Exception as e:
                    logging.error(f"DocTR: Error processing result pages: {e}")
                    return ""
                    
                if progress_callback:
                    # Signal completion
                    progress_callback(len(doc))
                    
                return "\n\n".join(text_parts)
                
            except Exception as e:
                logging.error(f"DocTR prediction failed: {e}")
                return ""
                
        except Exception as e:
            logging.error(f"DocTR extraction failed: {e}")
            return ""
        finally:
            self._clear_gpu_memory()



    def extract_with_kraken(self, pdf_path: str, progress_callback=None) -> str:
        """
        Extract text using Kraken with correct API usage.
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback for progress updates
            
        Returns:
            str: Extracted text
        """
        if not self._init_ocr('kraken'):
            return ""
        
        # Import minimal dependencies
        pdf2image = self._import_cache.import_module('pdf2image') 
        
        text_parts = []
        images = None
            
        try:
            # Convert PDF to images (this is thread-safe)
            with tqdm(desc="Converting PDF to images for Kraken", unit="page") as pbar:
                images = pdf2image.convert_from_path(
                    pdf_path,
                    dpi=300,
                    thread_count=1,  # Use single thread for conversion
                    grayscale=True
                )
                pbar.update(len(images))
            
            # Process images with safer approach
            with tqdm(total=len(images), desc="Kraken processing", unit="page") as pbar:
                # Import kraken correctly
                import kraken
                
                for i, image in enumerate(images, 1):
                    try:
                        # Use correct Kraken API modules based on documentation
                        # First binarize the image
                        binarized = kraken.binarization.nlbin(image)
                        
                        # Segment the image
                        segments = kraken.pageseg.segment(binarized)
                        
                        # Perform text recognition if available
                        try:
                            # Use the rpred module directly as shown in docs
                            if hasattr(kraken, 'rpred'):
                                # Simple model-less recognition for testing
                                text = kraken.rpred.rpred(None, image, segments)
                                if text:
                                    text_parts.append(text)
                                else:
                                    text_parts.append(f"[Kraken extracted {len(segments['boxes'])} text segments]")
                        except Exception as recog_error:
                            if self._debug:
                                logging.debug(f"Kraken recognition error: {recog_error}")
                            # Fall back to simple text extraction from segments
                            text_parts.append(f"[Kraken extracted {len(segments['boxes'])} text segments]")
                        
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(1)
                            
                    except Exception as e:
                        logging.error(f"Kraken failed on page {i}: {e}")
                        continue
                    finally:
                        image.close()
                        
        except Exception as e:
            logging.error(f"Kraken extraction failed: {e}")
            return ""
        finally:
            if images:
                for img in images:
                    try:
                        img.close()
                    except:
                        pass
            
        return '\n\n'.join(text_parts)


    def extract_with_easyocr(self, pdf_path: str, progress_callback=None) -> str:
        """
        Extract text using EasyOCR with proper API.
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback for progress updates
            
        Returns:
            str: Extracted text
        """
        if not self._init_ocr('easyocr'):
            return ""
        
        try:
            # Import required dependencies here to ensure they're available
            pdf2image = self._import_cache.import_module('pdf2image')
            import torch  # Add explicit torch import here
            
            text_parts = []
            images = None
            
            try:
                # Convert PDF to images (thread-safe)
                with tqdm(desc="Converting PDF to images for EasyOCR", unit="page") as pbar:
                    images = pdf2image.convert_from_path(
                        pdf_path,
                        dpi=300,
                        thread_count=1,  # Single thread for stability
                        grayscale=True
                    )
                    pbar.update(len(images))
                
                # Initialize reader if needed
                easyocr = self._easyocr
                reader = None
                
                # Initialize reader with English as default language
                try:
                    if not hasattr(self, '_reader') or self._reader is None:
                        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                        self._reader = reader
                    else:
                        reader = self._reader
                except Exception as init_error:
                    logging.error(f"EasyOCR reader initialization failed: {init_error}")
                    return ""
                
                # Process pages
                with tqdm(total=len(images), desc="EasyOCR processing", unit="page") as pbar:
                    for i, image in enumerate(images, 1):
                        try:
                            # Process page using EasyOCR
                            results = reader.readtext(
                                image,
                                detail=0,  # Just get the text
                                paragraph=True  # Combine text into paragraphs
                            )
                            
                            # Add extracted text
                            if results:
                                text_parts.append('\n'.join(results))
                            else:
                                text_parts.append(f"[EasyOCR found no text on page {i}]")
                            
                            # Update progress
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                        except Exception as e:
                            logging.error(f"EasyOCR failed on page {i}: {e}")
                        finally:
                            image.close()
                            
            finally:
                # Clean up resources
                if images:
                    for img in images:
                        try:
                            img.close()
                        except:
                            pass
                
                # Clear GPU memory
                self._clear_gpu_memory()
            
            return '\n\n'.join(text_parts)
        
        except Exception as e:
            logging.error(f"EasyOCR extraction failed: {e}")
            return ""


    def _init_ocr(self, method: str) -> bool:
        """
        Initialize OCR engine with correct dependency checks and API usage.
        
        Args:
            method: OCR method to initialize
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if method not in self._ocr_initialized:
            # Skip if already determined to be unavailable
            if method in self._ocr_failed_methods:
                return False
            
            # Check if we're in the main thread
            in_main_thread = threading.current_thread() is threading.main_thread()
            
            try:
                if method == 'tesseract':
                    # Check basic dependencies for tesseract
                    try:
                        # Check binary availability first
                        if not shutil.which('tesseract'):
                            if self._debug:
                                logging.debug("Tesseract binary not found in PATH")
                            self._ocr_initialized[method] = False
                            return False
                        
                        # Import dependencies
                        import pytesseract
                        import pdf2image
                        
                        # Store the imported modules for later use
                        self._pytesseract = pytesseract
                        self._pdf2image = pdf2image
                        
                        # Verify Tesseract installation
                        try:
                            version = pytesseract.get_tesseract_version()
                            if self._debug:
                                logging.debug(f"Found Tesseract version: {version}")
                            self._ocr_initialized[method] = True
                            return True
                        except Exception as e:
                            if self._debug:
                                logging.debug(f"Tesseract verification failed: {e}")
                            self._ocr_initialized[method] = False
                            return False
                    except Exception as e:
                        if self._debug:
                            logging.debug(f"Tesseract initialization error: {e}")
                        self._ocr_initialized[method] = False
                        return False
                        
                elif method == 'doctr':
                    try:
                        # Check if doctr is importable without initializing models
                        if not self._import_cache.is_available('doctr'):
                            if self._debug:
                                logging.debug("DocTR not available")
                            self._ocr_initialized[method] = False
                            return False
                        
                        # Import doctr without initializing models
                        import doctr
                        
                        # Check if doctr has the required modules
                        if not hasattr(doctr, 'io') or not hasattr(doctr, 'models'):
                            if self._debug:
                                logging.debug("DocTR missing required modules")
                            self._ocr_initialized[method] = False
                            return False
                        
                        # Store module references
                        self._doctr = doctr
                        
                        # Mark as initialized with deferred model loading
                        self._ocr_initialized[method] = True
                        self._doctr_predictor_loaded = False
                        return True
                            
                    except ImportError as e:
                        if self._debug:
                            logging.debug(f"DocTR import error: {e}")
                        self._ocr_initialized[method] = False
                        return False
                        
                elif method == 'kraken':
                    # Check for kraken availability
                    if not self._import_cache.is_available('kraken'):
                        if self._debug:
                            logging.debug("Kraken not available")
                        self._ocr_initialized[method] = False
                        return False
                    
                    try:
                        # Import kraken correctly based on documentation
                        import kraken
                        
                        # Check for the necessary modules according to docs
                        has_binarization = hasattr(kraken, 'binarization')
                        has_pageseg = hasattr(kraken, 'pageseg')
                        has_rpred = hasattr(kraken, 'rpred')
                        
                        if not (has_binarization and has_pageseg and has_rpred):
                            if self._debug:
                                logging.debug("Missing one or more Kraken modules")
                            self._ocr_initialized[method] = False
                            return False
                        
                        # Store modules for later use
                        self._kraken = kraken
                        self._kraken_binarization = kraken.binarization
                        self._kraken_pageseg = kraken.pageseg
                        self._kraken_rpred = kraken.rpred
                        
                        # Mark as initialized with deferred model loading
                        self._ocr_initialized[method] = True
                        self._kraken_model_loaded = False
                        return True
                            
                    except ImportError as e:
                        if self._debug:
                            logging.debug(f"Kraken import error: {e}")
                        self._ocr_initialized[method] = False
                        return False
                        
                elif method == 'easyocr':
                    if not self._import_cache.is_available('easyocr'):
                        if self._debug:
                            logging.debug("EasyOCR not available")
                        self._ocr_initialized[method] = False
                        return False
                        
                    # Check if torch is available
                    if not self._import_cache.is_available('torch'):
                        if self._debug:
                            logging.debug("PyTorch not available for EasyOCR")
                        self._ocr_initialized[method] = False
                        return False
                    
                    try:
                        # Import easyocr according to docs
                        import easyocr
                        import torch  # Explicitly import torch here
                        
                        # Store module reference
                        self._easyocr = easyocr
                        
                        # Mark as initialized with deferred reader creation
                        self._ocr_initialized[method] = True
                        self._easyocr_reader_loaded = False
                        return True
                            
                    except ImportError as e:
                        if self._debug:
                            logging.debug(f"EasyOCR import error: {e}")
                        self._ocr_initialized[method] = False
                        return False
                        
            except Exception as e:
                # Catch all other errors
                if self._debug:
                    logging.debug(f"Failed to initialize {method}: {e}")
                self._ocr_initialized[method] = False
                self._ocr_failed_methods.add(method)
        
        # Return cached result
        return self._ocr_initialized.get(method, False)


    def extract_with_tesseract(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using Tesseract OCR with progress bars"""
        # First verify tesseract is actually available
        if 'tesseract' not in self._initialized_methods:
            logging.error("Tesseract not available in initialized methods")
            return ""
            
        # Use the stored module references
        pytesseract = self._pytesseract
        pdf2image = self._pdf2image
        
        text_parts = []
        images = None
        
        try:
            # Convert PDF to images with progress bar
            with tqdm(desc="Converting PDF to images for tesseract", unit="page") as pbar:
                try:
                    images = pdf2image.convert_from_path(
                        pdf_path,
                        dpi=300,
                        thread_count=1,  # Use single thread for stability
                        grayscale=True,
                        size=(None, 2000),  # Limit height for memory
                        use_cropbox=True,   # Use cropbox for extraction
                        strict=False        # Continue even with errors
                    )
                    pbar.update(len(images))
                except Exception as e:
                    logging.error(f"PDF to image conversion failed: {e}")
                    # Try alternative conversion options
                    try:
                        logging.info("Trying alternative conversion method...")
                        images = pdf2image.convert_from_path(
                            pdf_path,
                            dpi=150,  # Lower DPI
                            thread_count=1,
                            grayscale=True,
                            first_page=1,
                            last_page=None,
                            strict=False
                        )
                        pbar.update(len(images))
                    except Exception as e2:
                        logging.error(f"Alternative conversion also failed: {e2}")
                        return ""
            
            # Process images with OCR
            with tqdm(total=len(images), desc="OCR Processing with tesseract", unit="page") as pbar:
                for i, image in enumerate(images, 1):
                    try:
                        # OCR with optimized settings
                        text = pytesseract.image_to_string(
                            image,
                            config='--oem 3 --psm 3 -l eng',  # Specify language and PSM mode
                        )
                        
                        if text.strip():
                            text_parts.append(text.strip())
                        
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(1)
                            
                    except Exception as e:
                        logging.error(f"OCR failed on page {i}: {e}")
                        continue
                    finally:
                        if image:
                            try:
                                image.close()
                            except:
                                pass
        except Exception as e:
            logging.error(f"Tesseract OCR extraction failed: {e}")
            return ""
        finally:
            # Clean up resources
            if images:
                for img in images:
                    try:
                        img.close()
                    except:
                        pass
        
        # Return combined text if any was extracted
        if text_parts:
            return '\n\n'.join(text_parts)
        else:
            logging.warning("No text extracted using Tesseract OCR")
            return ""


    def _preprocess_image(self, image) -> 'PIL.Image':
        """Optimize image for OCR"""
        try:
            PIL = self._import_cache.import_module('PIL')

            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = PIL.ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Apply adaptive thresholding
            threshold = self._adaptive_threshold(image)
            image = image.point(lambda x: 255 if x > threshold else 0)
            
            return image
        except Exception as e:
            logging.debug(f"Image preprocessing failed: {e}")
            return image

    def _adaptive_threshold(self, image, window_size=41, constant=2):
        """Calculate adaptive threshold for image"""
        try:
            import numpy as np
            img_array = np.array(image)
            mean = np.mean(img_array)
            std = np.std(img_array)
            return int(mean - constant * std)
        except:
            return 127
            
    def _configure_torch_security(self):
        """Configure PyTorch security settings"""
        try:
            import torch
            
            # Configure secure loading
            torch.backends.cudnn.benchmark = True  # Performance optimization
            torch.set_float32_matmul_precision('medium')  # Balance of speed and precision
            
            # Set security-related configurations
            torch.set_warn_always(False)
            
            # Use safer default tensor type - updated API calls
            torch.set_default_tensor_type(torch.FloatTensor)
            torch.set_default_dtype(torch.float32)
            
            # Configure device
            if torch.cuda.is_available():
                try:
                    torch.cuda.init()
                    device = torch.cuda.current_device()
                    logging.info(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
                except Exception as e:
                    logging.warning(f"CUDA initialization failed: {e}")
                    
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logging.info("Using MPS (Metal Performance Shaders) device")
            else:
                logging.info("Using CPU device")
                
        except Exception as e:
            logging.debug(f"PyTorch security configuration failed: {e}")
            # Continue without PyTorch security settings
            pass

    def _clear_gpu_memory(self):
        """Clear GPU memory after processing"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except:
                pass

    
def setup_logging(verbosity: int = 0):
    """Set up logging configuration."""
    levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    level = levels.get(verbosity, logging.DEBUG)
    
    format_str = '%(message)s' if verbosity == 0 else '%(levelname)s: %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pdf_extraction.log')
        ]
    )

class DocumentProcessor:
    """Main document processing coordinator"""
    
    def __init__(self, debug: bool = False):
        self.manager = ExtractionManager(debug=debug)
        self._debug = debug
        # (Optional) Initialize table extractor once if needed
        self._table_extractor = TableExtractor(ImportCache())
        
    def process_files(self, input_files: List[str], 
                output_dir: Optional[str] = None,
                method: Optional[str] = None,
                ocr_method: Optional[str] = None,
                password: Optional[str] = None,
                extract_tables: bool = False,  
                max_workers: int = None,
                noskip: bool = False,
                sort: bool = False,                
                rename_script_path: str = None,    
                llm_provider = None,  
                temperature: float = 0.7,     
                max_tokens: int = 250,        
                **kwargs) -> Dict[str, Any]:
        """Process multiple files with interrupt handling and optional sorting"""
        results = {}
        failed = []
        skipped = []
        
        # Ensure output directory exists
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine number of workers - fewer workers when sorting to avoid Ollama overload
        if sort:
            # When sorting is enabled, use fewer workers to prevent Ollama API overload
            max_workers = max_workers or min(4, os.cpu_count() or 1)  # Use at most 4 threads for sorting
        else:
            max_workers = max_workers or min(len(input_files), (os.cpu_count() or 1))
        
        # ProcessPoolExecutor doesn't work well with our thread_local OpenAI clients
        # So we stick with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            with tqdm(total=len(input_files), desc="Processing files", unit="file") as pbar:
                for input_file in input_files:
                    if shutdown_flag.is_set():
                        logging.info("Shutdown flag detected. Not submitting more jobs.")
                        break
                        
                    future = executor.submit(
                        self._process_single_file,
                        input_file,
                        output_dir,
                        method,
                        ocr_method,  # Add this parameter in the correct position
                        password,    # Move password to the correct position
                        extract_tables,
                        noskip,
                        sort,                     
                        rename_script_path,       
                        counters if 'counters' in locals() else None,
                        llm_provider,
                        temperature,
                        max_tokens,
                        **kwargs
                    )
                    futures[future] = input_file
                
                for future in as_completed(futures):
                    input_file = futures[future]
                    try:
                        result = future.result()
                        results[input_file] = result
                        if result.get('skipped', False):
                            skipped.append(input_file)
                        elif not result['success']:
                            failed.append((input_file, result.get('error', 'Unknown error')))
                            if self._debug:
                                logging.error(f"Failed to process {input_file}: {result.get('error')}")
                    except Exception as e:
                        failed.append((input_file, str(e)))
                        if self._debug:
                            logging.error(f"Failed to process {input_file}: {e}")
                    finally:
                        pbar.update(1)
                        
                    # Check shutdown flag periodically
                    if shutdown_flag.is_set() and not future.done():
                        future.cancel()
        
        # Print summary 
        if True: # or change to: self._debug
            successful = len([r for r in results.values() if r['success']])
            logging.info(f"\nProcessing Summary:")
            logging.info(f"Total files: {len(input_files)}")
            logging.info(f"Successful: {successful}")
            logging.info(f"Skipped: {len(skipped)}")
            logging.info(f"Failed: {len(failed)}")
            
            if skipped: # and self._debug:
                logging.info("\nSkipped files (output already exists):")
                for file in skipped:
                    logging.info(f"  {file}")
            
            if failed: # and self._debug:
                logging.info("\nFailed files:")
                for file, error in failed:
                    logging.info(f"  {file}: {error}")
        
        return {
            'results': results,
            'failed': failed,
            'skipped': skipped
        }
    
    def _extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract PDF metadata using PyPDF"""
        metadata = {}
        try:
            from pypdf import PdfReader
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                doc_info = reader.metadata
                if doc_info:
                    for key, value in doc_info.items():
                        metadata[key] = value
        except Exception as e:
            logging.warning(f"Failed to extract PDF metadata: {e}")
        return metadata

    def _extract_epub_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract EPUB metadata using ebooklib"""
        metadata = {}
        try:
            from ebooklib import epub
            book = epub.read_epub(file_path)
            titles = book.get_metadata('DC', 'title')
            creators = book.get_metadata('DC', 'creator')
            metadata['title'] = titles[0][0] if titles else None
            metadata['authors'] = [item[0] for item in creators] if creators else None
        except Exception as e:
            logging.warning(f"Failed to extract EPUB metadata: {e}")
        return metadata
    
    def _get_unique_output_path(self, input_file: str, output_dir: Optional[str] = None, noskip: bool = False) -> str:
        """
        Generate output path for a given input file
        
        Args:
            input_file: Path to input file
            output_dir: Optional output directory
            noskip: Whether to generate unique filenames for existing files
            
        Returns:
            Output path (without creating unique name if noskip=False)
        """
        # Convert input file to absolute path
        input_path = Path(input_file).resolve()
        
        # Use current directory if none specified, ensure it's a Path
        output_dir = Path(output_dir or '.').resolve()
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get base name without extension
        base_name = input_path.stem
        
        # Create basic output path
        output_name = f"{base_name}.txt"
        output_path = output_dir / output_name
        
        # If noskip is True, ensure we have a unique filename to avoid overwriting
        if noskip and output_path.exists():
            counter = 1
            while True:
                output_name = f"{base_name}_{counter}.txt"
                output_path = output_dir / output_name
                
                if not output_path.exists():
                    if self._debug:
                        logging.debug(f"Using unique output path: {output_path}")
                    break
                
                counter += 1
        
        return str(output_path)
    
    def _process_metadata_for_sorting(self, input_file, text, llm_provider, rename_script_path, output_path=None):
        """
        Extract and process metadata for sorting/renaming files.
        
        Args:
            input_file: Path to the input file
            text: Extracted text content
            llm_provider: LLM provider instance or OpenAI client for Ollama
            rename_script_path: Path to write rename commands
            output_path: Path to the output text file
            
        Returns:
            dict: Dictionary with result information
        """
        result = {
            'success': False,
            'metadata': None,
            'renamed': False,
            'error': None
        }
        
        try:
            logging.debug(f"Processing metadata for llm {llm_provider}.")

            # Get metadata using the appropriate provider
            is_openai_client = hasattr(llm_provider, 'chat') and hasattr(llm_provider.chat, 'completions')
            
            if is_openai_client:
                # Using OpenAI client for Ollama
                metadata_content = send_to_ollama_server(text, input_file, llm_provider)
            else:
                # Using an LLM provider instance
                metadata_content = send_to_llm(
                    text=text, 
                    filename=input_file, 
                    provider=llm_provider
                )
                
            if not metadata_content:
                result['error'] = "Failed to get metadata from LLM provider"
                return result
                
            # Parse metadata with improved parser
            metadata = parse_metadata(metadata_content)
            if not metadata:
                result['error'] = "Failed to parse metadata"
                return result
                
            # Get basic metadata fields
            author = metadata['author']
            title = metadata['title']
            year = metadata['year']
            language = metadata.get('language', 'en')
            
            # Validate and fix year
            year = validate_and_fix_year(year, os.path.basename(input_file), text[:5000])
            
            # Process and validate author name
            if not author or author.lower() in ["unknown", "unknownauthor", "n a"]:
                result['error'] = "Missing or invalid author name"
                return result
                
            # Process author with appropriate provider
            if is_openai_client:
                corrected_author = corrected_author = sort_author_names(
                    author_names=author,
                    provider=llm_provider,
                    #temperature=temperature,
                    #max_tokens=max_tokens
                )
            else:
                corrected_author = corrected_author = sort_author_names(
                    author_names=author,
                    provider=llm_provider,
                    #temperature=temperature,
                    #max_tokens=max_tokens
                )
                
            if not corrected_author or corrected_author == "UnknownAuthor":
                result['error'] = "Failed to format author name correctly"
                return result
                
            # Validate title
            if not title or title.lower() in ["unknown", "title", ""]:
                result['error'] = "Missing or invalid title"
                return result
                
            # Create target paths with sanitized names
            first_author = sanitize_filename(corrected_author)
            sanitized_title = sanitize_filename(title)
            
            # Create target directory path
            target_dir = os.path.join(os.path.dirname(input_file), first_author)
            
            # Create new filename with year and title
            file_extension = os.path.splitext(input_file)[1].lower()
            new_filename = f"{year} {sanitized_title}{file_extension}"
            
            # Add language code suffix for non-English files if detected
            if language and language.lower() not in ['en', 'eng', 'english', 'unknown']:
                # Extract just the base extension without dot
                base_ext = file_extension[1:] if file_extension.startswith('.') else file_extension
                # Replace the file extension with language code + extension
                new_filename = f"{year} {sanitized_title}.{language}.{base_ext}"
            
            logging.debug(f"New path/filename will be: {target_dir}/{new_filename}")
            
            # Add rename command to the script
            add_rename_command(
                rename_script_path,
                source_path=input_file,
                target_dir=target_dir,
                new_filename=new_filename,
                output_dir=os.path.dirname(output_path) if output_path else None
            )
            
            result['success'] = True
            result['metadata'] = {
                'author': corrected_author,
                'title': title,
                'year': year,
                'language': language
            }
            result['renamed'] = True
            
        except Exception as e:
            result['error'] = str(e)
            logging.error(f"Error processing metadata: {e}")
            
        return result

    def _handle_sorting(self, input_file, text, llm_provider, rename_script_path, output_path, counters):
        """
        Handle sorting operations for a single file.
        
        Args:
            input_file: Path to the input file
            text: Extracted text content
            llm_provider: LLM provider instance or OpenAI client
            rename_script_path: Path to the rename script
            output_path: Path to the output text file
            counters: Dictionary of counters
            
        Returns:
            dict: Dictionary with result information
        """
        try:
            # Process metadata for sorting
            sort_result = self._process_metadata_for_sorting(
                input_file=input_file,
                text=text,
                llm_provider=llm_provider,
                rename_script_path=rename_script_path,
                output_path=output_path
            )
            
            if sort_result['success']:
                counters['sorted'] += 1
                return {
                    'success': True,
                    'metadata': sort_result['metadata']
                }
            else:
                # Log the error
                error_msg = sort_result.get('error', "Unknown error during sorting")
                logging.warning(f"Failed to sort file {input_file}: {error_msg}")
                
                # Add to unparseables list
                with file_lock:
                    with open("unparseables.lst", "a") as unparseable_file:
                        unparseable_file.write(f"{input_file} - {error_msg}\n")
                        unparseable_file.flush()
                        
                counters['sort_failed'] += 1
                return {
                    'success': False,
                    'error': error_msg
                }
                
        except Exception as e:
            # Catch any unhandled exceptions
            error_msg = f"Error sorting file: {str(e)}"
            logging.error(error_msg)
            
            # Add to unparseables list
            with file_lock:
                with open("unparseables.lst", "a") as unparseable_file:
                    unparseable_file.write(f"{input_file} - {error_msg}\n")
                    unparseable_file.flush()
                    
            counters['sort_failed'] += 1
            return {
                'success': False,
                'error': error_msg
            }

    def _process_single_file(self, input_file: str,
                output_dir: Optional[str] = None,
                method: Optional[str] = None,
                ocr_method: Optional[str] = None,
                password: Optional[str] = None,
                extract_tables: bool = False,
                noskip: bool = False,
                sort: bool = False,
                rename_script_path: str = None,
                counters: Dict[str, int] = None,
                llm_provider = None,     
                temperature: float = 0.7,
                max_tokens: int = 250,
                **kwargs) -> Dict[str, Any]:
        """
        Process a single document, with optimized skipping logic for sorting
        
        Args:
            input_file: Path to input file
            output_dir: Optional output directory
            method: Preferred extraction method
            password: Password for encrypted documents
            extract_tables: Whether to extract tables
            noskip: Whether to process even if output exists
            sort: Whether to sort files based on content
            rename_script_path: Path to write rename commands
            counters: Dictionary for tracking statistics
            **kwargs: Additional extraction options
            
        Returns:
            Dict with processing results
        """
        # Initialize result dictionary
        result = {
            'success': False,
            'text': '',
            'tables': [],
            'metadata': {},
            'input_file': input_file,
            'skipped': False
        }
        
        # Initialize counters if not provided
        if counters is None:
            counters = {
                'total': 0,
                'processed': 0,
                'skipped': 0,
                'sorted': 0,
                'sort_failed': 0,
                'failed': 0
            }
        
        # Check for shutdown flag
        if shutdown_flag.is_set():
            logging.debug(f"Shutdown flag detected. Skipping {input_file}")
            result['error'] = "Processing aborted due to shutdown signal"
            return result
        
        try:
            # Generate basic output path (no uniqueness yet)
            logging.debug(f"Processing for {llm_provider}.")
            basic_output_path = self._get_unique_output_path(input_file, output_dir, noskip=False)
            
            # Check if we should skip this file
            should_skip = False
            if os.path.exists(basic_output_path) and not noskip:
                # If not sorting, skip existing files
                if not sort:
                    should_skip = True
                else:
                    # When sorting, skip only if file is already in rename script
                    if rename_script_path and os.path.exists(rename_script_path):
                        try:
                            # Check if input file path is in rename script
                            with open(rename_script_path, 'r') as script_file:
                                script_content = script_file.read()
                                if input_file in script_content:
                                    should_skip = True
                                    logging.debug(f"File {input_file} already in rename script, skipping")
                        except Exception as e:
                            logging.error(f"Error checking rename script: {e}")
                            # Continue processing if we can't check the rename script
            
            if should_skip:
                if self._debug:
                    logging.info(f"Skipping {input_file} - output file exists and already processed")
                result['success'] = True
                result['skipped'] = True
                result['output_path'] = basic_output_path
                counters['skipped'] += 1
                return result
            
            # Get the actual output path (which might be unique if noskip=True)
            output_path = self._get_unique_output_path(input_file, output_dir, noskip=noskip)
            
            # Determine if we need to extract text or can use existing file
            text = ""
            reused_text = False
            
            if sort and os.path.exists(basic_output_path):
                # Reuse existing text file for sorting
                try:
                    with open(basic_output_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    reused_text = True
                    logging.debug(f"Reusing existing text from {basic_output_path} for sorting")
                except Exception as e:
                    logging.error(f"Error reading existing text file: {e}")
                    # Will fall back to extraction
                    
            # Extract text if we couldn't reuse existing
            if not reused_text:
                if self._debug:
                    logging.info(f"Processing {input_file} -> {output_path}")
                    
                text = self.manager.extract(
                    input_file,
                    method=method,
                    ocr_method=ocr_method,
                    password=password,
                    extract_tables=extract_tables,
                    **kwargs
                )
            
            if text:
                result['text'] = text
                result['success'] = True
                counters['processed'] += 1
                
                # Only save the text to file if we extracted it (not if we reused existing)
                if not reused_text:
                    try:
                        # Make sure the directory exists
                        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(text)
                        result['output_path'] = output_path
                        
                        if self._debug:
                            logging.info(f"Saved text to {output_path}")
                        else:
                            logging.debug(f"Saved text to {output_path}")  # Log even in non-debug mode
                            
                    except Exception as e:
                        logging.error(f"Failed to write output file {output_path}: {e}")
                        result['success'] = False
                        result['error'] = str(e)
                        counters['failed'] += 1
                        return result
                else:
                    result['output_path'] = basic_output_path
                
                # Handle sorting if enabled
                try:
                    # First, check what type of provider we have
                    is_openai_client = False
                    if llm_provider is None:
                        # Fall back to local Ollama
                        openai_client = get_openai_client()
                        is_openai_client = True
                        metadata_content = send_to_ollama_server(text, input_file, openai_client)
                    else:
                        # Use the provided LLM provider
                        logging.debug(f"Sending to llm {llm_provider}.")
                        metadata_content = send_to_llm(
                            text=text, 
                            filename=input_file, 
                            provider=llm_provider
                        )
                    
                    if metadata_content:
                        # Parse metadata with improved parser
                        metadata = parse_metadata(metadata_content)
                        if metadata:
                            # Process author names
                            author = metadata['author']
                            logging.debug(f"extracted author: {author}")
                            
                            # Use appropriate method for author name sorting
                            if is_openai_client:
                                corrected_author = sort_author_names(author, openai_client)
                            else:
                                corrected_author = corrected_author = sort_author_names(
                                    author_names=author,
                                    provider=llm_provider,
                                    temperature=temperature,
                                    max_tokens=max_tokens
                                )
                            
                            logging.debug(f"corrected author: {corrected_author}")
                            metadata['author'] = corrected_author
                                
                            # Get file details
                            title = metadata['title']
                            year = metadata.get('year', 'Unknown')
                            
                            # Validate and fix year with new helper function
                            year = validate_and_fix_year(year, os.path.basename(input_file), text[:5000])
                            
                            # Get language if available
                            language = metadata.get('language', 'en')
                            
                            # Validate essential metadata
                            if not corrected_author or corrected_author == "UnknownAuthor" or not title:
                                logging.warning(f"Missing author or title for {input_file}. Skipping rename.")
                                with file_lock:
                                    with open("unparseables.lst", "a") as unparseable_file:
                                        unparseable_file.write(f"{input_file} - Missing metadata: Author='{corrected_author}', Title='{title}'\n")
                                        unparseable_file.flush()
                                counters['sort_failed'] += 1
                            else:
                                # Create target paths with sanitized names
                                first_author = sanitize_filename(corrected_author)
                                sanitized_title = sanitize_filename(title)
                                target_dir = os.path.join(os.path.dirname(input_file), first_author)
                                file_extension = os.path.splitext(input_file)[1].lower()
                                
                                # Create filename with appropriate formatting
                                # Handle non-English files with language code
                                if language and language.lower() not in ['en', 'eng', 'english', 'unknown']:
                                    # Extract just the base extension without dot
                                    base_ext = file_extension[1:] if file_extension.startswith('.') else file_extension
                                    # Add language code before extension
                                    new_filename = f"{year} {sanitized_title}.{language}.{base_ext}"
                                else:
                                    new_filename = f"{year} {sanitized_title}{file_extension}"
                                
                                logging.debug(f"New path/filename will be: {target_dir}/{new_filename}")
                                
                                # Add rename command with improved function
                                add_rename_command(
                                    rename_script_path,
                                    source_path=input_file,
                                    target_dir=target_dir,
                                    new_filename=new_filename,
                                    output_dir=os.path.dirname(output_path) if output_path else None
                                )
                                
                                result['metadata'] = metadata
                                counters['sorted'] += 1
                        else:
                            logging.warning(f"Failed to parse metadata for {input_file}")
                            with file_lock:
                                with open("unparseables.lst", "a") as unparseable_file:
                                    unparseable_file.write(f"{input_file} - Failed to parse metadata format: {metadata_content[:100]}...\n")
                                    unparseable_file.flush()
                            counters['sort_failed'] += 1
                    else:
                        logging.warning(f"Failed to get metadata from Ollama server for {input_file}")
                        with file_lock:
                            with open("unparseables.lst", "a") as unparseable_file:
                                unparseable_file.write(f"{input_file} - Failed to get metadata from Ollama server\n")
                                unparseable_file.flush()
                        counters['sort_failed'] += 1
                except Exception as sort_e:
                    logging.error(f"Error sorting file {input_file}: {sort_e}")
                    with file_lock:
                        with open("unparseables.lst", "a") as unparseable_file:
                            unparseable_file.write(f"{input_file} - Error during sorting: {str(sort_e)}\n")
                            unparseable_file.flush()
                    counters['sort_failed'] += 1
                
                # Extract tables if requested (only for PDFs)
                if extract_tables and input_file.lower().endswith('.pdf'):
                    try:
                        tables = self._table_extractor.extract_tables(input_file)
                        result['tables'] = [table.df.to_dict() for table in tables]
                        if self._debug:
                            logging.info(f"Extracted {len(tables)} tables from {input_file}")
                    except Exception as te:
                        logging.error(f"Table extraction failed for {input_file}: {te}")
                        result['tables'] = []
                
                # Extract file metadata
                result['metadata'].update(self._extract_metadata(input_file))
                    
            else:
                logging.error(f"Failed to extract text from {input_file}")
                result['error'] = "No text extracted"
                with file_lock:
                    with open("unparseables.lst", "a") as unparseable_file:
                        unparseable_file.write(f"{input_file} - No text extracted\n")
                        unparseable_file.flush()
                counters['failed'] += 1
                
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logging.error(error_msg)
            result['error'] = error_msg
            result['success'] = False
            counters['failed'] += 1
            
            with file_lock:
                with open("unparseables.lst", "a") as unparseable_file:
                    unparseable_file.write(f"{input_file} - Processing error: {str(e)}\n")
                    unparseable_file.flush()
        
        return result
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract document metadata"""
        metadata = {}
        try:
            file_info = Path(file_path)
            metadata.update({
                'filename': file_info.name,
                'size': file_info.stat().st_size,
                'modified': file_info.stat().st_mtime
            })
            
            # Extract document-specific metadata
            if file_info.suffix.lower() == '.pdf':
                metadata.update(self._extract_pdf_metadata(file_path))
            elif file_info.suffix.lower() == '.epub':
                metadata.update(self._extract_epub_metadata(file_path))
                
        except Exception as e:
            if self._debug:
                logging.error(f"Metadata extraction failed: {e}")
                
        return metadata


# Signal handler to set the shutdown flag
# Keep the global signal handler for setting the shutdown_flag
def signal_handler(signum, frame):
    """
    Signal handler for SIGINT, SIGTERM, and other interrupt signals.
    Sets the shutdown flag and logs the interrupt event.
    """
    if not shutdown_flag.is_set():  # Only log once
        signal_name = {
            signal.SIGINT: "SIGINT (Ctrl+C)",
            signal.SIGTERM: "SIGTERM",
            signal.SIGTSTP: "SIGTSTP (Ctrl+Z)"
        }.get(signum, f"Signal {signum}")
        
        logging.info(f"Received {signal_name}. Initiating graceful shutdown...")
        shutdown_flag.set()

# Register global handlers outside main()
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if platform.system() != 'Windows':
    signal.signal(signal.SIGTSTP, signal_handler)

def is_file_in_rename_script(rename_script_path, input_file):
    """
    Check if a file path is already mentioned in the rename script(s).
    Checks both bash and batch scripts if needed.
    
    Args:
        rename_script_path: Path to the rename script
        input_file: Input file path to check
        
    Returns:
        bool: True if file is already in any rename script, False otherwise
    """
    # Determine if we need to check both scripts
    is_windows = platform.system() == 'Windows'
    script_ext = os.path.splitext(rename_script_path)[1].lower()
    
    scripts_to_check = [rename_script_path]
    
    # Add batch script if on Windows and main script is not .bat
    if is_windows and script_ext != '.bat':
        batch_script_path = os.path.splitext(rename_script_path)[0] + '.bat'
        if os.path.exists(batch_script_path):
            scripts_to_check.append(batch_script_path)
    
    # Check each script
    for script_path in scripts_to_check:
        if not os.path.exists(script_path):
            continue
            
        try:
            with open(script_path, 'r') as script_file:
                content = script_file.read()
                
                # For Windows script, check both with forward and backslashes
                if script_path.endswith('.bat'):
                    unix_path = input_file
                    win_path = input_file.replace('/', '\\')
                    if unix_path in content or win_path in content:
                        return True
                else:
                    if input_file in content:
                        return True
        except Exception as e:
            logging.error(f"Error checking rename script {script_path}: {e}")
    
    # If we get here, the file isn't in any script
    return False
    
def sanitize_filename(name):
    """Sanitize a filename to ensure safe filesystem operations"""
    # Remove disallowed characters
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    
    # Handle spacing and initials
    if " " in name:
        name = re.sub(r'(\w)\.', r'\1', name)
        parts = name.split()
        new_parts = []
        current_initials = ""
        for part in parts:
            if len(part) == 1:
                current_initials += part
            else:
                if current_initials:
                    new_parts.append(current_initials)
                    current_initials = ""
                new_parts.append(part)
        if current_initials:
            new_parts.append(current_initials)
        name = " ".join(new_parts)
    
    return name.strip().replace('/', '')


def clean_author_name(author_name):
    """
    Clean author name by removing titles, extra spaces, and punctuation.
    """
    # Remove common titles
    author_name = re.sub(r'\b(Dr|Prof|Mr|Mrs|Ms|Rev|Sir)\.?\s+', '', author_name, flags=re.IGNORECASE)
    
    # Remove XML entities
    author_name = re.sub(r'&[a-zA-Z]+;', ' ', author_name)
    
    # Remove special characters but keep diacritics for non-English names
    author_name = re.sub(r'[^\w\s.\'-áéíóúàèìòùäëïöüÄËÏÖÜâêîôûÂÊÎÔÛñÑçÇ]', ' ', author_name, flags=re.UNICODE)
    
    # Clean up extra spaces
    author_name = re.sub(r'\s+', ' ', author_name).strip()
    
    return author_name

def valid_author_name(author_name):
    """
    Check if the author name is valid.
    Flexible validation to handle various name formats.
    """
    # Quick sanity check for empty input
    if not author_name or len(author_name.strip()) < 3:
        return False
        
    # Split into parts
    parts = author_name.strip().split()
    
    # Name should have at least two parts
    if len(parts) < 2:
        return False
        
    # Check for placeholder values
    lower_name = author_name.lower()
    if "unknown" in lower_name and len(parts) == 2 and parts[1].lower() == "unknown":
        # Special case: Allow "Lastname Unknown" as a valid format
        return True
        
    if any(placeholder in lower_name for placeholder in 
           ["unknownauthor", "lastname", "surname", "firstname", "author", "n a"]):
        return False
        
    # Check for reasonable character content
    # Allow letters, spaces, periods, hyphens, apostrophes and accented characters
    if not re.match(r'^[\w\s.\'-áéíóúàèìòùäëïöüÄËÏÖÜâêîôûÂÊÎÔÛñÑçÇ]+$', author_name, re.UNICODE):
        return False
        
    # Each part should be reasonably sized
    if any(len(part) < 1 or len(part) > 20 for part in parts):
        return False
        
    return True

def validate_and_fix_year(year, filename=None, text_sample=None):
    """
    Validate and fix the detected year.
    
    Args:
        year: The year to validate
        filename: Optional filename to extract year from if not valid
        text_sample: Optional text sample to look for year in if not in filename
        
    Returns:
        str: A valid year or "UnknownYear"
    """
    current_year = datetime.now().year
    
    # If year is valid (4 digits between 1500 and current year + 1)
    if year and re.match(r'^\d{4}$', year) and 1500 <= int(year) <= current_year + 1:
        return year
        
    # Try to extract from filename if provided
    if filename:
        filename_year = extract_year_from_filename(filename)
        if filename_year:
            return filename_year
            
    # Try to extract from text if provided
    if text_sample:
        # Look for copyright pattern
        copyright_match = re.search(r'copyright\s+©?\s*(\d{4})', text_sample, re.IGNORECASE)
        if copyright_match:
            return copyright_match.group(1)
            
        # Look for publication pattern
        pub_match = re.search(r'published\s+in\s+(\d{4})', text_sample, re.IGNORECASE)
        if pub_match:
            return pub_match.group(1)
            
        # Look for any 4-digit year in a reasonable range
        year_matches = re.findall(r'\b(1[5-9]\d\d|20[0-2]\d)\b', text_sample)
        if year_matches:
            # Take the most recent year that's not in the future
            valid_years = [y for y in year_matches if int(y) <= current_year]
            if valid_years:
                return max(valid_years)
    
    # Default if no valid year found
    return "UnknownYear"

def execute_rename_commands(script_path):
    """Execute the generated rename commands script"""
    try:
        # Ensure the file is closed before executing
        with open(script_path, 'r') as script_file:
            pass  # Just to ensure it's accessible and can be opened
        subprocess.run(['bash', script_path], check=True)
        logging.info(f"Successfully executed rename commands from {script_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing rename commands: {e}. Please check the rename script for issues.")
    except FileNotFoundError:
        logging.error(f"Rename script '{script_path}' not found.")
    except PermissionError:
        logging.error(f"Permission denied while executing the rename script '{script_path}'. Ensure it is executable.")
    except Exception as e:
        logging.error(f"Unexpected error during rename command execution: {e}")


def parse_metadata(content, verbose=False):
    """
    Parse metadata content returned by the Ollama server supporting multiple formats.
    Properly handles author names with commas.
    
    Returns:
        dict or None: Dictionary containing author, year, title, and language
    """
    # Remove XML declarations which might interfere with parsing
    content = re.sub(r'<\?xml[^>]+\?>', '', content)
    
    # Fix common tag issues
    content = content.replace("<TITLE", "<TITLE>").replace("<AUTHOR", "<AUTHOR>")
    content = content.replace("<YEAR", "<YEAR>").replace("<LANGUAGE", "<LANGUAGE>")
    
    # Try multiple tag formats for each field
    title_patterns = [
        r'<TITLE>(.*?)</TITLE>',
        r'<Title>(.*?)</Title>',
        r'<title>(.*?)</title>',
        r'"(.*?)"'  # For cases where title is just in quotes
    ]
    
    year_patterns = [
        r'<YEAR>(\d{4})</YEAR>',
        r'<Year>(\d{4})</Year>',
        r'<year>(\d{4})</year>'
    ]
    
    author_patterns = [
        r'<AUTHOR>(.*?)</AUTHOR>',
        r'<Author>(.*?)</Author>',
        r'<author>(.*?)</author>'
    ]
    
    language_patterns = [
        r'<LANGUAGE>(.*?)</LANGUAGE>',
        r'<Language>(.*?)</Language>',
        r'<language>(.*?)</language>'
    ]
    
    # Try to extract title
    title = None
    for pattern in title_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            title = match.group(1).strip()
            if title:
                break
    
    # Try to extract year
    year = "Unknown"
    for pattern in year_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            year = match.group(1).strip()
            if year:
                break
    
    # Try to extract author
    author = None
    for pattern in author_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            author = match.group(1).strip()
            if author:
                break
    
    # Try to extract language
    language = "en"
    for pattern in language_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            language = match.group(1).strip().lower()
            if language:
                break
    
    # Don't split on commas in author names - they're likely "Lastname, Firstname" format
    # Instead, handle multiple authors separated by semicolons
    if author and ';' in author:
        # Split on semicolons and keep only the first author
        author = author.split(';')[0].strip()
    
    # Further clean author name
    if author:
        # Remove XML entities
        author = re.sub(r'&[a-zA-Z]+;', ' ', author)
        # Remove placeholder text
        author = re.sub(r'\b(Lastname|Firstname|Surname)\b', '', author, flags=re.IGNORECASE)
        author = re.sub(r'\s+', ' ', author).strip()
    
    # Check if author is a template/placeholder
    if author and author.lower() in ["lastname firstname", "surname firstname"]:
        author = None
    
    # Validate extracted data
    if not title:
        logging.warning(f"No match for title in content")
        return None
    if not author:
        logging.warning(f"No match for author in content")
        return None
    
    # Sanitize filenames
    title = sanitize_filename(title) if title else "unknown"
    author = sanitize_filename(author) if author else "unknown"
    year = sanitize_filename(year)
    language = sanitize_filename(language)
    
    # Check for placeholder values
    if any(placeholder in (title.lower(), author.lower()) 
           for placeholder in ["unknown", "unknownauthor", "n a", ""]):
        logging.warning("Warning: Found 'unknown', 'n a', or empty strings in metadata.")
        return None
        
    return {'author': author, 'year': year, 'title': title, 'language': language}

def send_to_ollama_server(text, filename, openai_client, max_attempts=5, verbose=False):
    """
    Query the Ollama server to extract author, year, title, and language with exponential backoff.
    
    Returns:
        str: The formatted metadata response
    """
    base_retry_wait = 2  # Base wait time in seconds
    attempt = 1
    
    # Prepare different prompt templates to try if earlier ones fail
    prompt_templates = [
        # First attempt - simple structured format
        (
            f"Extract the main author name (Lastname Surname), "
            f"year of publication, title, and language from the following text, considering the filename '{os.path.basename(filename)}' "
            f"which may contain clues. I need the output **only** in the following format with no additional text or explanations: \n"
            f"<TITLE>The publication title</TITLE>\n<YEAR>2023</YEAR>\n<AUTHOR>Lastname Surname</AUTHOR>\n<LANGUAGE>en</LANGUAGE>\n\n"
        ),
        # Second attempt - emphasize exact format
        (
            f"I need to extract metadata from a document. Please give me ONLY these four tags with the information, and nothing else:\n"
            f"<TITLE>The exact title</TITLE>\n<YEAR>The publication year (4 digits)</YEAR>\n<AUTHOR>The main author's name (LastName FirstName)</AUTHOR>\n<LANGUAGE>The language code</LANGUAGE>\n\n"
            f"Document filename: {os.path.basename(filename)}\n"
        ),
        # Third attempt - even more explicit
        (
            f"You are a metadata extraction tool. Extract these fields from the text:\n"
            f"1. TITLE (the full publication title)\n"
            f"2. YEAR (the 4-digit publication year, use 'Unknown' if not found)\n"
            f"3. AUTHOR (the main author's last name and first name)\n"
            f"4. LANGUAGE (the 2-letter language code, e.g., 'en', 'de', 'fr')\n\n"
            f"Format your response EXACTLY like this with no other text:\n"
            f"<TITLE>The title</TITLE>\n<YEAR>2023</YEAR>\n<AUTHOR>Smith John</AUTHOR>\n<LANGUAGE>en</LANGUAGE>\n\n"
        )
    ]
    
    # Try different prompt templates if we encounter format issues
    while attempt <= max_attempts and not shutdown_flag.is_set():
        # Choose prompt template based on attempt number
        template_index = min(attempt - 1, len(prompt_templates) - 1)
        prompt_template = prompt_templates[template_index]
        
        logging.debug(f"Consulting Ollama server on file: {filename} (Attempt: {attempt}, Template: {template_index + 1})")
        
        # Build the final prompt with text sample
        prompt = prompt_template + f"Here is the document text:\n{text[:3000]}"  # Limit text to avoid token limits
        messages = [{"role": "user", "content": prompt}]
        
        with ollama_semaphore:
            try:
                response = openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=0.5,  # Reduced temperature for more consistent formatting
                    max_tokens=250,
                    messages=messages,
                    timeout=120  # 2 minute timeout
                )
                
                output = response.choices[0].message.content.strip()
                logging.debug(f"Metadata content received from server: {output}")
                
                # Use the new more flexible parser
                metadata = parse_metadata(output, verbose=verbose)
                if metadata:
                    return output
                else:
                    logging.warning(f"Unexpected response format from Ollama server: {output}")
                    # Less aggressive backoff for format issues - might not be server's fault
                    if attempt < max_attempts:
                        wait_time = base_retry_wait * (1.5 ** (attempt - 1))  # Gentler exponential backoff
                        logging.info(f"Retrying with different prompt in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                        attempt += 1
                        continue
                    return output
                
            except Exception as e:
                if "rate_limit" in str(e).lower() or "timeout" in str(e).lower():
                    # Use exponential backoff for rate limiting/timeouts
                    wait_time = base_retry_wait * (2 ** (attempt - 1))  # Exponential backoff
                    logging.info(f"Rate limit or timeout encountered. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                else:
                    logging.error(f"Error communicating with Ollama server for {filename}: {e}")
                    if attempt < max_attempts:
                        wait_time = base_retry_wait * (1.5 ** (attempt - 1))  # Gentler exponential backoff
                        logging.info(f"Retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                        attempt += 1
                        continue
                return ""
                
    logging.error(f"Maximum retry attempts reached for sending to Ollama server.")
    return ""

def initialize_rename_scripts(rename_script_path):
    """Initialize both bash and batch rename scripts if needed"""
    is_windows = platform.system() == 'Windows'
    
    # Get script extension
    script_ext = os.path.splitext(rename_script_path)[1].lower()
    create_bash = script_ext in ['', '.sh']
    create_batch = is_windows or script_ext == '.bat'
    
    # Derive the batch script path if needed
    batch_script_path = rename_script_path
    if create_batch and script_ext != '.bat':
        batch_script_path = os.path.splitext(rename_script_path)[0] + '.bat'
    
    # Initialize bash script
    if create_bash:
        with open(rename_script_path, "w") as bash_file:
            bash_file.write("#!/bin/bash\n")
            bash_file.write('set -e\n\n')
            bash_file.flush()
        
        try:
            os.chmod(rename_script_path, 0o755)  # Make executable
            logging.debug(f"Made bash script executable: {rename_script_path}")
        except Exception as e:
            logging.warning(f"Could not set executable permission on {rename_script_path}: {e}")
    
    # Initialize batch script
    if create_batch:
        with open(batch_script_path, "w") as batch_file:
            batch_file.write("@echo off\n")
            batch_file.write("setlocal enabledelayedexpansion\n\n")
            batch_file.write("rem Rename script for Windows\n\n")
            batch_file.flush()
    
    return {
        'bash_script': rename_script_path if create_bash else None,
        'batch_script': batch_script_path if create_batch else None
    }

def sort_author_names(author_names, provider, temperature: float = 0.3, 
                    max_tokens: int = 100, max_attempts: int = 5, 
                    verbose: bool = False):
    """
    Format author names into 'Lastname Firstname' format using LLM with backoff.
    
    Args:
        author_names: Author names to format
        provider: LLM provider instance or OpenAI client
        temperature: Temperature setting for generation
        max_tokens: Maximum tokens to generate
        max_attempts: Maximum retry attempts
        verbose: Whether to print debug info
    """
    if provider is None or isinstance(provider, (int, float)):
        logging.warning(f"Invalid provider passed to sort_author_names: {provider}")
        return author_names
    
    if not author_names or author_names.strip() in ["Unknown", "UnknownAuthor", "n a", ""]:
        return "UnknownAuthor"
    
    # First clean the input and handle commas properly
    author_names = author_names.replace('</AUTHOR>', '').replace('<AUTHOR>', '')
    
    # Check for comma format (likely "Lastname, Firstname")
    if ',' in author_names:
        # This is already likely in "Lastname, Firstname" format
        # Just remove the comma to get "Lastname Firstname"
        formatted_name = author_names.replace(',', ' ').strip()
        # Clean up extra spaces
        formatted_name = re.sub(r'\s+', ' ', formatted_name)
        
        # Validate the name has at least two parts
        if ' ' in formatted_name:
            logging.debug(f"Processed comma-formatted name: {author_names} -> {formatted_name}")
            return formatted_name
    
    # Remove placeholder text that might appear in responses
    formatted_author_names = re.sub(r'\b(Lastname|Firstname|Surname)\b', '', author_names, flags=re.IGNORECASE)
    formatted_author_names = re.sub(r'\s+', ' ', formatted_author_names).strip()
    
    # Handle multiple authors separated by different delimiters (but not commas)
    if any(delimiter in formatted_author_names for delimiter in [';', '&', ' and ']):
        # Split on these delimiters and keep only the first author
        for delimiter in [';', '&', ' and ']:
            if delimiter in formatted_author_names:
                formatted_author_names = formatted_author_names.split(delimiter)[0].strip()
                break
    
    # If after cleaning we have nothing, return unknown
    if not formatted_author_names or len(formatted_author_names.strip()) < 3:
        return "UnknownAuthor"
    
    # Check if name is already in "Lastname Firstname" format after basic processing
    parts = formatted_author_names.split()
    if len(parts) >= 2:
        # We have at least two parts, might be good enough
        return formatted_author_names
    
    # Determine if we're using OpenAI client for Ollama or an LLM provider
    is_openai_client = hasattr(provider, 'chat') and hasattr(provider.chat, 'completions')
    
    # Use LLM to get the correct format with retries
    base_retry_wait = 2  # Base wait time in seconds
    for attempt in range(1, max_attempts + 1):
        if verbose:
            logging.debug(f"Attempt {attempt} to sort author name: {formatted_author_names}")
        
        # Prepare different prompt templates to try if earlier ones fail
        if attempt == 1:
            prompt = (
                f"You will be given an author name that you must put into the format 'Lastname Firstname'. "
                f"If it appears to be just a last name without a first name, return it as is. "
                f"Examples:\n"
                f"- 'Michael Mustermann' → 'Mustermann Michael'\n"
                f"- 'Mustermann Michael' → 'Mustermann Michael' (already correct)\n"
                f"- 'Jean-Paul Sartre' → 'Sartre Jean-Paul'\n"
                f"- 'van Gogh Vincent' → 'van Gogh Vincent' (already correct)\n"
                f"- 'Butz' → 'Butz' (single name, return as is)\n"
                f"Respond ONLY with: <AUTHOR>Lastname Firstname</AUTHOR> or just <AUTHOR>Lastname</AUTHOR> for single names.\n\n"
                f"Author name: {formatted_author_names}"
            )
        else:
            # Try a different approach on subsequent attempts
            prompt = (
                f"I need the author name '{formatted_author_names}' formatted as 'Lastname Firstname(s)'. "
                f"If it's just a single name (like a last name), return it as is without adding anything. "
                f"Don't add any explanation, just return the correctly formatted name in this exact format: "
                f"<AUTHOR>Lastname Firstname</AUTHOR> or <AUTHOR>Lastname</AUTHOR> for single names."
            )
            
        messages = [{"role": "user", "content": prompt}]
        
        # Use the semaphore to limit concurrent requests
        with llm_semaphore:
            try:
                if not hasattr(provider, 'chat_completion'):
                    logging.warning(f"Provider {provider} missing chat_completion method")
                    return author_names
                
                if is_openai_client:
                    # Using OpenAI client for Ollama
                    response = provider.chat.completions.create(
                        model=MODEL_NAME,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        messages=messages
                    )
                    reformatted_name = response.choices[0].message.content.strip()
                else:
                    # Using an LLM provider instance
                    response = provider.chat_completion(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    reformatted_name = response["content"]
                
                # Check for tag issues and fix them
                if "<AUTHOR>" not in reformatted_name:
                    reformatted_name = f"<AUTHOR>{reformatted_name}</AUTHOR>"
                if "</AUTHOR>" not in reformatted_name and not reformatted_name.endswith("</AUTHOR>"):
                    reformatted_name = reformatted_name.replace("<AUTHOR>", "<AUTHOR>") + "</AUTHOR>"
                
                name_match = re.search(r'<AUTHOR>(.*?)</AUTHOR>', reformatted_name)
                if name_match:
                    ordered_name = name_match.group(1).strip()
                    # Final cleanup - remove any placeholder text that might appear
                    ordered_name = re.sub(r'\b(Lastname|Firstname|Surname)\b', '', ordered_name, flags=re.IGNORECASE)
                    ordered_name = clean_author_name(ordered_name)
                    logging.debug(f"Ordered name after cleaning: '{ordered_name}'")
                    
                    # Return it even if it's a single word - we won't add "Unknown"
                    if ordered_name and len(ordered_name) >= 2:
                        return ordered_name
                else:
                    # Try to extract the name without tags if tags are malformed
                    cleaned_response = reformatted_name.replace("</AUTHOR>", "").replace("<AUTHOR>", "").strip()
                    if cleaned_response and cleaned_response not in ['Lastname Firstname', 'Unknown']:
                        ordered_name = clean_author_name(cleaned_response)
                        if ordered_name and len(ordered_name) >= 2:
                            return ordered_name
                            
                    logging.warning(f"Failed to extract a valid name from: '{reformatted_name}', retrying...")
                
            except Exception as e:
                if "rate_limit" in str(e).lower() or "timeout" in str(e).lower():
                    wait_time = base_retry_wait * (2 ** (attempt - 1))
                    logging.info(f"Rate limit or timeout encountered. Retrying in {wait_time:.2f} seconds...")
                else:
                    logging.error(f"Error querying LLM for author names: {e}")
                    wait_time = base_retry_wait * (1.5 ** (attempt - 1))
                    logging.info(f"Retrying in {wait_time:.2f} seconds...")
                
                if attempt < max_attempts:
                    time.sleep(wait_time)
                    continue
                
                # Last resort - just return the original name, even if it's a single word
                return formatted_author_names
        
        # Wait a little between attempts even if no error occurred
        if attempt < max_attempts:
            time.sleep(1)  # Small pause between attempts
                
    logging.error(f"Maximum retry attempts reached for sorting author name: {formatted_author_names}")
    
    # Last resort - just return the original name
    return formatted_author_names

def extract_metadata(text, filename, llm_provider):
    """
    Extract metadata using the provided LLM provider
    
    Args:
        text: Text to analyze
        filename: Filename for context
        llm_provider: Either an OpenAI client for Ollama or an LLMProvider instance
    
    Returns:
        str: The metadata response
    """
    if isinstance(llm_provider, (int, float)):
        logging.error(f"Invalid llm_provider parameter: {llm_provider} (type: {type(llm_provider)})")
        return ""
    
    # Check if we're using OpenAI client for Ollama or an LLM provider
    is_openai_client = hasattr(llm_provider, 'chat') and hasattr(llm_provider.chat, 'completions')
    
    if is_openai_client:
        # Using OpenAI client for Ollama
        return send_to_ollama_server(text, filename, llm_provider)
    else:
        # Using an LLM provider instance
        return send_to_llm(text, filename, llm_provider)
    
def extract_year_from_filename(filename):
    """
    Try to extract a year (4 digits between 1500-2030) from a filename.
    
    Args:
        filename: The filename to check
        
    Returns:
        str: The year if found, None otherwise
    """
    # Look for a 4-digit number within the reasonable range for publication years
    matches = re.findall(r'\b(1[5-9]\d\d|20[0-2]\d)\b', filename)
    
    if matches:
        # Return the first reasonable match
        return matches[0]
    
    return None

def sort_author_names_old(author_names, openai_client, max_attempts=5, verbose=False):
    """Format author names into 'Lastname Firstname' format using LLM with backoff"""
    base_retry_wait = 2  # Base wait time in seconds
    for attempt in range(1, max_attempts + 1):
        logging.debug(f"Attempt {attempt} to sort author names: {author_names}")
        
        formatted_author_names = author_names.replace('&', ',')
        prompt = (
            f"You will be given an author name that you must put into the format 'Lastname Surname'. "
            f"So, you must first make an educated guess if the given input is already in this format. If so, return it back. "
            f"If not and it is more plausibly in the format 'Surname(s) Lastname', you must reformat it. "
            f"Example: 'Michael Max Mustermann' must become 'Mustermann Michael Max' and 'Joe A. Doe' must become 'Doe Joe A'. "
            f"No comma after the Lastname! "
            f"If you are given multiple person names, only keep the first and omit all others. "
            f"If it is impossible to come up with a correct name, return <AUTHOR>n a</AUTHOR>. "
            f"You must give the output in the format: <AUTHOR>Lastname Surname(s)</AUTHOR>. "
            f"Here are the name parts: <AUTHOR>{formatted_author_names}</AUTHOR>"
        )
        messages = [{"role": "user", "content": prompt}]
        
        # Use the semaphore to limit concurrent requests
        with ollama_semaphore:
            try:
                response = openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=0.5,
                    max_tokens=250,  # Reduced from 500 to save tokens
                    messages=messages
                )
                
                reformatted_name = response.choices[0].message.content.strip()
                name_match = re.search(r'<AUTHOR>(.*?)</AUTHOR>', reformatted_name)
                if name_match:
                    ordered_name = name_match.group(1).strip().split(",")[0].strip()
                    ordered_name = clean_author_name(ordered_name)
                    logging.debug(f"Ordered name after cleaning: '{ordered_name}'")
                    if valid_author_name(ordered_name):
                        return ordered_name
                    else:
                        logging.warning(f"Invalid author name format detected: '{ordered_name}', retrying...")
                else:
                    logging.warning("Failed to extract a valid name, retrying...")
                
            except Exception as e:
                if "rate_limit" in str(e).lower() or "timeout" in str(e).lower():
                    wait_time = base_retry_wait * (2 ** (attempt - 1))
                    logging.info(f"Rate limit or timeout encountered. Retrying in {wait_time:.2f} seconds...")
                else:
                    logging.error(f"Error querying Ollama server for author names: {e}")
                    wait_time = base_retry_wait * (1.5 ** (attempt - 1))
                    logging.info(f"Retrying in {wait_time:.2f} seconds...")
                
                if attempt < max_attempts:
                    time.sleep(wait_time)
                    continue
                return "UnknownAuthor"
        
        # Wait a little between attempts even if no error occurred
        if attempt < max_attempts:
            time.sleep(1)  # Small pause between attempts
                
    logging.error("Maximum retry attempts reached for sorting author names.")
    return "UnknownAuthor"

def add_rename_command(rename_script_path, source_path, target_dir, new_filename, output_dir=None):
    """
    Add mkdir and mv commands to the rename script.
    Also moves the corresponding .txt file if it exists.
    Generates appropriate commands for both bash and Windows batch scripts.
    
    Parameters:
        rename_script_path: Path to the rename script file
        source_path: Source file path
        target_dir: Target directory path - will have any commas removed
        new_filename: New filename
        output_dir: Optional output directory for text files
    """
    # Remove any commas from target directory name
    target_dir = target_dir.replace(',', '')
    
    # Determine if we're on Windows
    is_windows = platform.system() == 'Windows'
    
    # Get script extension to determine if we need to create both scripts
    script_ext = os.path.splitext(rename_script_path)[1].lower()
    create_bash = script_ext in ['', '.sh']
    create_batch = is_windows or script_ext == '.bat'
    
    # Derive the batch script path if needed
    batch_script_path = rename_script_path
    if create_batch and script_ext != '.bat':
        batch_script_path = os.path.splitext(rename_script_path)[0] + '.bat'
    
    # Prepare paths for bash script
    escaped_source_path = escape_special_chars(source_path)
    escaped_target_dir = escape_special_chars(target_dir)
    escaped_target_path = escape_special_chars(os.path.join(target_dir, new_filename))
    
    # Prepare paths for Windows batch script
    # Convert forward slashes to backslashes for Windows paths
    win_source = source_path.replace('/', '\\')
    win_target_dir_path = target_dir.replace('/', '\\')
    win_target_full = os.path.join(target_dir, new_filename).replace('/', '\\')
    
    # Add quotes for Windows paths
    win_source_path = '"' + win_source + '"'
    win_target_dir = '"' + win_target_dir_path + '"'
    win_target_path = '"' + win_target_full + '"'

    # Determine the corresponding text file paths
    if output_dir:
        # If output_dir is specified, text files are in that directory
        txt_source_path = os.path.join(output_dir, os.path.splitext(os.path.basename(source_path))[0] + ".txt")
    else:
        # Otherwise, text files are in the same directory as the source files
        txt_source_path = os.path.splitext(source_path)[0] + ".txt"
        
    # Target text file will be in the target directory with related name
    txt_new_filename = os.path.splitext(new_filename)[0] + ".txt"
    txt_target_path = os.path.join(target_dir, txt_new_filename)
    
    # Escape for bash
    escaped_txt_source_path = escape_special_chars(txt_source_path)
    escaped_txt_target_path = escape_special_chars(txt_target_path)
    
    # Prepare for Windows batch
    win_txt_source = txt_source_path.replace('/', '\\')
    win_txt_target = txt_target_path.replace('/', '\\')
    
    # Add quotes for Windows text paths
    win_txt_source_path = '"' + win_txt_source + '"'
    win_txt_target_path = '"' + win_txt_target + '"'
    
    # Write to the bash script if needed
    if create_bash:
        with file_lock:
            with open(rename_script_path, "a") as bash_file:
                # Create the target directory
                bash_file.write(f"mkdir -p {escaped_target_dir}\n")
                
                # Move the original file
                bash_file.write(f"mv {escaped_source_path} {escaped_target_path}\n")
                
                # Check if the corresponding text file exists and move it too
                bash_file.write(f"# Also move the text file if it exists\n")
                bash_file.write(f"if [ -f {escaped_txt_source_path} ]; then\n")
                bash_file.write(f"  mv {escaped_txt_source_path} {escaped_txt_target_path}\n")
                bash_file.write(f"fi\n\n")
                bash_file.flush()
    
    # Write to the Windows batch script if needed
    if create_batch:
        with file_lock:
            with open(batch_script_path, "a") as batch_file:
                # Create the target directory (mkdir in Windows automatically creates parent dirs)
                batch_file.write(f"if not exist {win_target_dir} mkdir {win_target_dir}\n")
                
                # Move the original file
                batch_file.write(f"move {win_source_path} {win_target_path}\n")
                
                # Check if the corresponding text file exists and move it too
                batch_file.write(f"rem Also move the text file if it exists\n")
                batch_file.write(f"if exist {win_txt_source_path} (\n")
                batch_file.write(f"  move {win_txt_source_path} {win_txt_target_path}\n")
                batch_file.write(f")\n\n")
                batch_file.flush()
    
    logging.debug(f"Added rename command: {source_path} -> {os.path.join(target_dir, new_filename)}")
    if os.path.exists(txt_source_path):
        logging.debug(f"Will also move text file: {txt_source_path} -> {txt_target_path}")
    
    return {
        'bash_script': rename_script_path if create_bash else None,
        'batch_script': batch_script_path if create_batch else None,
        'target_dir': target_dir,
        'new_path': os.path.join(target_dir, new_filename)
    }

def detect_language(text, min_text_length=100, max_sample_length=1000, verbose=False):
    """
    Detect language of text using multiple fallback methods.
    
    Args:
        text: Text to analyze
        min_text_length: Minimum text length to attempt detection
        max_sample_length: Maximum sample length to use
        verbose: Whether to print debug information
    
    Returns:
        str: ISO 639-1 two-letter language code (e.g., 'en', 'de', 'fr') or None if detection fails
    """
    if not text or len(text.strip()) < min_text_length:
        if verbose:
            print(f"Text too short for reliable language detection ({len(text) if text else 0} chars)")
        return None
    
    # Use a sample of text to avoid processing very large documents
    sample = text[:max_sample_length].strip()
    
    # Try langdetect first (fastest and most reliable)
    try:
        from langdetect import detect, DetectorFactory, LangDetectException
        # Make detection deterministic
        DetectorFactory.seed = 0
        try:
            return detect(sample)
        except LangDetectException as e:
            if verbose:
                print(f"langdetect failed: {str(e)}")
    except ImportError:
        if verbose:
            print("langdetect not available")
    
    # Try langid second (good balance of speed and accuracy)
    try:
        import langid
        lang, _ = langid.classify(sample)
        return lang
    except ImportError:
        if verbose:
            print("langid not available")
    
    # Try cld3 third (neural network-based, good for short text)
    try:
        import cld3
        prediction = cld3.get_language(sample)
        if prediction.is_reliable:
            return prediction.language
    except ImportError:
        if verbose:
            print("cld3 not available")
    
    # Final fallback: check for common language patterns
    return _fallback_detect_language(sample)

def _fallback_detect_language(text):
    """
    Very basic fallback language detection based on character patterns.
    Only detects a few common languages with distinctive characters.
    
    Args:
        text: Text to analyze
        
    Returns:
        str: ISO 639-1 language code or None
    """
    # Count characters in specific Unicode ranges
    counts = {
        'en': 0,  # English/Latin
        'zh': 0,  # Chinese
        'ja': 0,  # Japanese
        'ko': 0,  # Korean
        'ru': 0,  # Russian/Cyrillic
        'ar': 0,  # Arabic
        'de': 0,  # German
    }
    
    # Common German words that help distinguish from English
    german_words = ['der', 'die', 'das', 'und', 'ist', 'von', 'zu', 'mit', 'sich', 'auf', 'für', 'ein', 'eine']
    german_word_count = sum(1 for word in text.lower().split() if word in german_words)
    
    for char in text:
        # Chinese characters
        if '\u4e00' <= char <= '\u9fff':
            counts['zh'] += 1
        # Japanese-specific characters (Hiragana & Katakana)
        elif '\u3040' <= char <= '\u30ff':
            counts['ja'] += 1
        # Korean Hangul
        elif '\uac00' <= char <= '\ud7a3':
            counts['ko'] += 1
        # Russian/Cyrillic
        elif '\u0400' <= char <= '\u04ff':
            counts['ru'] += 1
        # Arabic
        elif '\u0600' <= char <= '\u06ff':
            counts['ar'] += 1
        # Latin alphabet (English, German, etc.)
        elif 'a' <= char.lower() <= 'z':
            counts['en'] += 1
    
    # Adjust for German if we see German-specific words
    if german_word_count > 5 and counts['en'] > 0:
        return 'de'
    
    # Return the language with the most character matches
    if any(counts.values()):
        # Get the language with highest count, excluding English initially
        non_latin = {k: v for k, v in counts.items() if k != 'en'}
        if non_latin and max(non_latin.values()) > 0:
            return max(non_latin.items(), key=lambda x: x[1])[0]
        # If no non-Latin characters detected, return English
        return 'en'
    
    return None

def escape_special_chars(filename):
    """
    Safely escape special characters in filenames for shell commands.
    """
    try:
        import shlex
        return shlex.quote(filename)
    except ImportError:
        logging.warning("shlex module not available. Falling back to regex-based escaping.")
        return re.sub(r'([$`"\\])', r'\\\1', filename)
    
def get_openai_client():
    """Initialize or return thread-local OpenAI client for Ollama"""
    if not hasattr(thread_local, "client"):
        try:
            # Import within the function to ensure it's available
            from openai import OpenAI
            
            thread_local.client = OpenAI(
                base_url="http://localhost:11434/v1/",
                api_key="ollama"
            )
            logging.debug("OpenAI client initialized successfully for thread.")
        except ImportError as e:
            logging.critical(f"OpenAI client not available: {e}")
            logging.critical("Please install the OpenAI client: pip install openai")
            raise
        except Exception as e:
            logging.critical(f"Failed to initialize OpenAI client in thread: {e}")
            raise
    return thread_local.client

def main():
    import glob
    
    """Command-line interface entry point"""
    parser = argparse.ArgumentParser(
        description="Document Text Extraction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              %(prog)s input.pdf
              %(prog)s -o output_dir/ *.pdf
              %(prog)s -m pymupdf -p password input.pdf
              %(prog)s -t -j output.json *.pdf
              %(prog)s --noskip input.pdf  # Process even if output exists
              %(prog)s --sort *.pdf  # Sort and rename files based on content
              %(prog)s --sort --execute-rename *.pdf  # Sort and immediately execute rename commands
        """)
    )

    # Add sort argument to the argparse parser in main()
    parser.add_argument(
        '--sort',
        action='store_true',
        help="Sort and rename files based on content analysis with Ollama"
    )

    # Add related arguments for renaming control
    parser.add_argument(
        '--execute-rename',
        action='store_true',
        help="Automatically execute the generated rename commands"
    )

    parser.add_argument(
        '--rename-script',
        default="rename_commands.sh",
        help="Path to write the rename commands (default: rename_commands.sh)"
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help="Input files to process (supports wildcards)"
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        help="Output directory for extracted text files (default: current directory)",
        default='.'  # Default to current directory
    )
    
    parser.add_argument(
        '-m', '--method',
        help="Preferred extraction method"
    )

    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help="Process files recursively through subdirectories"
    )
    
    parser.add_argument(
        '-p', '--password',
        help="Password for encrypted documents"
    )

    parser.add_argument(
        '--ocr-method',
        choices=['auto', 'tesseract', 'doctr', 'easyocr', 'kraken'],
        default='auto',
        help="Preferred OCR method when text extraction is needed"
    )
    
    parser.add_argument(
        '-t', '--tables',
        action='store_true',
        help="Extract tables (PDF only)"
    )
    
    parser.add_argument(
        '-j', '--json',
        help="Save results to JSON file"
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        help="Maximum number of worker threads"
    )
    
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help="Enable debug logging"
    )
    
    parser.add_argument(
        '--noskip',
        action='store_true',
        help="Process files even if output text file already exists"
    )

    # Add to the argument parser in main()
    parser.add_argument(
        '--llm-provider',
        choices=['ollama', 'groq', 'cohere', 'openai', 'glhf', 'huggingface'],
        default='ollama',
        help="LLM provider to use for metadata extraction (default: ollama)"
    )

    parser.add_argument(
        '--llm-model',
        help="Model name to use with the LLM provider"
    )

    parser.add_argument(
        '--api-key',
        help="API key for cloud LLM providers (Groq, Poe)"
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5,
        help="Temperature setting for LLM generation (0.0-1.0)"
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=250,
        help="Maximum tokens in LLM response"
    )

    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    try:
        # Expand file patterns
        input_files = []
        if args.recursive:
            for pattern in args.files:
                # If the pattern is an existing directory, walk it recursively
                if os.path.isdir(pattern):
                    # Directory walking code unchanged
                    ...
                # Check if the pattern is an existing file (handles spaces in filenames)
                elif os.path.isfile(pattern):
                    input_files.append(pattern)
                else:
                    # Pattern globbing for wildcards
                    matched_files = glob.glob(pattern, recursive=True)
                    if matched_files:
                        input_files.extend(matched_files)
                    else:
                        logging.warning(f"No files found matching pattern: {pattern}")
        else:
            for pattern in args.files:
                # Check if the pattern is an existing file (handles spaces in filenames)
                if os.path.isfile(pattern):
                    input_files.append(pattern)
                else:
                    matched_files = glob.glob(pattern)
                    if matched_files:
                        input_files.extend(matched_files)
                    else:
                        logging.warning(f"No files found matching pattern: {pattern}")
        
        if not input_files:
            logging.error("No input files found")
            return 1
            
        # Initialize processor
        processor = DocumentProcessor(debug=args.debug)
        
        # Initialize Ollama client and rename script if sorting is enabled
        openai_client = None
        llm_provider = None
        rename_script_path = None
        
        if args.sort:
            try:
                if args.llm_provider == "ollama":
                    # For Ollama, we keep the existing behavior
                    try:
                        from openai import OpenAI
                        OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
                        logging.info("OpenAI client for Ollama is available")
                    except ImportError:
                        logging.error("OpenAI client not available. Install with 'pip install openai'")
                        logging.error("Proceeding without sorting functionality")
                        args.sort = False
                    except Exception as e:
                        logging.error(f"Error initializing OpenAI client: {e}")
                        logging.error("Proceeding without sorting functionality")
                        args.sort = False
                else:
                    # For other providers, use the new provider factory
                    try:
                        llm_provider = get_llm_provider(
                            provider_type=args.llm_provider,
                            model_name=args.llm_model,
                            api_key=args.api_key
                        )
                        logging.info(f"Initialized {args.llm_provider} LLM provider")
                    except ImportError as e:
                        logging.error(f"Required libraries not available for {args.llm_provider}: {e}")
                        logging.error("Proceeding without sorting functionality")
                        args.sort = False
                    except Exception as e:
                        logging.error(f"Error initializing LLM provider: {e}")
                        logging.error("Proceeding without sorting functionality")
                        args.sort = False
                
                if args.sort:  # Check again in case we disabled it due to errors
                    # Initialize rename script
                    rename_script_path = args.rename_script
                    script_paths = initialize_rename_scripts(rename_script_path)
            except Exception as e:
                logging.error(f"Error initializing sorting functionality: {e}")
                args.sort = False
                llm_provider = None
                rename_script_path = None
        
        try:
            # Process files with periodic shutdown checks
            # Note: we don't set up signal handling here anymore - 
            # it's done globally outside this function

            # Pass sort-related parameters to process_files
            results = processor.process_files(
                input_files,
                output_dir=args.output_dir,
                method=args.method,
                ocr_method=args.ocr_method,
                password=args.password,
                extract_tables=args.tables,
                max_workers=args.workers,
                noskip=args.noskip,
                sort=args.sort,                     # sorting parameter
                rename_script_path=rename_script_path,  # sorting parameter
                llm_provider=llm_provider,
                temperature=args.temperature,     
                max_tokens=args.max_tokens,        
            )
            
            # Handle results
            if args.json:
                import json
                with open(args.json, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Handle rename script if sorting was enabled
            if args.sort and args.execute_rename:
                is_windows = platform.system() == 'Windows'
                
                if is_windows:
                    batch_script_path = os.path.splitext(rename_script_path)[0] + '.bat'
                    if os.path.exists(batch_script_path):
                        logging.info("Executing rename commands using batch script...")
                        try:
                            subprocess.run(['cmd', '/c', batch_script_path], check=True)
                            logging.info("Successfully executed rename commands")
                        except subprocess.CalledProcessError as e:
                            logging.error(f"Error executing rename commands: {e}")
                    else:
                        logging.error(f"Batch script {batch_script_path} not found")
                else:
                    # Unix execution (unchanged)
                    logging.info("Executing rename commands...")
                    execute_rename_commands(rename_script_path)
            elif args.sort:
                # Just print instructions
                if platform.system() == 'Windows':
                    batch_script_path = os.path.splitext(rename_script_path)[0] + '.bat'
                    logging.info(f"Rename commands written to {rename_script_path} and {batch_script_path}")
                    logging.info(f"Review and execute manually with: bash {rename_script_path}")
                    logging.info(f"  or on Windows: {batch_script_path}")
                else:
                    logging.info(f"Rename commands written to {rename_script_path}")
                    logging.info(f"Review and execute manually with: bash {rename_script_path}")
            
            # Update return code logic to include skipped files in the summary
            successful = len(results.get('results', {})) - len(results.get('failed', []))
            skipped = len(results.get('skipped', []))
            
            if args.debug:
                logging.info(f"Summary: {successful} succeeded, {skipped} skipped, {len(results.get('failed', []))} failed")
            
            return 0 if not results.get('failed') else 1
            
        except KeyboardInterrupt:
            if shutdown_flag.is_set():
                # Proper handling after graceful shutdown flag was set
                if args.sort and rename_script_path:
                    try:
                        os.chmod(rename_script_path, 0o755)
                        logging.info(f"Made rename script executable: {rename_script_path}")
                        logging.info(f"Review and execute manually with: bash {rename_script_path}")
                    except Exception as e:
                        logging.error(f"Error setting permissions on rename script: {e}")
                logging.info("Operation gracefully terminated after interrupt")
            else:
                # Direct KeyboardInterrupt without going through signal handler
                print("\nOperation cancelled by user")
                if args.sort and rename_script_path:
                    try:
                        os.chmod(rename_script_path, 0o755)
                        logging.info(f"Made rename script executable: {rename_script_path}")
                        logging.info(f"You can still use the partial rename script: bash {rename_script_path}")
                    except Exception:
                        pass
            return 130
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
