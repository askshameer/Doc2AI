"""
Model-specific templates for fine-tuning dataset generation.
Supports different LLM architectures and training frameworks.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

from core.data_models import Chunk, Document, ConversationTurn


@dataclass
class ModelConfig:
    """Configuration for a specific model type."""
    name: str
    provider: str
    max_tokens: int
    conversation_format: str
    system_message_supported: bool
    special_tokens: Dict[str, str]
    template_type: str


class BaseTemplate(ABC):
    """Abstract base class for model-specific templates."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    def format_conversation(self, chunk: Chunk, document: Document, **kwargs) -> Dict[str, Any]:
        """Format a chunk into a conversation for fine-tuning."""
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for the given text."""
        pass
    
    def validate_entry(self, entry: Dict[str, Any]) -> List[str]:
        """Validate a fine-tuning entry and return warnings."""
        warnings = []
        
        # Check token count
        total_tokens = self.estimate_total_tokens(entry)
        if total_tokens > self.config.max_tokens:
            warnings.append(f"Entry exceeds max tokens: {total_tokens}/{self.config.max_tokens}")
        
        return warnings
    
    def estimate_total_tokens(self, entry: Dict[str, Any]) -> int:
        """Estimate total tokens for an entire entry."""
        if 'messages' in entry:
            return sum(self.estimate_tokens(msg.get('content') or '') for msg in entry['messages'])
        elif 'conversations' in entry:
            return sum(self.estimate_tokens(turn.get('content') or '') for turn in entry['conversations'])
        else:
            return self.estimate_tokens(str(entry))


class OpenAITemplate(BaseTemplate):
    """Template for OpenAI GPT models (GPT-3.5, GPT-4, etc.)."""
    
    def format_conversation(self, chunk: Chunk, document: Document, **kwargs) -> Dict[str, Any]:
        """Format for OpenAI fine-tuning API."""
        system_message = kwargs.get('system_message', 
            "You are a helpful assistant that answers questions based on the provided document content.")
        
        # Generate question based on content
        question = self._generate_question(chunk, document, **kwargs)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
            {"role": "assistant", "content": chunk.text}
        ]
        
        return {
            "messages": messages,
            "metadata": {
                "model_type": "openai",
                "source": document.metadata.filename,
                "chunk_id": chunk.chunk_id
            }
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation for OpenAI models (1 token ≈ 4 characters)."""
        if text is None:
            return 0
        return len(str(text)) // 4
    
    def _generate_question(self, chunk: Chunk, document: Document, **kwargs) -> str:
        """Generate a question based on chunk content."""
        question_type = kwargs.get('question_type', 'general')
        
        if question_type == 'summary':
            return "Please summarize the key points from this content."
        elif question_type == 'explanation':
            return "Can you explain this concept in detail?"
        elif question_type == 'context':
            section = chunk.location.section or "document"
            return f"What information is provided in the {section}?"
        else:
            return "Based on the document content, please provide a comprehensive answer."


class AnthropicTemplate(BaseTemplate):
    """Template for Anthropic Claude models."""
    
    def format_conversation(self, chunk: Chunk, document: Document, **kwargs) -> Dict[str, Any]:
        """Format for Anthropic Claude fine-tuning."""
        system_message = kwargs.get('system_message',
            "You are Claude, a helpful AI assistant. Answer questions based on the provided document content.")
        
        question = self._generate_question(chunk, document, **kwargs)
        
        # Claude format with Human/Assistant prefixes
        conversation = f"Human: {question}\n\nAssistant: {chunk.text}"
        
        return {
            "prompt": conversation,
            "system": system_message,
            "metadata": {
                "model_type": "anthropic",
                "source": document.metadata.filename,
                "chunk_id": chunk.chunk_id
            }
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Token estimation for Claude (similar to GPT but slightly different)."""
        return len(text) // 3.5
    
    def _generate_question(self, chunk: Chunk, document: Document, **kwargs) -> str:
        """Generate questions optimized for Claude's conversation style."""
        question_type = kwargs.get('question_type', 'analytical')
        
        if question_type == 'analytical':
            return "Please analyze and explain the key concepts in this content."
        elif question_type == 'comprehensive':
            return "Provide a comprehensive overview of this information."
        elif question_type == 'contextual':
            return f"Based on this excerpt from {document.metadata.filename}, what are the main points?"
        else:
            return "Please explain this content in a clear and helpful way."


class LlamaTemplate(BaseTemplate):
    """Template for Meta's Llama models."""
    
    def format_conversation(self, chunk: Chunk, document: Document, **kwargs) -> Dict[str, Any]:
        """Format for Llama model fine-tuning."""
        system_message = kwargs.get('system_message',
            "You are a helpful, respectful and honest assistant. Answer questions based on the provided information.")
        
        question = self._generate_question(chunk, document, **kwargs)
        
        # Llama-2 chat format
        conversation = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{question} [/INST] {chunk.text} </s>"
        
        return {
            "text": conversation,
            "metadata": {
                "model_type": "llama",
                "source": document.metadata.filename,
                "chunk_id": chunk.chunk_id
            }
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Token estimation for Llama models."""
        return len(text) // 4
    
    def _generate_question(self, chunk: Chunk, document: Document, **kwargs) -> str:
        """Generate questions suitable for Llama's instruction format."""
        question_type = kwargs.get('question_type', 'instruction')
        
        if question_type == 'instruction':
            return "Please explain the following information clearly and accurately:"
        elif question_type == 'qa':
            return "What does this text tell us?"
        elif question_type == 'summarize':
            return "Summarize the key points from this content:"
        else:
            return "Based on the following information, provide a helpful response:"


class MistralTemplate(BaseTemplate):
    """Template for Mistral AI models."""
    
    def format_conversation(self, chunk: Chunk, document: Document, **kwargs) -> Dict[str, Any]:
        """Format for Mistral model fine-tuning."""
        system_message = kwargs.get('system_message',
            "You are a helpful assistant that provides accurate information based on the given content.")
        
        question = self._generate_question(chunk, document, **kwargs)
        
        # Mistral instruction format
        messages = [
            {"role": "user", "content": f"{system_message}\n\nQuestion: {question}"},
            {"role": "assistant", "content": chunk.text}
        ]
        
        return {
            "messages": messages,
            "metadata": {
                "model_type": "mistral",
                "source": document.metadata.filename,
                "chunk_id": chunk.chunk_id
            }
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Token estimation for Mistral models."""
        return len(text) // 4
    
    def _generate_question(self, chunk: Chunk, document: Document, **kwargs) -> str:
        """Generate questions for Mistral's format."""
        return "Please provide information about the following content:"


class HuggingFaceTemplate(BaseTemplate):
    """Generic template for HuggingFace transformers."""
    
    def format_conversation(self, chunk: Chunk, document: Document, **kwargs) -> Dict[str, Any]:
        """Format for generic HuggingFace models."""
        instruction_template = kwargs.get('instruction_template', 
            "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.")
        
        question = self._generate_question(chunk, document, **kwargs)
        
        return {
            "instruction": instruction_template,
            "input": question,
            "output": chunk.text,
            "metadata": {
                "model_type": "huggingface",
                "source": document.metadata.filename,
                "chunk_id": chunk.chunk_id
            }
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Generic token estimation."""
        return len(text) // 4
    
    def _generate_question(self, chunk: Chunk, document: Document, **kwargs) -> str:
        """Generate instruction-style prompts."""
        return f"Extract and explain the key information from this document section."


# Model configurations
MODEL_CONFIGS = {
    "gpt-3.5-turbo": ModelConfig(
        name="GPT-3.5 Turbo",
        provider="OpenAI",
        max_tokens=4096,
        conversation_format="messages",
        system_message_supported=True,
        special_tokens={},
        template_type="openai"
    ),
    "gpt-4": ModelConfig(
        name="GPT-4",
        provider="OpenAI", 
        max_tokens=8192,
        conversation_format="messages",
        system_message_supported=True,
        special_tokens={},
        template_type="openai"
    ),
    "claude-3-haiku": ModelConfig(
        name="Claude 3 Haiku",
        provider="Anthropic",
        max_tokens=200000,
        conversation_format="prompt",
        system_message_supported=True,
        special_tokens={"human": "Human:", "assistant": "Assistant:"},
        template_type="anthropic"
    ),
    "claude-3-sonnet": ModelConfig(
        name="Claude 3 Sonnet",
        provider="Anthropic",
        max_tokens=200000,
        conversation_format="prompt",
        system_message_supported=True,
        special_tokens={"human": "Human:", "assistant": "Assistant:"},
        template_type="anthropic"
    ),
    "llama-2-7b": ModelConfig(
        name="Llama 2 7B",
        provider="Meta",
        max_tokens=4096,
        conversation_format="text",
        system_message_supported=True,
        special_tokens={"bos": "<s>", "eos": "</s>", "inst_start": "[INST]", "inst_end": "[/INST]"},
        template_type="llama"
    ),
    "llama-2-13b": ModelConfig(
        name="Llama 2 13B",
        provider="Meta",
        max_tokens=4096,
        conversation_format="text",
        system_message_supported=True,
        special_tokens={"bos": "<s>", "eos": "</s>", "inst_start": "[INST]", "inst_end": "[/INST]"},
        template_type="llama"
    ),
    "mistral-7b": ModelConfig(
        name="Mistral 7B",
        provider="Mistral AI",
        max_tokens=8192,
        conversation_format="messages",
        system_message_supported=False,
        special_tokens={},
        template_type="mistral"
    ),
    "custom-hf": ModelConfig(
        name="Custom HuggingFace",
        provider="HuggingFace",
        max_tokens=2048,
        conversation_format="instruction",
        system_message_supported=True,
        special_tokens={},
        template_type="huggingface"
    )
}

# Template classes mapping
TEMPLATE_CLASSES = {
    "openai": OpenAITemplate,
    "anthropic": AnthropicTemplate, 
    "llama": LlamaTemplate,
    "mistral": MistralTemplate,
    "huggingface": HuggingFaceTemplate
}


class ModelTemplateManager:
    """Manager class for handling different model templates."""
    
    def __init__(self):
        self.templates = {}
        for model_id, config in MODEL_CONFIGS.items():
            template_class = TEMPLATE_CLASSES[config.template_type]
            self.templates[model_id] = template_class(config)
    
    def get_template(self, model_id: str) -> BaseTemplate:
        """Get template for a specific model."""
        if model_id not in self.templates:
            raise ValueError(f"Unsupported model: {model_id}")
        return self.templates[model_id]
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with their configurations."""
        return [
            {
                "id": model_id,
                "name": config.name,
                "provider": config.provider,
                "max_tokens": config.max_tokens,
                "supports_system": config.system_message_supported
            }
            for model_id, config in MODEL_CONFIGS.items()
        ]
    
    def format_for_model(self, model_id: str, chunk: Chunk, document: Document, **kwargs) -> Dict[str, Any]:
        """Format a chunk for a specific model."""
        template = self.get_template(model_id)
        return template.format_conversation(chunk, document, **kwargs)
    
    def validate_for_model(self, model_id: str, entry: Dict[str, Any]) -> List[str]:
        """Validate an entry for a specific model."""
        template = self.get_template(model_id)
        return template.validate_entry(entry)
    
    def get_model_info(self, model_id: str) -> ModelConfig:
        """Get detailed information about a model."""
        if model_id not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_id}")
        return MODEL_CONFIGS[model_id]
    
    def add_custom_model(
        self, 
        model_id: str, 
        name: str, 
        provider: str,
        max_tokens: int,
        template_type: str,
        **kwargs
    ):
        """Add a custom model configuration."""
        if template_type not in TEMPLATE_CLASSES:
            raise ValueError(f"Unknown template type: {template_type}")
        
        config = ModelConfig(
            name=name,
            provider=provider,
            max_tokens=max_tokens,
            conversation_format=kwargs.get('conversation_format', 'messages'),
            system_message_supported=kwargs.get('system_message_supported', True),
            special_tokens=kwargs.get('special_tokens', {}),
            template_type=template_type
        )
        
        MODEL_CONFIGS[model_id] = config
        template_class = TEMPLATE_CLASSES[template_type]
        self.templates[model_id] = template_class(config)