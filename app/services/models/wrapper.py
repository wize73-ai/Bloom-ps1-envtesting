"""
Model Wrappers Module for CasaLingua
Provides standardized interfaces between models and the pipeline

These wrappers ensure that all models, regardless of their underlying implementation,
expose a consistent interface to the pipeline components. They also apply
model-specific optimizations and handle specialized processing.
"""

import os
import logging
import time
import functools
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import torch
import numpy as np
from enum import Enum

# Import from wrapper_base to avoid circular imports
from app.services.models.wrapper_base import BaseModelWrapper, ModelInput, ModelOutput

# Configure logging
logger = logging.getLogger(__name__)

# Model types
class ModelType(str, Enum):
    TRANSLATION = "translation"
    LANGUAGE_DETECTION = "language_detection"
    NER_DETECTION = "ner_detection"
    SIMPLIFIER = "simplifier"
    RAG_GENERATOR = "rag_generator"
    RAG_RETRIEVER = "rag_retriever"
    ANONYMIZER = "anonymizer"
    
# Define wrapper_map at module level so it can be accessed by fix_circular_import
wrapper_map = {}  # Will be initialized in get_wrapper_for_model

class TranslationModelWrapper(BaseModelWrapper):
    """Wrapper for translation models"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess input for translation model"""
        # Basic input validation
        if not input_data.text:
            logger.warning("Empty input text for translation")
            return {"error": "Empty input text"}
        
        source_lang = input_data.source_language
        target_lang = input_data.target_language
        
        # Handle parameters
        parameters = input_data.parameters or {}
        domain = parameters.get("domain", None)
        
        # Check for legal domain - Add null check to prevent NoneType error
        is_legal_domain = domain and domain.lower() in ["legal", "housing", "housing_legal"]
        
        # Get model configuration for token limits
        max_length = self.config.get("max_length", 512)
        
        # Prepare tokenizer inputs
        if self.tokenizer is None:
            logger.error("No tokenizer available for translation")
            return {"error": "No tokenizer available"}
        
        try:
            # Check if model is MBART or MT5
            is_mbart = "mbart" in str(self.model.__class__).lower()
            is_mt5 = "mt5" in str(self.model.__class__).lower()
            
            # Special handling for MBART
            if is_mbart:
                # MBART requires special tokens for languages
                src_lang_code = self._get_mbart_lang_code(source_lang)
                tgt_lang_code = self._get_mbart_lang_code(target_lang)
                
                # Apply MBART-specific preprocessing
                tokenizer_kwargs = {
                    "padding": True,
                    "truncation": True,
                    "return_tensors": "pt",
                    "max_length": max_length,
                }
                
                # Handle source and target language codes
                if hasattr(self.tokenizer, "src_lang") and hasattr(self.tokenizer, "tgt_lang"):
                    self.tokenizer.src_lang = src_lang_code
                    self.tokenizer.tgt_lang = tgt_lang_code
                    logger.debug(f"Set MBART langs: src={src_lang_code}, tgt={tgt_lang_code}")
                
                # Tokenize with MBART config
                inputs = self.tokenizer(input_data.text, **tokenizer_kwargs)
                
                # Add language IDs if needed
                if hasattr(self.tokenizer, "lang_code_to_id"):
                    forced_bos_token_id = self.tokenizer.lang_code_to_id.get(tgt_lang_code)
                    if forced_bos_token_id is not None:
                        logger.debug(f"Added forced BOS token ID: {forced_bos_token_id}")
                        return {
                            "inputs": inputs,
                            "forced_bos_token_id": forced_bos_token_id,
                            "is_mbart": True
                        }
                
                return {
                    "inputs": inputs,
                    "is_mbart": True
                }
                
            # Special handling for MT5
            elif is_mt5:
                # MT5 requires prefixed task description
                prefix = f"translate {source_lang} to {target_lang}: "
                text_with_prefix = prefix + input_data.text
                
                # Tokenize with MT5 config
                inputs = self.tokenizer(
                    text_with_prefix,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length
                )
                
                return {
                    "inputs": inputs,
                    "is_mt5": True
                }
                
            # Generic transformer models
            else:
                # For other models, use standard tokenization
                inputs = self.tokenizer(
                    input_data.text,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length
                )
                
                return {
                    "inputs": inputs,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "domain": domain
                }
                
        except Exception as e:
            logger.error(f"Error preprocessing input for translation: {str(e)}", exc_info=True)
            # Return error information for handling in inference step
            return {
                "error": f"Preprocessing error: {str(e)}",
                "source_text": input_data.text,
                "source_lang": source_lang,
                "target_lang": target_lang
            }

    def _get_mbart_lang_code(self, language: str) -> str:
        """Get the correct language code for MBART"""
        # MBART uses special language codes
        mbart_lang_map = {
            "ar": "ar_AR",
            "cs": "cs_CZ",
            "de": "de_DE",
            "en": "en_XX",
            "es": "es_XX",
            "et": "et_EE",
            "fi": "fi_FI",
            "fr": "fr_XX",
            "gu": "gu_IN",
            "hi": "hi_IN",
            "it": "it_IT",
            "ja": "ja_XX",
            "kk": "kk_KZ",
            "ko": "ko_KR",
            "lt": "lt_LT",
            "lv": "lv_LV",
            "my": "my_MM",
            "ne": "ne_NP",
            "nl": "nl_XX",
            "ro": "ro_RO",
            "ru": "ru_RU",
            "si": "si_LK",
            "tr": "tr_TR",
            "vi": "vi_VN",
            "zh": "zh_CN"
        }
        
        return mbart_lang_map.get(language, f"{language}_XX")
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run inference with the translation model"""
        # Check for preprocessing errors
        if "error" in preprocessed:
            logger.error(f"Cannot run inference due to preprocessing error: {preprocessed['error']}")
            return {"error": preprocessed["error"]}
        
        # Get generation parameters
        inputs = preprocessed.get("inputs")
        if inputs is None:
            logger.error("No inputs provided for translation inference")
            return {"error": "No inputs provided"}
        
        # Move inputs to device
        for key in inputs:
            if hasattr(inputs[key], "to") and callable(inputs[key].to):
                inputs[key] = inputs[key].to(self.device)
        
        # Prepare generation kwargs
        generation_kwargs = {
            "max_length": self.config.get("max_length", 512),
            "num_beams": self.config.get("num_beams", 4),
            "temperature": self.config.get("temperature", 1.0),
            "no_repeat_ngram_size": self.config.get("no_repeat_ngram_size", 3),
            "early_stopping": True,
        }
        
        # Add special handling for MBART
        is_mbart = preprocessed.get("is_mbart", False)
        is_mt5 = preprocessed.get("is_mt5", False)
        
        if is_mbart and "forced_bos_token_id" in preprocessed:
            generation_kwargs["forced_bos_token_id"] = preprocessed["forced_bos_token_id"]
        
        # Run inference
        try:
            with torch.no_grad():
                # Generate translation
                outputs = self.model.generate(**inputs, **generation_kwargs)
                return outputs
                
        except Exception as e:
            logger.error(f"Error during model generation: {str(e)}", exc_info=True)
            # Return an empty output as fallback
            if is_mt5:
                logger.warning("MT5 generation failed, returning empty output")
                # Create a dummy output with a special token indicating an error
                if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
                    return torch.tensor([[self.tokenizer.pad_token_id]])
                else:
                    return torch.tensor([[0]])  # Fallback to 0 if no pad token
            elif is_mbart:
                logger.warning("MBART generation failed, returning empty output")
                if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
                    return torch.tensor([[self.tokenizer.pad_token_id]])
                else:
                    return torch.tensor([[0]])  # Fallback to 0 if no pad token
            
            # Try to get a meaningful error message
            error_msg = f"Generation failed: {str(e)}"
            logger.error(error_msg)
            
            # Return a special error token or sequence
            return {"error": error_msg}
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess the model output to get the translated text"""
        # Check for errors from inference
        if isinstance(model_output, dict) and "error" in model_output:
            logger.error(f"Cannot postprocess due to inference error: {model_output['error']}")
            return ModelOutput(
                result="",
                metadata={
                    "error": model_output["error"],
                    "source_language": input_data.source_language,
                    "target_language": input_data.target_language
                }
            )
        
        # Get parameters
        parameters = input_data.parameters or {}
        
        try:
            # Decode the model output
            if self.tokenizer is None:
                logger.error("No tokenizer available for postprocessing")
                return ModelOutput(
                    result="",
                    metadata={"error": "No tokenizer available"}
                )
            
            # Handle tensor output
            if isinstance(model_output, torch.Tensor):
                # Skip special tokens in the output
                translated_text = self.tokenizer.decode(
                    model_output[0], 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            else:
                logger.warning(f"Unexpected output type: {type(model_output)}")
                translated_text = str(model_output)
            
            # Check for empty or invalid output and provide fallback
            if not translated_text or translated_text.isspace():
                logger.warning("Empty translation result, using original text as fallback")
                return ModelOutput(
                    result=input_data.text,  # Return original text as fallback
                    metadata={
                        "error": "Empty translation result",
                        "source_language": input_data.source_language,
                        "target_language": input_data.target_language,
                        "fallback": True
                    }
                )
            
            # Post-processing for MT5
            is_mt5 = "mt5" in str(self.model.__class__).lower()
            if is_mt5:
                # Sometimes MT5 repeats the input prefix in the output
                prefix = f"translate {input_data.source_language} to {input_data.target_language}: "
                if translated_text.startswith(prefix):
                    translated_text = translated_text[len(prefix):]
            
            # Return the processed translation
            return ModelOutput(
                result=translated_text,
                metadata={
                    "source_language": input_data.source_language,
                    "target_language": input_data.target_language,
                    "model_type": "mbart" if "mbart" in str(self.model.__class__).lower() else
                               "mt5" if is_mt5 else
                               "transformer"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in translation postprocessing: {str(e)}", exc_info=True)
            # Return an error result
            return ModelOutput(
                result="",
                metadata={
                    "error": f"Postprocessing error: {str(e)}",
                    "source_language": input_data.source_language,
                    "target_language": input_data.target_language
                }
            )

class LanguageDetectionWrapper(BaseModelWrapper):
    """Wrapper for language detection models"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess input for language detection"""
        # Basic input validation
        if not input_data.text:
            logger.warning("Empty input text for language detection")
            return {"error": "Empty input text"}
        
        # Prepare tokenizer inputs
        if self.tokenizer is None:
            logger.error("No tokenizer available for language detection")
            return {"error": "No tokenizer available"}
        
        try:
            # Get model configuration for token limits
            max_length = self.config.get("max_length", 512)
            
            # Tokenize with standard config
            inputs = self.tokenizer(
                input_data.text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length
            )
            
            return {
                "inputs": inputs,
                "original_text": input_data.text
            }
                
        except Exception as e:
            logger.error(f"Error preprocessing input for language detection: {str(e)}", exc_info=True)
            # Return error information for handling in inference step
            return {
                "error": f"Preprocessing error: {str(e)}",
                "original_text": input_data.text
            }
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run inference with the language detection model"""
        # Check for preprocessing errors
        if "error" in preprocessed:
            logger.error(f"Cannot run inference due to preprocessing error: {preprocessed['error']}")
            return {"error": preprocessed["error"]}
        
        # Get inputs
        inputs = preprocessed.get("inputs")
        if inputs is None:
            logger.error("No inputs provided for language detection inference")
            return {"error": "No inputs provided"}
        
        # Move inputs to device
        for key in inputs:
            if hasattr(inputs[key], "to") and callable(inputs[key].to):
                inputs[key] = inputs[key].to(self.device)
        
        # Run inference
        try:
            with torch.no_grad():
                # Get model outputs
                outputs = self.model(**inputs)
                return outputs
                
        except Exception as e:
            logger.error(f"Error during language detection: {str(e)}", exc_info=True)
            # Return an error
            error_msg = f"Language detection failed: {str(e)}"
            logger.error(error_msg)
            
            # Return a special error token or sequence
            return {"error": error_msg}
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess the model output to get the detected language"""
        # Check for errors from inference
        if isinstance(model_output, dict) and "error" in model_output:
            logger.error(f"Cannot postprocess due to inference error: {model_output['error']}")
            return ModelOutput(
                result={"language": "en", "confidence": 0.0},  # Default to English with zero confidence
                metadata={"error": model_output["error"]}
            )
        
        try:
            # Process XLM-RoBERTa language detection model output
            if hasattr(model_output, "logits"):
                logits = model_output.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
                
                # Get the language labels if available
                id2label = getattr(self.model.config, "id2label", None)
                
                if id2label:
                    # Get highest probability language
                    max_prob_idx = torch.argmax(probabilities).item()
                    detected_lang = id2label[max_prob_idx]
                    confidence = probabilities[max_prob_idx].item()
                    
                    # Clean up language code if needed
                    if "_" in detected_lang:
                        detected_lang = detected_lang.split("_")[0]
                    
                    # Return structured result
                    return ModelOutput(
                        result={"language": detected_lang, "confidence": confidence},
                        metadata={"model_type": "xlm-roberta"}
                    )
                else:
                    # If no labels are available
                    logger.warning("No language labels available in model configuration")
                    return ModelOutput(
                        result={"language": "en", "confidence": 0.0},
                        metadata={"error": "No language labels available"}
                    )
            else:
                # Unknown model output format
                logger.warning(f"Unexpected output type for language detection: {type(model_output)}")
                return ModelOutput(
                    result={"language": "en", "confidence": 0.0},
                    metadata={"error": "Unsupported model output format"}
                )
            
        except Exception as e:
            logger.error(f"Error in language detection postprocessing: {str(e)}", exc_info=True)
            # Return an error result with English fallback
            return ModelOutput(
                result={"language": "en", "confidence": 0.0},
                metadata={"error": f"Postprocessing error: {str(e)}"}
            )

class SimplificationModelWrapper(BaseModelWrapper):
    """Wrapper for text simplification models"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess input for simplification model"""
        # Basic input validation
        if not input_data.text:
            logger.warning("Empty input text for simplification")
            return {"error": "Empty input text"}
        
        # Get parameters
        parameters = input_data.parameters or {}
        language = input_data.source_language or "en"
        target_level = parameters.get("target_level", "simple")
        
        # Handle numeric level conversion
        level = 1  # Default level
        if isinstance(target_level, str):
            if target_level.isdigit():
                level = int(target_level)
            elif target_level.lower() == "simple":
                level = 1
            elif target_level.lower() == "moderate":
                level = 3
            elif target_level.lower() == "complex":
                level = 5
        elif isinstance(target_level, int):
            level = target_level
        
        # Prepare tokenizer inputs
        if self.tokenizer is None:
            logger.error("No tokenizer available for simplification")
            return {"error": "No tokenizer available"}
        
        try:
            # Get model configuration for token limits
            max_length = self.config.get("max_length", 512)
            
            # Prepare prefix for instruction-tuned models
            prefix = f"simplify to level {level}: "
            text_with_prefix = prefix + input_data.text
            
            # Tokenize with standard config
            inputs = self.tokenizer(
                text_with_prefix,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length
            )
            
            return {
                "inputs": inputs,
                "original_text": input_data.text,
                "language": language,
                "level": level
            }
                
        except Exception as e:
            logger.error(f"Error preprocessing input for simplification: {str(e)}", exc_info=True)
            # Return error information for handling in inference step
            return {
                "error": f"Preprocessing error: {str(e)}",
                "original_text": input_data.text,
                "language": language,
                "level": level
            }
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run inference with the simplification model"""
        # Check for preprocessing errors
        if "error" in preprocessed:
            logger.error(f"Cannot run inference due to preprocessing error: {preprocessed['error']}")
            return {"error": preprocessed["error"]}
        
        # Get inputs
        inputs = preprocessed.get("inputs")
        if inputs is None:
            logger.error("No inputs provided for simplification inference")
            return {"error": "No inputs provided"}
        
        # Move inputs to device
        for key in inputs:
            if hasattr(inputs[key], "to") and callable(inputs[key].to):
                inputs[key] = inputs[key].to(self.device)
        
        # Prepare generation kwargs
        generation_kwargs = {
            "max_length": self.config.get("max_length", 512),
            "num_beams": self.config.get("num_beams", 4),
            "temperature": self.config.get("temperature", 1.0),
            "no_repeat_ngram_size": self.config.get("no_repeat_ngram_size", 3),
            "early_stopping": True,
        }
        
        # Run inference
        try:
            with torch.no_grad():
                # Generate simplified text
                outputs = self.model.generate(**inputs, **generation_kwargs)
                return outputs
                
        except Exception as e:
            logger.error(f"Error during simplification: {str(e)}", exc_info=True)
            # Return an error
            error_msg = f"Simplification failed: {str(e)}"
            logger.error(error_msg)
            
            # Return a special error token or sequence
            return {"error": error_msg}
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess the model output to get the simplified text"""
        # Check for errors from inference
        if isinstance(model_output, dict) and "error" in model_output:
            logger.error(f"Cannot postprocess due to inference error: {model_output['error']}")
            return ModelOutput(
                result="",
                metadata={
                    "error": model_output["error"],
                    "fallback": "original"
                }
            )
        
        # Get parameters
        parameters = input_data.parameters or {}
        
        try:
            # Decode the model output
            if self.tokenizer is None:
                logger.error("No tokenizer available for postprocessing")
                return ModelOutput(
                    result=input_data.text,  # Return original as fallback
                    metadata={"error": "No tokenizer available", "fallback": "original"}
                )
            
            # Handle tensor output
            if isinstance(model_output, torch.Tensor):
                # Skip special tokens in the output
                simplified_text = self.tokenizer.decode(
                    model_output[0], 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            else:
                logger.warning(f"Unexpected output type: {type(model_output)}")
                simplified_text = str(model_output)
            
            # Remove any instruction prefix
            target_level = parameters.get("target_level", "1")
            prefix = f"simplify to level {target_level}: "
            if simplified_text.startswith(prefix):
                simplified_text = simplified_text[len(prefix):]
            
            # Check for empty or invalid output and provide fallback
            if not simplified_text or simplified_text.isspace():
                logger.warning("Empty simplification result, using original text as fallback")
                return ModelOutput(
                    result=input_data.text,  # Return original text as fallback
                    metadata={
                        "error": "Empty simplification result",
                        "fallback": "original"
                    }
                )
            
            # Return the processed simplification
            return ModelOutput(
                result=simplified_text,
                metadata={
                    "language": input_data.source_language or "en",
                    "level": parameters.get("target_level", "1"),
                    "original_length": len(input_data.text),
                    "simplified_length": len(simplified_text)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in simplification postprocessing: {str(e)}", exc_info=True)
            # Return an error result with original text as fallback
            return ModelOutput(
                result=input_data.text,
                metadata={
                    "error": f"Postprocessing error: {str(e)}",
                    "fallback": "original"
                }
            )

class NERDetectionWrapper(BaseModelWrapper):
    """Wrapper for Named Entity Recognition models"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess for NER model"""
        # Implementation details...
        return {"inputs": {}}
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run NER model inference"""
        # Implementation details...
        return {}
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess NER model output"""
        # Implementation details...
        return ModelOutput(result={})

class RAGGeneratorWrapper(BaseModelWrapper):
    """Wrapper for RAG generator models"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess for RAG generator"""
        # Implementation details...
        return {"inputs": {}}
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run RAG generator inference"""
        # Implementation details...
        return {}
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess RAG generator output"""
        # Implementation details...
        return ModelOutput(result={})

class RAGRetrieverWrapper(BaseModelWrapper):
    """Wrapper for RAG retriever models"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess for RAG retriever"""
        # Implementation details...
        return {"inputs": {}}
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run RAG retriever inference"""
        # Implementation details...
        return {}
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess RAG retriever output"""
        # Implementation details...
        return ModelOutput(result={})

class AnonymizerWrapper(BaseModelWrapper):
    """Wrapper for text anonymization models"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess for anonymizer"""
        # Implementation details...
        return {"inputs": {}}
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run anonymizer inference"""
        # Implementation details...
        return {}
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess anonymizer output"""
        # Implementation details...
        return ModelOutput(result={})

def create_model_wrapper(model_type: str, model, tokenizer, config: Dict[str, Any] = None, **kwargs):
    """
    Factory function to create the appropriate wrapper for a model type
    
    Args:
        model_type: Type of model to wrap
        model: The model to wrap
        tokenizer: The tokenizer to use
        config: Configuration parameters
        
    Returns:
        BaseModelWrapper: Appropriate model wrapper
    """
    # Special case for mbart_translation - ALWAYS use TranslationModelWrapper
    if model_type == 'mbart_translation' or 'mbart' in model_type.lower():
        logger.info(f"Using TranslationModelWrapper for MBART model type: {model_type}")
        return TranslationModelWrapper(model, tokenizer, config, **kwargs)
    
    # Special case for translation - ALWAYS use TranslationModelWrapper 
    if model_type == 'translation' or 'translation' in model_type.lower():
        logger.info(f"Using TranslationModelWrapper for translation model type: {model_type}")
        return TranslationModelWrapper(model, tokenizer, config, **kwargs)
        
    # Use wrapper_map for other model types
    if model_type in wrapper_map and wrapper_map[model_type]:
        logger.info(f"Using wrapper from map for model type: {model_type}")
        wrapper_class = wrapper_map[model_type]
        return wrapper_class(model, tokenizer, config, **kwargs)
    
    # Special handling for embedding models
    if model_type == 'embedding_model':
        try:
            from app.services.models.embedding_wrapper import EmbeddingModelWrapper
            return EmbeddingModelWrapper(model, tokenizer, config, **kwargs)
        except ImportError:
            logger.warning(f"EmbeddingModelWrapper not available, using base wrapper for {model_type}")
    
    # Fallback to base wrapper with warning
    logger.warning(f"No specific wrapper for model type: {model_type}, using base wrapper")
    return BaseModelWrapper(model, tokenizer, config, **kwargs)

# Alias for backward compatibility
get_wrapper_for_model = create_model_wrapper