"""
Text Simplification Pipeline for CasaLingua

This module handles text simplification to different levels of complexity,
with support for target grade levels and domain-specific simplification.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import re
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from app.utils.logging import get_logger

logger = get_logger("casalingua.core.simplifier")

class SimplificationPipeline:
    """
    Text simplification pipeline.
    
    Features:
    - Multiple simplification levels
    - Grade-level targeting
    - Context-aware simplification
    - Domain-specific simplification
    """
    
    def __init__(
        self,
        model_manager,
        config: Dict[str, Any] = None,
        registry_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the simplification pipeline.
        
        Args:
            model_manager: Model manager for accessing simplification models
            config: Configuration dictionary
            registry_config: Registry configuration for models
        """
        self.model_manager = model_manager
        self.config = config or {}
        self.registry_config = registry_config or {}
        self.initialized = False
        
        # Model type for simplification
        self.model_type = "simplifier"
        
        # Simplification levels map to grade levels
        self.level_grade_map: Dict[int, int] = {
            1: 12,  # College level
            2: 10,  # High school 
            3: 8,   # Middle school
            4: 6,   # Elementary school
            5: 4    # Early elementary
        }
        
        # Load grade level vocabulary
        self.grade_level_vocabulary: Dict[int, Dict[str, Dict[str, str]]] = {}
        
        logger.info("Simplification pipeline created (not yet initialized)")
    
    async def initialize(self) -> None:
        """
        Initialize the simplification pipeline.
        
        This loads necessary models and prepares the pipeline.
        """
        if self.initialized:
            logger.warning("Simplification pipeline already initialized")
            return
        
        logger.info("Initializing simplification pipeline")
        
        # Load simplification model
        try:
            logger.info(f"Loading simplification model ({self.model_type})")
            await self.model_manager.load_model(self.model_type)
            logger.info("Simplification model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading simplification model: {str(e)}")
            logger.warning("Pipeline will function with reduced capabilities")
        
        # Initialize grade level vocabulary
        await self._load_grade_level_vocabulary()
        
        self.initialized = True
        logger.info("Simplification pipeline initialization complete")
    
    async def simplify(self, 
                      text: str, 
                      language: str,
                      level: int = 3,
                      grade_level: Optional[int] = None,
                      options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Simplify text to the specified level.
        
        Args:
            text: Text to simplify
            language: Language code
            level: Simplification level (1-5, where 5 is simplest)
            grade_level: Target grade level (1-12)
            options: Additional options
                - context: Additional context for simplification
                - domain: Specific domain (legal, medical, etc.)
                - preserve_formatting: Whether to preserve formatting
                - model_name: Specific model to use
                
        Returns:
            Dict containing:
            - simplified_text: Simplified text
            - model_used: Name of model used
            - level: Simplification level used
            - grade_level: Target grade level
            - metrics: Readability metrics
        """
        if not self.initialized:
            await self.initialize()
        
        if not text:
            return {"simplified_text": "", "model_used": "none"}
        
        options = options or {}
        
        # Determine target grade level
        if grade_level:
            # If grade level is specified directly, use it
            target_grade = grade_level
            # Find the closest simplification level
            level = min(self.level_grade_map.items(), key=lambda x: abs(x[1] - target_grade))[0]
        else:
            # Convert level to grade level
            level = max(1, min(5, level))  # Ensure level is between 1 and 5
            target_grade = self.level_grade_map.get(level, 8)
        
        logger.debug(f"Simplifying text to level {level} (grade {target_grade})")
        
        try:
            # Import prompt enhancer if available
            try:
                from app.services.models.simplifier_prompt_enhancer import SimplifierPromptEnhancer
                prompt_enhancer = SimplifierPromptEnhancer()
                enhanced_prompt = True
            except ImportError:
                enhanced_prompt = False
                logger.debug("Simplifier prompt enhancer not available")
            
            # Get model ID if specified, otherwise use default
            model_id = options.get("model_name", self.model_type)
            
            # Prepare simplification input
            input_data = {
                "text": text,
                "source_language": language,
                "parameters": {
                    "level": level,
                    "grade_level": target_grade,
                    "domain": options.get("domain"),
                    "preserve_formatting": options.get("preserve_formatting", True)
                }
            }
            
            # Add context if provided
            if "context" in options:
                input_data["context"] = options["context"]
            
            # Enhance prompt if available
            if enhanced_prompt:
                try:
                    # Enhance prompt with model-specific optimization
                    enhanced_data = prompt_enhancer.enhance_prompt(
                        text=text,
                        model_name=model_id,
                        level=level,
                        grade_level=target_grade,
                        language=language,
                        domain=options.get("domain"),
                        preserve_formatting=options.get("preserve_formatting", True)
                    )
                    
                    # Update input data with enhanced prompt
                    if "prompt" in enhanced_data:
                        input_data["prompt"] = enhanced_data["prompt"]
                    
                    # Update parameters with optimized values
                    if "parameters" in enhanced_data:
                        input_data["parameters"].update(enhanced_data["parameters"])
                        
                    logger.debug("Using enhanced prompt for simplification")
                except Exception as e:
                    logger.warning(f"Error enhancing simplification prompt: {str(e)}")
            
            # Run simplification model
            start_time = time.time()
            try:
                result = await self.model_manager.run_model(
                    model_id,
                    "process",
                    input_data
                )
                processing_time = time.time() - start_time
                
                # Extract simplification results
                if isinstance(result, dict) and "result" in result:
                    simplified_text = result["result"]
                    
                    # Handle different result formats
                    if isinstance(simplified_text, list) and simplified_text:
                        simplified_text = simplified_text[0]
                    elif not isinstance(simplified_text, str):
                        simplified_text = str(simplified_text)
                else:
                    # Handle unexpected result format
                    simplified_text = str(result) if result else None
                
                # If we still don't have a valid simplified text, use the rule-based fallback
                if not simplified_text or simplified_text == "None" or simplified_text.strip() == "":
                    logger.warning(f"Model-based simplification failed, using rule-based fallback for {language}")
                    simplified_text = self._rule_based_simplify(text, language, level)
            except Exception as e:
                logger.error(f"Model-based simplification error: {str(e)}", exc_info=True)
                processing_time = time.time() - start_time
                simplified_text = self._rule_based_simplify(text, language, level)
            
            # Apply grade level vocabulary if available
            if target_grade in self.grade_level_vocabulary:
                simplified_text = self._apply_grade_level_vocabulary(
                    simplified_text,
                    language,
                    target_grade
                )
            
            # Calculate readability metrics
            metrics = self._calculate_readability_metrics(simplified_text, language)
            
            # Track simplification metrics
            try:
                # Track metrics for this operation
                from app.audit.metrics import MetricsCollector
                metrics_collector = MetricsCollector.get_instance()
                metrics_collector.record_simplification_metrics(
                    language=language,
                    text_length=len(text),
                    simplified_length=len(simplified_text),
                    level=str(level),
                    processing_time=processing_time,
                    model_id=model_id
                )
                
                # Log to audit system
                from app.audit.logger import AuditLogger
                audit_logger = AuditLogger()
                await audit_logger.log_simplification(
                    text_length=len(text),
                    simplified_length=len(simplified_text),
                    language=language,
                    level=str(level),
                    model_id=model_id,
                    processing_time=processing_time,
                    metadata={
                        "grade_level": target_grade,
                        "domain": options.get("domain"),
                        "enhanced_prompt": enhanced_prompt,
                        "readability_metrics": metrics
                    }
                )
                
                # Verify quality if veracity auditor is available
                try:
                    from app.audit.veracity import VeracityAuditor
                    veracity = VeracityAuditor()
                    verification_result = await veracity._verify_simplification(
                        text,
                        simplified_text,
                        language,
                        {
                            "level": level,
                            "grade_level": target_grade,
                            "domain": options.get("domain")
                        }
                    )
                    
                    # Add verification metrics to the result
                    metrics["verification"] = {
                        "score": verification_result.get("score", 0.0),
                        "verified": verification_result.get("verified", False),
                        "issues": len(verification_result.get("issues", []))
                    }
                except (ImportError, Exception) as e:
                    logger.debug(f"Could not verify simplification quality: {str(e)}")
            except (ImportError, Exception) as e:
                logger.debug(f"Could not log simplification metrics: {str(e)}")
            
            logger.debug(f"Simplification completed in {processing_time:.3f}s")
            
            return {
                "simplified_text": simplified_text,
                "model_used": model_id,
                "level": level,
                "grade_level": target_grade,
                "metrics": metrics,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Simplification error: {str(e)}", exc_info=True)
            raise
    
    async def _load_grade_level_vocabulary(self) -> None:
        """
        Load grade-level vocabulary for different languages.
        
        This loads vocabulary appropriate for different grade levels
        to ensure simplified text uses appropriate vocabulary.
        """
        try:
            # In a real implementation, this would load from files
            # For now, we'll use a simple dictionary
            
            self.grade_level_vocabulary = {
                # Grade 4 vocabulary (elementary)
                4: {
                    "en": {
                        "utilize": "use",
                        "purchase": "buy",
                        "indicate": "show",
                        "sufficient": "enough",
                        "assist": "help",
                        "obtain": "get",
                        "require": "need",
                        "additional": "more",
                        "prior to": "before",
                        "subsequently": "later"
                    }
                },
                # Grade 8 vocabulary (middle school)
                8: {
                    "en": {
                        "utilize": "use",
                        "purchase": "buy",
                        "indicate": "show",
                        "sufficient": "enough",
                        "subsequently": "later"
                    }
                }
            }
            
            logger.info(f"Loaded grade level vocabulary for {len(self.grade_level_vocabulary)} grade levels")
            
        except Exception as e:
            logger.error(f"Error loading grade level vocabulary: {str(e)}", exc_info=True)
            self.grade_level_vocabulary = {}
    
    def _apply_grade_level_vocabulary(self, 
                                    text: str,
                                    language: str,
                                    grade_level: int) -> str:
        """
        Apply grade-level appropriate vocabulary.
        
        Args:
            text: Text to process
            language: Language code
            grade_level: Target grade level
            
        Returns:
            Text with grade-appropriate vocabulary
        """
        # Find closest available grade level
        available_grades = list(self.grade_level_vocabulary.keys())
        if not available_grades:
            return text
        
        closest_grade = min(available_grades, key=lambda g: abs(g - grade_level))
        grade_vocab = self.grade_level_vocabulary.get(closest_grade, {}).get(language, {})
        
        if not grade_vocab:
            return text
        
        # Apply vocabulary replacements
        processed_text = text
        for complex_word, simple_word in grade_vocab.items():
            pattern = r'\b' + re.escape(complex_word) + r'\b'
            processed_text = re.sub(pattern, simple_word, processed_text, flags=re.IGNORECASE)
        
        return processed_text
    
    def _calculate_readability_metrics(self, text: str, language: str) -> Dict[str, float]:
        """
        Calculate readability metrics for the simplified text.
        
        Args:
            text: Text to analyze
            language: Language code
            
        Returns:
            Dict with readability metrics
        """
        # Only English is fully supported for now
        if language != "en":
            return {"estimated_grade_level": 0}
        
        try:
            # Count words and sentences
            words = len(re.findall(r'\b\w+\b', text))
            sentences = len(re.findall(r'[.!?]+', text)) or 1
            
            # Count syllables (rough approximation)
            syllables = 0
            for word in re.findall(r'\b\w+\b', text):
                word = word.lower()
                if len(word) <= 3:
                    syllables += 1
                else:
                    # Count vowel groups as syllables
                    vowels = "aeiouy"
                    count = 0
                    prev_is_vowel = False
                    for char in word:
                        is_vowel = char in vowels
                        if is_vowel and not prev_is_vowel:
                            count += 1
                        prev_is_vowel = is_vowel
                    
                    # Adjust for common patterns
                    if word.endswith('e'):
                        count -= 1
                    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                        count += 1
                    if count == 0:
                        count = 1
                    
                    syllables += count
            
            # Calculate Flesch-Kincaid Grade Level
            words_per_sentence = words / sentences
            syllables_per_word = syllables / words
            fk_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
            
            # Ensure grade level is reasonable
            fk_grade = max(1, min(12, fk_grade))
            
            return {
                "estimated_grade_level": round(fk_grade, 1),
                "words_per_sentence": round(words_per_sentence, 1),
                "syllables_per_word": round(syllables_per_word, 1),
                "words": words,
                "sentences": sentences
            }
            
        except Exception as e:
            logger.error(f"Error calculating readability metrics: {str(e)}", exc_info=True)
            return {"estimated_grade_level": 0}
            
    def _rule_based_simplify(self, text: str, level: int, language: str = "en", domain: str = None) -> str:
        """
        Apply rule-based simplification with a specific level.
        
        Args:
            text: Text to simplify
            level: Simplification level (1-5, where 5 is simplest)
            language: Language code
            domain: Optional domain for domain-specific simplification
            
        Returns:
            Simplified text
        """
        
        # If no text, return empty string
        if not text:
            return ""
        
        # Check if legal domain
        is_legal_domain = domain and "legal" in domain.lower()
        
        # Define vocabulary replacements for different levels
        replacements = {}
        
        # Level 1 (minimal simplification)
        level1_replacements = {
            r'\butilize\b': 'use',
            r'\bpurchase\b': 'buy',
            r'\bsubsequently\b': 'later',
            r'\bfurnish\b': 'provide',
            r'\baforementioned\b': 'previously mentioned',
            r'\bdelineated\b': 'outlined',
            r'\bin accordance with\b': 'according to'
        }
        
        # Level 2
        level2_replacements = {
            r'\bindicate\b': 'show',
            r'\bsufficient\b': 'enough',
            r'\badditional\b': 'more',
            r'\bprior to\b': 'before',
            r'\bverifying\b': 'proving',
            r'\brequirements\b': 'rules'
        }
        
        # Level 3
        level3_replacements = {
            r'\bassist\b': 'help',
            r'\bobtain\b': 'get',
            r'\brequire\b': 'need',
            r'\bcommence\b': 'start',
            r'\bterminate\b': 'end',
            r'\bdemonstrate\b': 'show',
            r'\bdelineated\b': 'described',
            r'\bin accordance with\b': 'following',
            r'\bemployment status\b': 'job status',
            r'\bapplication procedure\b': 'application process'
        }
        
        # Level 4
        level4_replacements = {
            r'\bregarding\b': 'about',
            r'\bimplement\b': 'use',
            r'\bnumerous\b': 'many',
            r'\bfacilitate\b': 'help',
            r'\binitial\b': 'first',
            r'\battempt\b': 'try',
            r'\bapplicant\b': 'you',
            r'\bfurnish\b': 'give',
            r'\baforementioned\b': 'this',
            r'\bdelineated\b': 'listed',
            r'\bverifying\b': 'that proves',
            r'\bemployment status\b': 'job information',
            r'\bapplication procedure\b': 'steps',
            r'\bdocumentation\b': 'papers',
            r'\bsection\b': 'part'
        }
        
        # Level 5
        level5_replacements = {
            r'\binquire\b': 'ask',
            r'\bascertain\b': 'find out',
            r'\bcomprehend\b': 'understand',
            r'\bnevertheless\b': 'but',
            r'\btherefore\b': 'so',
            r'\bfurthermore\b': 'also',
            r'\bconsequently\b': 'so',
            r'\bapproximately\b': 'about',
            r'\bmodification\b': 'change',
            r'\bendeavor\b': 'try',
            r'\bproficiency\b': 'skill',
            r'\bnecessitate\b': 'need',
            r'\bacquisition\b': 'getting',
            r'\bemployment status\b': 'job info',
            r'\bapplication procedure\b': 'form',
            r'\bmust\b': 'need to'
        }
        
        # Add replacements based on level
        replacements.update(level1_replacements)
        if level >= 2:
            replacements.update(level2_replacements)
        if level >= 3:
            replacements.update(level3_replacements)
        if level >= 4:
            replacements.update(level4_replacements)
        if level >= 5:
            replacements.update(level5_replacements)
        
        # Handle sentence splitting for higher levels
        if level >= 3:
            # Split text into sentences
            sentences = re.split(r'([.!?])', text)
            processed_sentences = []
            
            # Process each sentence
            i = 0
            while i < len(sentences):
                if i + 1 < len(sentences):
                    # Combine sentence with its punctuation
                    sentence = sentences[i] + sentences[i+1]
                    i += 2
                else:
                    sentence = sentences[i]
                    i += 1
                    
                # Skip empty sentences
                if not sentence.strip():
                    continue
                    
                # For higher simplification levels, break long sentences
                if len(sentence.split()) > 15:
                    # More aggressive splitting for highest levels
                    if level >= 4:
                        clauses = re.split(r'([,;:])', sentence)
                        for j in range(0, len(clauses), 2):
                            if j + 1 < len(clauses):
                                processed_sentences.append(clauses[j] + clauses[j+1])
                            else:
                                processed_sentences.append(clauses[j])
                    else:
                        # Less aggressive for level 3
                        clauses = re.split(r'([;:])', sentence) 
                        for j in range(0, len(clauses), 2):
                            if j + 1 < len(clauses):
                                processed_sentences.append(clauses[j] + clauses[j+1])
                            else:
                                processed_sentences.append(clauses[j])
                else:
                    processed_sentences.append(sentence)
            
            # Join sentences
            simplified_text = " ".join(processed_sentences)
        else:
            # For lower levels, don't split sentences
            simplified_text = text
        
        # Apply word replacements
        for pattern, replacement in replacements.items():
            try:
                simplified_text = re.sub(pattern, replacement, simplified_text, flags=re.IGNORECASE)
            except:
                # Skip problematic patterns
                pass
        
        # Clean up spaces
        simplified_text = re.sub(r'\s+', ' ', simplified_text).strip()
        
        # For highest level, add explaining phrases
        if level == 5:
            if is_legal_domain:
                simplified_text += " This means you need to follow what the law says."
            else:
                simplified_text += " This means you need to show the required information."
        
        return simplified_text
    
    def _get_language_replacements(self, language: str, level: int) -> Dict[str, str]:
        """
        Get language-specific word replacements based on simplification level.
        
        Args:
            language: Language code
            level: Simplification level (1-5)
            
        Returns:
            Dictionary of complex words to simple words
        """
        # Default English replacements
        en_replacements = {
            # Level 1 (minimal simplification)
            1: {
                "utilize": "use",
                "purchase": "buy",
                "subsequently": "later"
            },
            # Level 2
            2: {
                "utilize": "use",
                "purchase": "buy",
                "indicate": "show",
                "sufficient": "enough",
                "subsequently": "later",
                "additional": "more",
                "prior to": "before"
            },
            # Level 3
            3: {
                "utilize": "use",
                "purchase": "buy",
                "indicate": "show",
                "sufficient": "enough",
                "assist": "help",
                "obtain": "get",
                "require": "need",
                "additional": "more",
                "prior to": "before",
                "subsequently": "later",
                "commence": "start",
                "terminate": "end",
                "demonstrate": "show"
            },
            # Level 4
            4: {
                "utilize": "use",
                "purchase": "buy",
                "indicate": "show",
                "sufficient": "enough",
                "assist": "help",
                "obtain": "get",
                "require": "need",
                "additional": "more",
                "prior to": "before",
                "subsequently": "later",
                "commence": "start",
                "terminate": "end",
                "demonstrate": "show",
                "regarding": "about",
                "implement": "use",
                "numerous": "many",
                "facilitate": "help",
                "initial": "first",
                "attempt": "try"
            },
            # Level 5 (maximum simplification)
            5: {
                "utilize": "use",
                "purchase": "buy",
                "indicate": "show",
                "sufficient": "enough",
                "assist": "help",
                "obtain": "get",
                "require": "need",
                "additional": "more",
                "prior to": "before",
                "subsequently": "later",
                "commence": "start",
                "terminate": "end",
                "demonstrate": "show",
                "regarding": "about",
                "implement": "use",
                "numerous": "many",
                "facilitate": "help",
                "initial": "first",
                "attempt": "try",
                "inquire": "ask",
                "ascertain": "find out",
                "comprehend": "understand",
                "nevertheless": "however",
                "therefore": "so",
                "furthermore": "also",
                "consequently": "so",
                "approximately": "about",
                "modification": "change",
                "endeavor": "try",
                "proficiency": "skill",
                "necessitate": "need",
                "acquisition": "getting",
                "immersion": "practice",
                "assimilation": "learning"
            }
        }
        
        # Spanish replacements
        es_replacements = {
            # Level 3 (medium simplification)
            3: {
                "utilizar": "usar",
                "adquirir": "comprar",
                "indicar": "mostrar",
                "suficiente": "bastante",
                "ayudar": "asistir",
                "obtener": "conseguir",
                "requerir": "necesitar",
                "adicional": "más",
                "antes de": "previamente a",
                "subsecuentemente": "después"
            },
            # Add more levels as needed
        }
        
        # Select the appropriate replacements based on language and level
        if language == "es":
            # Get the closest available level for Spanish
            available_levels = sorted(es_replacements.keys())
            if not available_levels:
                return {}
                
            closest_level = min(available_levels, key=lambda l: abs(l - level))
            return es_replacements[closest_level]
        else:
            # Default to English
            # Get the closest available level
            available_levels = sorted(en_replacements.keys())
            if not available_levels:
                return {}
                
            closest_level = min(available_levels, key=lambda l: abs(l - level))
            return en_replacements[closest_level]