"""
Simplifier Prompt Enhancement Module for CasaLingua

This module provides sophisticated model-aware prompting to enhance text simplification quality
through advanced prompt engineering, model specialization, and domain-specific strategies.

It provides:
1. Model-specific prompt templates and strategies
2. Five distinct simplification levels
3. Domain-specific simplification strategies
4. Grade-level targeting for educational content
5. Language-specific simplification approaches
6. Parameter optimization based on model characteristics
7. Special handling for complex text structures
"""

import logging
import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Set

# Configure logging
logger = logging.getLogger(__name__)

class SimplifierPromptEnhancer:
    """
    Enhances prompts for text simplification models to improve quality
    through model-aware prompt engineering, level-specific instructions,
    and domain-specific simplification strategies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the simplifier prompt enhancer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Define the 5 simplification levels
        # Level 1: Academic (minimal simplification, advanced vocabulary)
        # Level 2: Standard (moderate simplification, standard vocabulary)
        # Level 3: Simplified (significant simplification, accessible vocabulary)
        # Level 4: Basic (highly simplified, basic vocabulary)
        # Level 5: Elementary (maximum simplification, elementary vocabulary)
        self.simplification_levels = {
            1: {
                "name": "Academic",
                "description": "Maintain advanced vocabulary but improve clarity and structure",
                "grade_level": 12,  # College level
                "target_audience": "Academic, professional",
                "sentence_length": "Long, complex sentences allowed",
                "vocabulary_level": "Advanced, technical vocabulary preserved"
            },
            2: {
                "name": "Standard",
                "description": "Use standard vocabulary with clear sentence structure",
                "grade_level": 10,  # High school
                "target_audience": "General adult audience",
                "sentence_length": "Medium to long sentences",
                "vocabulary_level": "Standard vocabulary, some specialized terms"
            },
            3: {
                "name": "Simplified",
                "description": "Simplify vocabulary and sentence structure for better comprehension",
                "grade_level": 8,   # Middle school
                "target_audience": "General population, ESL readers",
                "sentence_length": "Medium sentences, clear structure",
                "vocabulary_level": "Common vocabulary, few specialized terms"
            },
            4: {
                "name": "Basic",
                "description": "Use basic vocabulary and simple sentences for easy comprehension",
                "grade_level": 6,   # Elementary school
                "target_audience": "Early ESL, low literacy readers",
                "sentence_length": "Short, direct sentences",
                "vocabulary_level": "Basic vocabulary, everyday terms"
            },
            5: {
                "name": "Elementary",
                "description": "Maximum simplification with elementary vocabulary and very simple structure",
                "grade_level": 4,   # Early elementary
                "target_audience": "Young readers, very low literacy",
                "sentence_length": "Very short, simple sentences",
                "vocabulary_level": "Elementary vocabulary only"
            }
        }
        
        # Track known models and their capabilities
        self.model_capabilities = {
            "bart-large": {
                "strengths": ["english_simplification", "paraphrasing", "structure_preservation"],
                "weaknesses": ["non_english_languages", "very_complex_texts"],
                "preferred_domains": ["general", "educational", "news"],
                "instruction_style": "detailed",  # BART benefits from detailed instructions
                "max_prompt_tokens": 200
            },
            "bart-base": {
                "strengths": ["english_simplification", "paraphrasing"],
                "weaknesses": ["technical_terminology", "non_english_languages"],
                "preferred_domains": ["general", "casual"],
                "instruction_style": "detailed",
                "max_prompt_tokens": 150
            },
            "t5-base": {
                "strengths": ["instruction_following", "flexible_simplification"],
                "weaknesses": ["very_technical_content", "specialized_domains"],
                "preferred_domains": ["general", "educational"],
                "instruction_style": "explicit",  # T5 benefits from explicit instructions
                "max_prompt_tokens": 200
            },
            "t5-small": {
                "strengths": ["instruction_following", "basic_simplification"],
                "weaknesses": ["complex_structures", "technical_content", "domain_specific"],
                "preferred_domains": ["general", "educational"],
                "instruction_style": "explicit",
                "max_prompt_tokens": 150
            },
            "default": {
                "strengths": ["general_simplification"],
                "weaknesses": [],
                "preferred_domains": ["general"],
                "instruction_style": "balanced",
                "max_prompt_tokens": 100
            }
        }
        
        # Domain-specific simplification strategies
        self.domain_strategies = {
            "legal": {
                "description": "Simplification of legal texts with preservation of key legal concepts",
                "key_terms_preservation": ["contract", "agreement", "party", "clause", "provision", 
                                           "jurisdiction", "plaintiff", "defendant", "liability", "damages"],
                "structural_elements": ["sections", "paragraphs", "numbered lists", "definitions"],
                "explanatory_additions": True,  # Add explanatory notes for complex terms
                "key_term_substitutions": {
                    1: {},  # Level 1: No substitutions
                    2: {
                        "pursuant to": "according to",
                        "herein": "in this document",
                        "therein": "in that document"
                    },
                    3: {
                        "pursuant to": "according to",
                        "herein": "in this document",
                        "therein": "in that document",
                        "prior to": "before",
                        "subsequent to": "after",
                        "commence": "start",
                        "terminate": "end"
                    },
                    4: {
                        "pursuant to": "following",
                        "herein": "here",
                        "therein": "there",
                        "prior to": "before",
                        "subsequent to": "after",
                        "commence": "start",
                        "terminate": "end",
                        "plaintiff": "person suing",
                        "defendant": "person being sued",
                        "provision": "rule",
                        "jurisdiction": "authority"
                    },
                    5: {
                        "pursuant to": "by",
                        "herein": "here",
                        "therein": "there",
                        "prior to": "before",
                        "subsequent to": "after",
                        "commence": "start",
                        "terminate": "end",
                        "plaintiff": "person suing",
                        "defendant": "person being sued",
                        "provision": "rule",
                        "jurisdiction": "power",
                        "shall": "must",
                        "liable": "responsible",
                        "covenant": "promise",
                        "damages": "money"
                    }
                }
            },
            "medical": {
                "description": "Simplification of medical texts with preservation of medical accuracy",
                "key_terms_preservation": ["diagnosis", "treatment", "medication", "symptom", "prognosis", 
                                          "chronic", "acute", "condition"],
                "structural_elements": ["sections", "bulleted lists", "instructions"],
                "explanatory_additions": True,  # Add explanations for medical terms
                "key_term_substitutions": {
                    1: {},  # Level 1: No substitutions
                    2: {
                        "utilize": "use",
                        "administer": "give",
                        "monitor": "watch"
                    },
                    3: {
                        "utilize": "use",
                        "administer": "give",
                        "monitor": "watch",
                        "initiate": "start",
                        "terminate": "stop",
                        "adverse effects": "side effects",
                        "physician": "doctor"
                    },
                    4: {
                        "utilize": "use",
                        "administer": "give",
                        "monitor": "watch",
                        "initiate": "start",
                        "terminate": "stop",
                        "adverse effects": "side effects",
                        "physician": "doctor",
                        "hypertension": "high blood pressure",
                        "cardiac": "heart",
                        "pulmonary": "lung",
                        "diabetes mellitus": "diabetes",
                        "analgesic": "pain reliever"
                    },
                    5: {
                        "utilize": "use",
                        "administer": "give",
                        "monitor": "watch",
                        "initiate": "start",
                        "terminate": "stop",
                        "adverse effects": "side effects",
                        "physician": "doctor",
                        "hypertension": "high blood pressure",
                        "cardiac": "heart",
                        "pulmonary": "lung",
                        "diabetes mellitus": "diabetes",
                        "analgesic": "pain reliever",
                        "oral": "by mouth",
                        "topical": "on skin",
                        "condition": "problem",
                        "medication": "medicine",
                        "symptom": "sign",
                        "prescription": "doctor's order"
                    }
                }
            },
            "technical": {
                "description": "Simplification of technical documentation with preservation of technical accuracy",
                "key_terms_preservation": ["database", "server", "protocol", "algorithm", "interface", 
                                           "function", "class", "method", "parameter", "variable"],
                "structural_elements": ["code blocks", "step-by-step instructions", "diagrams"],
                "explanatory_additions": True,  # Add explanations for technical concepts
                "key_term_substitutions": {
                    1: {},  # Level 1: No substitutions
                    2: {
                        "utilize": "use",
                        "implementation": "version",
                        "functionality": "features"
                    },
                    3: {
                        "utilize": "use",
                        "implementation": "version",
                        "functionality": "features",
                        "initialize": "start",
                        "terminate": "end",
                        "prerequisite": "requirement"
                    },
                    4: {
                        "utilize": "use",
                        "implementation": "version",
                        "functionality": "features",
                        "initialize": "start",
                        "terminate": "end",
                        "prerequisite": "requirement",
                        "configuration": "setup",
                        "authenticate": "sign in",
                        "parameter": "setting",
                        "interface": "connection"
                    },
                    5: {
                        "utilize": "use",
                        "implementation": "version",
                        "functionality": "features",
                        "initialize": "start",
                        "terminate": "end",
                        "prerequisite": "need",
                        "configuration": "setup",
                        "authenticate": "sign in",
                        "parameter": "setting",
                        "interface": "screen",
                        "database": "stored information",
                        "execute": "run",
                        "compile": "build",
                        "navigate": "go to",
                        "directive": "instruction"
                    }
                }
            },
            "financial": {
                "description": "Simplification of financial texts with preservation of financial accuracy",
                "key_terms_preservation": ["asset", "liability", "equity", "dividend", "interest", 
                                           "principal", "maturity", "portfolio", "diversification"],
                "structural_elements": ["tables", "charts", "summaries"],
                "explanatory_additions": True,  # Add explanations for financial concepts
                "key_term_substitutions": {
                    1: {},  # Level 1: No substitutions
                    2: {
                        "utilize": "use",
                        "initiate": "start",
                        "terminate": "end",
                        "additional": "more"
                    },
                    3: {
                        "utilize": "use",
                        "initiate": "start",
                        "terminate": "end",
                        "additional": "more",
                        "equity": "ownership",
                        "appreciation": "increase in value",
                        "depreciation": "decrease in value",
                        "portfolio": "collection of investments"
                    },
                    4: {
                        "utilize": "use",
                        "initiate": "start",
                        "terminate": "end",
                        "additional": "more",
                        "equity": "ownership",
                        "appreciation": "increase in value",
                        "depreciation": "decrease in value",
                        "portfolio": "collection of investments",
                        "assets": "things you own",
                        "liabilities": "things you owe",
                        "dividend": "payment to shareholders",
                        "interest": "extra money paid",
                        "compound interest": "interest on interest"
                    },
                    5: {
                        "utilize": "use",
                        "initiate": "start",
                        "terminate": "end",
                        "additional": "more",
                        "equity": "ownership",
                        "appreciation": "increase in value",
                        "depreciation": "decrease in value",
                        "portfolio": "investments",
                        "assets": "things you own",
                        "liabilities": "things you owe",
                        "dividend": "company payment",
                        "interest": "extra money",
                        "compound interest": "growing money",
                        "maturity": "end date",
                        "principal": "main amount",
                        "diversification": "spreading out risk",
                        "investment": "money you put in",
                        "stock": "company share",
                        "bond": "company loan"
                    }
                }
            },
            "educational": {
                "description": "Simplification of educational content targeting specific grade levels",
                "key_terms_preservation": ["learning", "concept", "theory", "example", "definition",
                                          "problem", "solution", "evidence", "conclusion"],
                "structural_elements": ["headings", "bullet points", "examples", "summaries"],
                "explanatory_additions": True,  # Add explanations for educational concepts
                "grade_level_adaptations": True,  # Specifically adapt to grade levels
                "key_term_substitutions": {
                    1: {},  # Level 1: No substitutions
                    2: {
                        "utilize": "use",
                        "therefore": "so",
                        "furthermore": "also",
                        "demonstrate": "show"
                    },
                    3: {
                        "utilize": "use",
                        "therefore": "so",
                        "furthermore": "also",
                        "demonstrate": "show",
                        "comprehend": "understand",
                        "sufficient": "enough",
                        "assistance": "help",
                        "approximately": "about"
                    },
                    4: {
                        "utilize": "use",
                        "therefore": "so",
                        "furthermore": "also",
                        "demonstrate": "show",
                        "comprehend": "understand",
                        "sufficient": "enough",
                        "assistance": "help",
                        "approximately": "about",
                        "concept": "idea",
                        "analyze": "study",
                        "identify": "find",
                        "summarize": "sum up"
                    },
                    5: {
                        "utilize": "use",
                        "therefore": "so",
                        "furthermore": "also",
                        "demonstrate": "show",
                        "comprehend": "understand",
                        "sufficient": "enough",
                        "assistance": "help",
                        "approximately": "about",
                        "concept": "idea",
                        "analyze": "look at",
                        "identify": "find",
                        "summarize": "tell the main points",
                        "observe": "see",
                        "explain": "tell why",
                        "define": "tell what it means",
                        "compare": "tell how they are alike",
                        "contrast": "tell how they are different",
                        "evidence": "proof",
                        "conclusion": "end result"
                    }
                }
            }
        }
        
        # Prompt templates for different instruction styles and models
        self.prompt_templates = {
            "minimal": "simplify text level {level}: {text}",
            "balanced": "simplify the following text to level {level} (1-5, where 5 is simplest): {text}",
            "detailed": "simplify this text to {level_name} level (level {level}/5). Use {vocabulary_level} and {sentence_length}. Target audience: {target_audience}. {text}",
            "explicit": "simplify this text to grade {grade_level} reading level. Use {vocabulary_level} and create {sentence_length}. Maintain the original meaning while making the text more accessible to {target_audience}. Text to simplify: {text}",
            "grade_level": "simplify this text to a {grade_level}th grade reading level: {text}"
        }
        
        # Domain-specific prompt templates
        self.domain_prompt_templates = {
            "legal": {
                "detailed": "simplify this legal text to {level_name} level (level {level}/5). Preserve key legal terms {preserve_terms} while simplifying language. Use {vocabulary_level} and {sentence_length}. Target audience: {target_audience}. Text to simplify: {text}",
                "explicit": "simplify this legal text to grade {grade_level} reading level while preserving legal meaning. Maintain key legal terms but explain them in simpler language. Text to simplify: {text}"
            },
            "medical": {
                "detailed": "simplify this medical text to {level_name} level (level {level}/5). Preserve medical accuracy and key terms {preserve_terms} while simplifying language. Use {vocabulary_level} and {sentence_length}. Target audience: {target_audience}. Text to simplify: {text}",
                "explicit": "simplify this medical text to grade {grade_level} reading level while maintaining medical accuracy. Use plain language explanations for medical terms where appropriate. Text to simplify: {text}"
            },
            "technical": {
                "detailed": "simplify this technical document to {level_name} level (level {level}/5). Preserve key technical concepts {preserve_terms} while simplifying language. Use {vocabulary_level} and {sentence_length}. Target audience: {target_audience}. Text to simplify: {text}",
                "explicit": "simplify this technical document to grade {grade_level} reading level while maintaining technical accuracy. Explain technical concepts in simpler terms. Text to simplify: {text}"
            }
        }
        
        # Grade level vocabulary and sentence characteristics
        self.grade_level_guides = {
            4: {  # 4th grade
                "sentence_length": 8,  # Average words per sentence
                "words_per_paragraph": 45,
                "syllables_per_word": 1.3,
                "common_connectors": ["and", "but", "or", "so", "because"],
                "vocab_level": "elementary",
                "sentence_structure": "simple"
            },
            6: {  # 6th grade
                "sentence_length": 12,
                "words_per_paragraph": 75,
                "syllables_per_word": 1.5,
                "common_connectors": ["and", "but", "or", "so", "because", "however", "therefore"],
                "vocab_level": "basic",
                "sentence_structure": "simple to compound"
            },
            8: {  # 8th grade
                "sentence_length": 15,
                "words_per_paragraph": 100,
                "syllables_per_word": 1.6,
                "common_connectors": ["and", "but", "or", "so", "because", "however", "therefore", "although", "despite"],
                "vocab_level": "intermediate",
                "sentence_structure": "compound and some complex"
            },
            10: {  # 10th grade
                "sentence_length": 18,
                "words_per_paragraph": 125,
                "syllables_per_word": 1.7,
                "common_connectors": ["and", "but", "or", "so", "because", "however", "therefore", "although", "despite", "nevertheless", "furthermore"],
                "vocab_level": "advanced",
                "sentence_structure": "compound and complex"
            },
            12: {  # 12th grade / college
                "sentence_length": 22,
                "words_per_paragraph": 150,
                "syllables_per_word": 1.8,
                "common_connectors": ["and", "but", "or", "so", "because", "however", "therefore", "although", "despite", "nevertheless", "furthermore", "consequently", "conversely"],
                "vocab_level": "advanced to specialized",
                "sentence_structure": "varied complex"
            }
        }
        
        # Quality hints for different models and scenarios
        self.quality_hints = {
            "bart": [
                "maintain original meaning while simplifying",
                "use simpler vocabulary and shorter sentences",
                "preserve key concepts while simplifying language"
            ],
            "t5": [
                "simplify to target reading level",
                "rephrase complex sentences as shorter ones",
                "use clearer language without changing meaning"
            ],
            "domain_specific": [
                "preserve key domain terms while simplifying surrounding language",
                "explain specialized concepts in simpler terms",
                "maintain the original meaning while simplifying structure"
            ],
            "default": [
                "simplify text to appropriate level",
                "preserve original meaning"
            ]
        }
        
        logger.info("Simplifier prompt enhancer initialized")
    
    def get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """
        Get capability information for a specific model.
        
        Args:
            model_name: The model name to look up
            
        Returns:
            Dictionary of model capabilities
        """
        # Look for exact match
        if model_name in self.model_capabilities:
            return self.model_capabilities[model_name]
        
        # Look for partial match
        for model_key in self.model_capabilities:
            if model_key in model_name:
                return self.model_capabilities[model_key]
        
        # Fall back to default
        return self.model_capabilities["default"]
    
    def get_model_instruction_style(self, model_name: str) -> str:
        """
        Get the optimal instruction style for a model.
        
        Args:
            model_name: The model name to look up
            
        Returns:
            Instruction style (minimal, balanced, detailed, explicit)
        """
        capabilities = self.get_model_capabilities(model_name)
        return capabilities.get("instruction_style", "balanced")
    
    def get_simplification_level_info(self, level: int) -> Dict[str, Any]:
        """
        Get information about a specific simplification level.
        
        Args:
            level: Simplification level (1-5)
            
        Returns:
            Dictionary with level information
        """
        level = max(1, min(5, level))  # Ensure level is between 1 and 5
        return self.simplification_levels[level]
    
    def get_domain_strategy(self, domain: str) -> Dict[str, Any]:
        """
        Get domain-specific simplification strategy.
        
        Args:
            domain: Domain name
            
        Returns:
            Dictionary with domain-specific strategy
        """
        # Check for exact match
        if domain in self.domain_strategies:
            return self.domain_strategies[domain]
        
        # Check for partial match
        for domain_key in self.domain_strategies:
            if domain_key in domain:
                return self.domain_strategies[domain_key]
        
        # Fall back to empty strategy
        return {}
    
    def enhance_prompt(
        self, 
        text: str, 
        model_name: str, 
        level: int = 3, 
        grade_level: Optional[int] = None,
        language: str = "en",
        domain: Optional[str] = None,
        preserve_formatting: bool = True,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Enhance the text simplification prompt for a specific model.
        
        Args:
            text: Text to simplify
            model_name: Name/identifier of the model
            level: Simplification level (1-5, where 5 is simplest)
            grade_level: Target grade level (1-12)
            language: Language code
            domain: Specific domain for targeted simplification
            preserve_formatting: Whether to preserve original formatting
            parameters: Additional parameters for customization
            
        Returns:
            Dictionary with enhanced prompt and parameters
        """
        parameters = parameters or {}
        
        # Ensure level is between 1 and 5
        level = max(1, min(5, level))
        
        # Get level information
        level_info = self.get_simplification_level_info(level)
        
        # Determine grade level if not provided
        if grade_level is None:
            grade_level = level_info["grade_level"]
        else:
            # Ensure grade level is between 1 and 12
            grade_level = max(1, min(12, grade_level))
        
        # Get model capabilities and instruction style
        capabilities = self.get_model_capabilities(model_name)
        instruction_style = self.get_model_instruction_style(model_name)
        
        # Create appropriate prompt based on model capabilities and domain
        prompt = self._create_simplification_prompt(
            text, level, grade_level, language, domain, 
            instruction_style, capabilities, preserve_formatting
        )
        
        # Optimize parameters for this model and simplification level
        optimized_params = self._optimize_parameters(
            level, grade_level, language, domain, capabilities, parameters
        )
        
        return {
            "prompt": prompt,
            "parameters": optimized_params
        }
    
    def _create_simplification_prompt(
        self, 
        text: str, 
        level: int,
        grade_level: int,
        language: str,
        domain: Optional[str],
        instruction_style: str,
        capabilities: Dict[str, Any],
        preserve_formatting: bool
    ) -> str:
        """
        Create a model-appropriate simplification prompt.
        
        Args:
            text: Text to simplify
            level: Simplification level (1-5)
            grade_level: Target grade level
            language: Language code
            domain: Specific domain
            instruction_style: Model's preferred instruction style
            capabilities: Model capabilities
            preserve_formatting: Whether to preserve original formatting
            
        Returns:
            Optimized prompt string
        """
        # Get level information
        level_info = self.get_simplification_level_info(level)
        
        # Choose appropriate base template based on instruction style
        if instruction_style == "minimal":
            template = self.prompt_templates["minimal"]
        elif instruction_style == "balanced":
            template = self.prompt_templates["balanced"]
        elif instruction_style == "explicit":
            template = self.prompt_templates["explicit"]
        else:  # detailed
            template = self.prompt_templates["detailed"]
        
        # Get domain-specific strategy if applicable
        domain_strategy = self.get_domain_strategy(domain) if domain else {}
        
        # Use domain-specific template if available
        if domain and domain_strategy:
            if domain in self.domain_prompt_templates:
                domain_templates = self.domain_prompt_templates[domain]
                if instruction_style in domain_templates:
                    template = domain_templates[instruction_style]
        
        # Get key terms to preserve for domain if applicable
        preserve_terms = ""
        if domain and domain_strategy and "key_terms_preservation" in domain_strategy:
            # Select a subset of important terms based on level
            terms_to_include = min(6 - level, 5)  # Level 1: 5 terms, Level 5: 1 term
            if terms_to_include > 0:
                key_terms = domain_strategy["key_terms_preservation"][:terms_to_include]
                preserve_terms = ", ".join(key_terms)
        
        # Add quality hints based on model and domain
        hints = []
        
        # Add model-specific hints
        for model_key in self.quality_hints:
            if model_key in model_name:
                hints.extend(self.quality_hints[model_key][:2])  # Add up to 2 hints
                break
        
        # Add domain-specific hints
        if domain and domain_strategy:
            hints.extend(self.quality_hints["domain_specific"][:1])
        
        # If no specific hints were added, use default
        if not hints:
            hints.extend(self.quality_hints["default"])
        
        # Format template with level information
        prompt_params = {
            "level": level,
            "level_name": level_info["name"],
            "grade_level": grade_level,
            "target_audience": level_info["target_audience"],
            "vocabulary_level": level_info["vocabulary_level"],
            "sentence_length": level_info["sentence_length"],
            "preserve_terms": preserve_terms,
            "text": text
        }
        
        # Add formatting preservation instruction if needed
        if preserve_formatting and "detailed" in instruction_style:
            format_instruction = "Preserve paragraph breaks, bullet points, and other formatting elements. "
            
            # Add formatting preservation based on template type
            if "{text}" in template:
                template = template.replace("{text}", format_instruction + "{text}")
        
        # Format the template
        try:
            prompt = template.format(**prompt_params)
        except KeyError as e:
            # Fallback to simpler template if formatting fails
            logger.warning(f"Template formatting error: {e}, using fallback template")
            fallback_template = self.prompt_templates["balanced"]
            prompt = fallback_template.format(level=level, text=text)
        
        # Add hints for detailed instruction styles
        if instruction_style in ["detailed", "explicit"] and hints:
            hint_text = " ".join(hints)
            prompt = f"{hint_text}. {prompt}"
        
        # Limit prompt length based on model capabilities
        max_tokens = capabilities.get("max_prompt_tokens", 100)
        
        # Simple tokenization for length limiting
        words = prompt.split()
        if len(words) > max_tokens:
            # Keep the instruction part and truncate the text if needed
            instruction_part = " ".join(words[:min(50, max_tokens // 2)])
            text_limit = max_tokens - len(instruction_part.split())
            text_part = " ".join(words[-text_limit:])
            prompt = f"{instruction_part} {text_part}"
        
        return prompt
    
    def _optimize_parameters(
        self,
        level: int,
        grade_level: int,
        language: str,
        domain: Optional[str],
        capabilities: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize parameters for the specific model and simplification level.
        
        Args:
            level: Simplification level (1-5)
            grade_level: Target grade level
            language: Language code
            domain: Specific domain
            capabilities: Model capabilities
            parameters: User-provided parameters
            
        Returns:
            Optimized parameter dictionary
        """
        # Start with user parameters
        optimized = parameters.copy()
        
        # Add level-specific parameters if not specified by user
        if "temperature" not in optimized:
            # Higher temperature for more creative simplification at lower levels
            temperature_by_level = {
                1: 0.7,  # Level 1: More creative rewording
                2: 0.6,
                3: 0.5,
                4: 0.4,
                5: 0.3   # Level 5: More deterministic simplification
            }
            optimized["temperature"] = temperature_by_level.get(level, 0.5)
        
        if "top_p" not in optimized:
            # Focus sampling more for higher simplification levels
            top_p_by_level = {
                1: 0.9,  # Level 1: More diverse outputs
                2: 0.85,
                3: 0.8,
                4: 0.75,
                5: 0.7   # Level 5: More focused outputs
            }
            optimized["top_p"] = top_p_by_level.get(level, 0.8)
        
        if "num_beams" not in optimized:
            # More beams for higher precision at higher simplification levels
            beams_by_level = {
                1: 4,  # Level 1: Standard beam search
                2: 4,
                3: 5,
                4: 6,
                5: 8   # Level 5: More careful generation for elementary level
            }
            optimized["num_beams"] = beams_by_level.get(level, 4)
        
        if "length_penalty" not in optimized:
            # Length penalties by level - lower levels might need longer outputs
            # to explain complex concepts, higher levels need shorter outputs
            length_penalty_by_level = {
                1: 1.2,  # Level 1: Slightly encourage longer outputs
                2: 1.1,
                3: 1.0,  # Level 3: Neutral
                4: 0.9,
                5: 0.8   # Level 5: Encourage shorter outputs
            }
            optimized["length_penalty"] = length_penalty_by_level.get(level, 1.0)
        
        # Domain-specific parameter adjustments
        if domain:
            domain_strategy = self.get_domain_strategy(domain)
            if domain == "legal" or "legal" in domain:
                # Legal domain needs careful simplification to preserve meaning
                if "num_beams" not in optimized or optimized["num_beams"] < 6:
                    optimized["num_beams"] = max(optimized.get("num_beams", 4), 6)
                # More conservative parameters for legal text
                if "do_sample" not in optimized:
                    optimized["do_sample"] = level < 4  # Only sample for levels 1-3
                if level >= 4 and "temperature" in optimized:
                    # More deterministic for highest simplification levels
                    optimized["temperature"] = min(optimized["temperature"], 0.4)
            
            elif domain == "medical" or "medical" in domain:
                # Medical domain needs precision in terminology
                if "num_beams" not in optimized or optimized["num_beams"] < 5:
                    optimized["num_beams"] = max(optimized.get("num_beams", 4), 5)
                if "repetition_penalty" not in optimized:
                    optimized["repetition_penalty"] = 1.2  # Avoid repetition
            
            elif domain == "technical" or "technical" in domain:
                # Technical domain needs precision in concept explanation
                if "do_sample" not in optimized:
                    optimized["do_sample"] = level < 4  # Only sample for levels 1-3
        
        # Add language-specific optimizations
        if language != "en":
            # For non-English, we might need different parameters
            if "num_beams" not in optimized or optimized["num_beams"] < 5:
                optimized["num_beams"] = max(optimized.get("num_beams", 4), 5)
        
        # Set max length based on input length and simplification level
        if "max_length" not in optimized:
            # For level 1-2, output might be longer due to explanations
            # For level 4-5, output should be shorter due to simplification
            input_length = len(text.split())
            level_multipliers = {
                1: 1.2,  # Level 1: Might be longer to explain
                2: 1.1,
                3: 1.0,  # Level 3: Similar length
                4: 0.9,
                5: 0.8   # Level 5: Should be shorter
            }
            multiplier = level_multipliers.get(level, 1.0)
            optimized["max_length"] = min(1024, int(input_length * multiplier * 1.5) + 50)
        
        return optimized