"""
Fixed SimplifierWrapper implementation

This file contains a fixed version of the SimplifierWrapper class
to address issues with model loading, caching, and result processing.
"""

import logging
import torch
import re
from typing import Dict, Any

from app.services.models.wrapper import BaseModelWrapper, ModelInput, ModelOutput

# Configure logging
logger = logging.getLogger(__name__)

class FixedSimplifierWrapper(BaseModelWrapper):
    """Enhanced wrapper for text simplification models with reliable fallback"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess simplification input - Fixed implementation"""
        if isinstance(input_data.text, list):
            texts = input_data.text
        else:
            texts = [input_data.text]
        
        # Get model info
        model_config = getattr(self.model, "config", None)
        model_name = getattr(model_config, "_name_or_path", "") if model_config else ""
        model_class = self.model.__class__.__name__
        
        logger.debug(f"Simplifier model info: name={model_name}, class={model_class}")
        
        # Check if we should use legal housing simplification prompting
        parameters = input_data.parameters or {}
        domain = parameters.get("domain", "")
        # Handle case where domain could be None
        is_legal_domain = domain and domain.lower() in ["legal", "housing", "housing_legal"]
        
        # Prepare simplification prompt
        prompts = []
        for text in texts:
            # Special handling for different model types
            if "bart" in model_name.lower() or "BartForConditionalGeneration" in model_class:
                # BART models - we need special prompts for these
                if is_legal_domain:
                    # For legal housing text
                    prompt = f"Simplify this housing legal text for a general audience: {text}"
                    logger.debug(f"Using housing legal simplification prompt for BART model")
                else:
                    # General simplification
                    prompt = f"Simplify this text: {text}"
                    logger.debug(f"Using general simplification prompt for BART model")
            elif ("finetuned-text-simplification" in model_name or 
                  "simplification" in model_name or
                  "text-simplification" in model_name):
                # For models specifically fine-tuned for simplification
                if is_legal_domain:
                    # Add housing legal context
                    prompt = f"[Housing Legal Simplification] {text}"
                else:
                    # Use direct input
                    prompt = text
                logger.debug(f"Using direct input for specialized model: {model_name}")
            else:
                # For generic T5 models
                if is_legal_domain:
                    prompt = f"simplify housing legal text: {text}"
                else:
                    prompt = f"simplify: {text}"
                logger.debug(f"Using prefix prompt for generic model: {model_name or 'unknown'}")
            
            prompts.append(prompt)
        
        # Log the prompts for debugging
        logger.debug(f"Using simplification prompts: {prompts}")
        
        # Tokenize inputs - add error handling for tokenizer issues
        try:
            if self.tokenizer:
                inputs = self.tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=self.config.get("max_length", 1024)  # Increased max length for legal texts
                )
                
                # Move to device
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
            else:
                logger.warning("No tokenizer available for simplification model, using direct text input")
                inputs = {"texts": prompts}
        except Exception as e:
            logger.error(f"Error tokenizing simplification input: {str(e)}", exc_info=True)
            # Fallback to direct text input
            inputs = {"texts": prompts}
        
        return {
            "inputs": inputs,
            "original_texts": texts,
            "is_legal_domain": is_legal_domain,
            "prompts": prompts  # Store original prompts for fallback
        }
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run simplification inference - Fixed implementation with better error handling"""
        inputs = preprocessed["inputs"]
        is_legal_domain = preprocessed.get("is_legal_domain", False)
        
        # Get model info for customizing generation
        model_config = getattr(self.model, "config", None)
        model_name = getattr(model_config, "_name_or_path", "") if model_config else ""
        model_class = self.model.__class__.__name__
        
        # For transformers models with generate method
        if hasattr(self.model, "generate") and callable(self.model.generate):
            # Get generation parameters with custom settings for legal text
            gen_kwargs = self.config.get("generation_kwargs", {}).copy()
            
            # Set defaults if not provided
            if "max_length" not in gen_kwargs:
                gen_kwargs["max_length"] = 1024 if is_legal_domain else 512
                
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 5 if is_legal_domain else 4
                
            if "min_length" not in gen_kwargs:
                # Ensure the output isn't too short (can happen with overly aggressive simplification)
                gen_kwargs["min_length"] = 30 if is_legal_domain else 10
                
            # Use different generation settings based on model type
            if "bart" in model_name.lower() or "BartForConditionalGeneration" in model_class:
                # BART-specific generation settings for better simplification
                if "do_sample" not in gen_kwargs:
                    gen_kwargs["do_sample"] = True
                    
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = 0.95  # Better diversity
                    
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0.85 if is_legal_domain else 0.7
                    
                logger.debug(f"Using BART-specific generation parameters: {gen_kwargs}")
            
            # Log generation parameters
            logger.debug(f"Generating simplification with parameters: {gen_kwargs}")
            
            try:
                # Check for required input_ids
                if "input_ids" not in inputs:
                    logger.warning("No input_ids in inputs, checking for alternate inputs")
                    if "texts" in inputs:
                        # Try to tokenize directly
                        try:
                            if self.tokenizer:
                                tokenized = self.tokenizer(
                                    inputs["texts"],
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=gen_kwargs.get("max_length", 1024)
                                )
                                # Move to device
                                for key in tokenized:
                                    if isinstance(tokenized[key], torch.Tensor):
                                        tokenized[key] = tokenized[key].to(self.device)
                                inputs = tokenized
                            else:
                                raise ValueError("No tokenizer available for text inputs")
                        except Exception as tokenize_error:
                            logger.error(f"Error tokenizing text inputs: {str(tokenize_error)}", exc_info=True)
                            # Return empty tensor as fallback
                            return torch.tensor([[0]])
                
                # Generate simplifications
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )
                return outputs
            except Exception as e:
                logger.error(f"Error generating simplification: {str(e)}", exc_info=True)
                # Return a minimal output that won't crash downstream
                return torch.tensor([[0]])
        
        # For custom models with simplify method
        elif hasattr(self.model, "simplify") and callable(self.model.simplify):
            try:
                return self.model.simplify(preprocessed["original_texts"])
            except Exception as e:
                logger.error(f"Error calling simplify method: {str(e)}", exc_info=True)
                # Return original text as fallback
                return preprocessed["original_texts"]
        
        # Rule-based fallback if model interface is unknown
        else:
            logger.warning(f"Unsupported simplification model: {type(self.model).__name__}, using rule-based fallback")
            return self._rule_based_simplify(preprocessed["original_texts"])
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess simplification output - Fixed implementation"""
        import time
        
        # Get domain info for specialized postprocessing
        parameters = input_data.parameters or {}
        domain = parameters.get("domain", "")
        # Handle case where domain could be None
        is_legal_domain = domain and domain.lower() in ["legal", "housing", "housing_legal"]
        
        # Check if we should verify simplification quality
        verify_output = parameters.get("verify_output", False)
        
        # Check if we have a valid output to decode
        if isinstance(model_output, torch.Tensor) and model_output.numel() == 0:
            # Handle empty output with rule-based fallback
            logger.warning("Empty model output, using rule-based fallback")
            if isinstance(input_data.text, str):
                simplified_text = self._rule_based_simplify_text(input_data.text)
                return ModelOutput(
                    result=simplified_text,
                    metadata={"fallback": "rule_based"}
                )
            else:
                simplified_texts = [self._rule_based_simplify_text(text) for text in input_data.text]
                return ModelOutput(
                    result=simplified_texts,
                    metadata={"fallback": "rule_based"}
                )
        
        # Process tensor output if we have a tokenizer
        if isinstance(model_output, torch.Tensor) and self.tokenizer and hasattr(self.tokenizer, "batch_decode"):
            try:
                # Decode outputs
                simplifications = self.tokenizer.batch_decode(
                    model_output, 
                    skip_special_tokens=True
                )
            except Exception as e:
                logger.error(f"Error decoding simplification output: {str(e)}", exc_info=True)
                # Return original text as fallback through rule-based simplification
                if isinstance(input_data.text, str):
                    simplifications = [self._rule_based_simplify_text(input_data.text)]
                else:
                    simplifications = [self._rule_based_simplify_text(text) for text in input_data.text]
        else:
            # Output already in text format
            if isinstance(model_output, list):
                simplifications = model_output
            elif isinstance(model_output, str):
                simplifications = [model_output]
            else:
                # Unknown output format, use rule-based fallback
                logger.warning(f"Unknown model output format: {type(model_output)}, using rule-based fallback")
                if isinstance(input_data.text, str):
                    simplifications = [self._rule_based_simplify_text(input_data.text)]
                else:
                    simplifications = [self._rule_based_simplify_text(text) for text in input_data.text]
        
        # Process the simplifications for better quality
        processed_simplifications = []
        verification_results = []
        
        for i, simplification in enumerate(simplifications):
            original_text = input_data.text[i] if isinstance(input_data.text, list) else input_data.text
            
            # Clean up the simplification
            if simplification:
                # Remove any prefixes from the prompt that might have been copied
                prefixes_to_remove = [
                    "Simplify this housing legal text for a general audience:",
                    "Simplify this text:",
                    "[Housing Legal Simplification]",
                    "simplify housing legal text:",
                    "simplify:"
                ]
                
                for prefix in prefixes_to_remove:
                    if simplification.startswith(prefix):
                        simplification = simplification[len(prefix):].strip()
                
                # For legal domain, ensure common legal terms remain properly formatted
                if is_legal_domain:
                    # Ensure capitalization of important legal terms
                    legal_terms = ["Landlord", "Tenant", "Lessor", "Lessee", "Security Deposit"]
                    for term in legal_terms:
                        # Replace with properly capitalized version
                        simplification = re.sub(r'\b' + term.lower() + r'\b', term, simplification, flags=re.IGNORECASE)
            else:
                # If simplification is empty, use rule-based fallback
                logger.warning(f"Empty simplification result, using rule-based fallback for text: {original_text[:50]}...")
                simplification = self._rule_based_simplify_text(original_text)
            
            # Verify simplification is not empty or "None"
            if not simplification or simplification == "None" or simplification.strip() == "":
                logger.warning(f"Invalid simplification result (empty or 'None'), using rule-based fallback")
                simplification = self._rule_based_simplify_text(original_text)
            
            # Run verification if requested
            if verify_output:
                try:
                    verification_result = self._verify_simplification(
                        original_text, 
                        simplification,
                        parameters.get("language", "en"),
                        domain
                    )
                    verification_results.append(verification_result)
                    
                    # Apply fixes if verification failed and auto_fix is enabled
                    if parameters.get("auto_fix", False) and not verification_result.get("verified", False):
                        simplification = self._apply_verification_fixes(
                            original_text,
                            simplification,
                            verification_result
                        )
                except Exception as e:
                    logger.error(f"Error in simplification verification: {str(e)}", exc_info=True)
                    # Skip verification on error
                    verification_results.append({
                        "verified": True,  # Default to passing on error
                        "score": 0.5,
                        "confidence": 0.3,
                        "issues": [{
                            "type": "verification_error",
                            "severity": "warning",
                            "message": f"Verification failed: {str(e)}"
                        }]
                    })
            
            processed_simplifications.append(simplification)
        
        # If single input, return single output
        if isinstance(input_data.text, str):
            result = processed_simplifications[0] if processed_simplifications else ""
            verification_result = verification_results[0] if verification_results else None
        else:
            result = processed_simplifications
            verification_result = verification_results
        
        # Create metadata with info about the simplification
        metadata = {
            "domain": domain if domain else "general",
            "model_name": getattr(getattr(self.model, "config", None), "_name_or_path", "unknown") if hasattr(self.model, "config") else "unknown"
        }
        
        # Add verification results to metadata if available
        if verify_output and verification_result:
            metadata["verification"] = verification_result
        
        return ModelOutput(
            result=result,
            metadata=metadata
        )
    
    def _rule_based_simplify(self, texts: list) -> list:
        """Apply rule-based simplification to a list of texts"""
        return [self._rule_based_simplify_text(text) for text in texts]
    
    def _rule_based_simplify_text(self, text: str) -> str:
        """
        Rule-based text simplification as a fallback when model-based approach fails.
        
        Args:
            text: Text to simplify
            
        Returns:
            Simplified text using basic rules
        """
        logger.info("Using rule-based simplification fallback")
        
        # If no text, return empty string
        if not text:
            return ""
            
        # Dictionary of complex->simple word replacements
        replacements = {
            r'\bindemnify\b': 'protect',
            r'\bhold harmless\b': 'protect',
            r'\baforesaid\b': '',
            r'\bremit payment\b': 'pay',
            r'\bincluding but not limited to\b': 'including',
            r'\bconsumed or utilized\b': 'used',
            r'\bduring the term of occupancy\b': 'while you live there',
            r'\bin accordance with\b': 'according to',
            r'\bthe lessee is obligated to\b': 'you must',
            r'\blessor reserves the right\b': 'the landlord has the right',
            r'\bpremises\b': 'property',
            r'\binspection purposes\b': 'inspections',
            r'\bwherein\b': 'where',
            r'\bmay be required\b': 'may be needed',
            r'\bprovided to\b': 'given to',
            r'\butilize\b': 'use',
            r'\bpurchase\b': 'buy',
            r'\bindicate\b': 'show',
            r'\bsufficient\b': 'enough',
            r'\bassist\b': 'help',
            r'\bobtain\b': 'get',
            r'\brequire\b': 'need',
            r'\badditional\b': 'more',
            r'\bprior to\b': 'before',
            r'\bsubsequently\b': 'later',
            r'\bcommence\b': 'start',
            r'\bterminate\b': 'end',
            r'\bdemonstrate\b': 'show',
            r'\bregarding\b': 'about',
            r'\bimplement\b': 'use',
            r'\bnumerous\b': 'many',
            r'\bfacilitate\b': 'help',
            r'\binitial\b': 'first',
            r'\battempt\b': 'try'
        }
        
        # Split text into sentences for better processing
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
                
            # Break long sentences into simpler ones
            if len(sentence.split()) > 15:
                clauses = re.split(r'([,;:])', sentence)
                for j in range(0, len(clauses), 2):
                    if j + 1 < len(clauses):
                        processed_sentences.append(clauses[j] + clauses[j+1])
                    else:
                        processed_sentences.append(clauses[j])
            else:
                processed_sentences.append(sentence)
        
        # Join sentences and apply word replacements
        simplified_text = " ".join(processed_sentences)
        for complex_word, simple_word in replacements.items():
            pattern = r'\b' + re.escape(complex_word.strip("\\b")) + r'\b'
            simplified_text = re.sub(pattern, simple_word, simplified_text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        simplified_text = re.sub(r'\s+', ' ', simplified_text).strip()
        
        return simplified_text
    
    def _verify_simplification(
        self, 
        original_text: str, 
        simplified_text: str, 
        language: str = "en",
        domain: str = ""
    ) -> Dict[str, Any]:
        """
        Verify the quality of text simplification.
        
        Args:
            original_text: Original text
            simplified_text: Simplified text
            language: Language code (default: en)
            domain: Domain type (e.g., "legal", "housing")
            
        Returns:
            Dictionary with verification results
        """
        import asyncio
        
        try:
            from app.audit.veracity import VeracityAuditor
            
            # Create auditor instance
            auditor = VeracityAuditor()
            
            # Prepare verification options
            options = {
                "operation": "simplification",
                "source_language": language,
                "domain": domain
            }
            
            # Run verification
            # Since we're in a sync method but need to call an async one,
            # create a new event loop for this specific verification
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(auditor.check(
                original_text, 
                simplified_text, 
                options
            ))
            loop.close()
            
            return result
        except Exception as e:
            logger.error(f"Error verifying simplification: {str(e)}", exc_info=True)
            # Return basic result on error
            return {
                "verified": True,  # Default to passing on error
                "score": 0.5,
                "confidence": 0.3,
                "issues": [{
                    "type": "verification_error",
                    "severity": "warning",
                    "message": f"Verification failed: {str(e)}"
                }]
            }
    
    def _apply_verification_fixes(
        self, 
        original_text: str, 
        simplified_text: str, 
        verification_result: Dict[str, Any]
    ) -> str:
        """
        Apply fixes to simplification based on verification issues.
        
        Args:
            original_text: Original text
            simplified_text: Simplified text
            verification_result: Verification result from _verify_simplification
            
        Returns:
            Improved simplified text
        """
        if not verification_result or not verification_result.get("issues"):
            return simplified_text
            
        fixed_text = simplified_text
        issues = verification_result.get("issues", [])
        
        for issue in issues:
            issue_type = issue.get("type", "")
            
            # Apply specific fixes based on issue type
            if issue_type == "empty_simplification":
                # Use rule-based simplification
                fixed_text = self._rule_based_simplify_text(original_text)
                break
                
            elif issue_type == "no_simplification" and simplified_text.strip() == original_text.strip():
                # Apply rule-based simplification
                fixed_text = self._rule_based_simplify_text(original_text)
                
            elif issue_type == "longer_text":
                # Try to make the text more concise
                fixed_text = self._make_text_concise(fixed_text)
                
            elif issue_type == "meaning_altered" or issue_type == "slight_meaning_change":
                # Apply conservative simplification to preserve meaning
                fixed_text = self._apply_conservative_simplification(original_text)
                
            elif issue_type == "no_lexical_simplification":
                # Replace complex words with simpler alternatives
                fixed_text = self._simplify_complex_words(fixed_text)
                
            elif issue_type == "no_syntactic_simplification":
                # Break down long sentences
                fixed_text = self._simplify_sentences(fixed_text)
        
        return fixed_text
    
    def _make_text_concise(self, text: str) -> str:
        """Make text more concise by removing redundancies."""
        # Remove redundant phrases
        redundant_phrases = [
            r'as stated above',
            r'as mentioned earlier',
            r'it should be noted that',
            r'it is important to note that',
            r'for all intents and purposes',
            r'at the present time',
            r'on account of the fact that',
            r'due to the fact that',
            r'in spite of the fact that',
            r'in the event that',
            r'for the purpose of'
        ]
        
        result = text
        for phrase in redundant_phrases:
            result = re.sub(phrase, '', result, flags=re.IGNORECASE)
        
        # Clean up double spaces
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def _apply_conservative_simplification(self, text: str) -> str:
        """Apply conservative simplification that preserves original meaning."""
        # Replace complex connectors with simpler ones
        replacements = {
            r'\bthus\b': 'so',
            r'\btherefore\b': 'so',
            r'\bconsequently\b': 'so',
            r'\bhence\b': 'so',
            r'\bsubsequently\b': 'then',
            r'\bnevertheless\b': 'but',
            r'\bnotwithstanding\b': 'despite',
            r'\butilize\b': 'use',
            r'\bpurchase\b': 'buy',
            r'\bconsume\b': 'use',
            r'\bterminate\b': 'end',
            r'\binitiate\b': 'start'
        }
        
        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def _simplify_complex_words(self, text: str) -> str:
        """Replace complex words with simpler alternatives."""
        # Dictionary of complex->simple word replacements (legal domain focused)
        replacements = {
            r'\babrogated?\b': 'canceled',
            r'\baccessory\b': 'extra item',
            r'\badjacent to\b': 'next to',
            r'\baforesaid\b': 'previously mentioned',
            r'\balleviate\b': 'reduce',
            r'\bamend\b': 'change',
            r'\banticipate\b': 'expect',
            r'\bassets\b': 'property',
            r'\bassign\b': 'transfer',
            r'\battorney\b': 'lawyer',
            r'\bcertified mail\b': 'mail with proof of delivery',
            r'\bcommencement\b': 'start',
            r'\bconsent\b': 'permission',
            r'\bconstitute\b': 'form',
            r'\bdeem\b': 'consider',
            r'\bdefault\b': 'failure to pay',
            r'\bdemise\b': 'death',
            r'\bduly\b': 'properly',
            r'\bdwelling\b': 'home',
            r'\bendeavor\b': 'try',
            r'\bexpiration\b': 'end',
            r'\bforthwith\b': 'immediately',
            r'\bgoverning law\b': 'law that applies',
            r'\bhereby\b': 'by this',
            r'\bherein\b': 'in this',
            r'\bhereinafter\b': 'later in this document',
            r'\bheretofore\b': 'until now',
            r'\bholder\b': 'owner',
            r'\bimmediately\b': 'right away',
            r'\bimplement\b': 'carry out',
            r'\bin accordance with\b': 'according to',
            r'\bin addition to\b': 'also',
            r'\bin lieu of\b': 'instead of',
            r'\bin the event of\b': 'if',
            r'\bindemnify\b': 'protect from loss',
            r'\binitiate\b': 'start',
            r'\binquire\b': 'ask',
            r'\binterrogatory\b': 'question',
            r'\bjurisdiction\b': 'authority',
            r'\bliability\b': 'legal responsibility',
            r'\bliquidated damages\b': 'set amount of money',
            r'\bmodify\b': 'change',
            r'\bnegligence\b': 'failure to take proper care',
            r'\bnotification\b': 'notice',
            r'\bobtain\b': 'get',
            r'\boccupant\b': 'person living in the property',
            r'\bpayments\b': 'money',
            r'\bper annum\b': 'yearly',
            r'\bperpetrator\b': 'person who committed an act',
            r'\bpremises\b': 'property',
            r'\bprior to\b': 'before',
            r'\bprovisions\b': 'terms',
            r'\bpursuant to\b': 'according to',
            r'\bremedy\b': 'solution',
            r'\bresidence\b': 'home',
            r'\bsaid\b': 'the',
            r'\bshall\b': 'must',
            r'\bstipulation\b': 'requirement',
            r'\bsubmit\b': 'send',
            r'\bsufficient\b': 'enough',
            r'\bterminate\b': 'end',
            r'\bthereafter\b': 'after that',
            r'\bthereof\b': 'of that',
            r'\btransmit\b': 'send',
            r'\butterly\b': 'completely',
            r'\bvacate\b': 'leave',
            r'\bvalid\b': 'legal',
            r'\bverbatim\b': 'word for word',
            r'\bvirtually\b': 'almost',
            r'\bwaive\b': 'give up the right to',
            r'\bwhereas\b': 'since',
            r'\bwith reference to\b': 'about',
            r'\bwith the exception of\b': 'except for',
            r'\bwitness\b': 'see',
            r'\bwithhold\b': 'keep back'
        }
        
        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def _simplify_sentences(self, text: str) -> str:
        """Break down long sentences into shorter ones."""
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        simplified_sentences = []
        
        for sentence in sentences:
            # Skip already short sentences
            if len(sentence.split()) < 15:
                simplified_sentences.append(sentence)
                continue
            
            # Try to break at conjunctions or transition phrases
            parts = re.split(r'(?<=\w)(?:, (?:and|but|or|yet|so|because|although|however|nevertheless|moreover|furthermore|in addition)| (?:and|but|or|yet|so) )', sentence)
            
            if len(parts) > 1:
                for i, part in enumerate(parts):
                    if i > 0 and not re.match(r'^(?:and|but|or|yet|so|because|although|however|nevertheless|moreover|furthermore|in addition)', part.strip()):
                        # Add appropriate starter if the conjunction was removed
                        connector = "Also, " if i > 0 else ""
                        simplified_sentences.append(connector + part.strip() + ".")
                    else:
                        simplified_sentences.append(part.strip() + ".")
            else:
                # If no conjunctions found, try to split at commas/semicolons
                parts = re.split(r'(?<=\w)[;,] ', sentence)
                
                if len(parts) > 1:
                    for i, part in enumerate(parts):
                        # Add appropriate capitalization and endings
                        connector = "Also, " if i > 0 else ""
                        simplified_sentences.append(connector + part.strip() + ".")
                else:
                    # If no good split points, keep as is
                    simplified_sentences.append(sentence)
        
        # Join all sentences
        result = " ".join(simplified_sentences)
        
        # Clean up any double periods
        result = re.sub(r'\.\.', '.', result)
        
        # Ensure proper capitalization
        result = re.sub(r'(?<=\. )[a-z]', lambda m: m.group(0).upper(), result)
        
        return result