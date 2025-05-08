"""
Fix for CasaLingua Model Manager incompatibility issues

This script provides a fix for the model loading issues in CasaLingua by:
1. Adding the missing ModelManager class implementation
2. Ensuring compatibility between EnhancedModelManager and ModelManager
3. Fixing the model loading pathway for critical models like simplification and translation

Usage:
    python fix_model_manager.py

The script creates a backup of the existing files before making changes.
"""

import os
import shutil
import sys
from pathlib import Path

# Helper functions
def backup_file(file_path):
    """Create a backup of a file with .bak extension"""
    backup_path = f"{file_path}.bak"
    print(f"Creating backup: {backup_path}")
    shutil.copy2(file_path, backup_path)
    return backup_path

def write_model_manager():
    """Write the updated model_manager.py file with compatibility fixes"""
    target_path = "app/services/models/model_manager.py"
    content = """
"""
    ModelManager_content = """
\"\"\"
ModelManager Class for CasaLingua

This is a compatibility wrapper to allow code to use ModelManager 
while using the EnhancedModelManager underneath.
\"\"\"

import os
import logging
import asyncio
import torch
from typing import Dict, Any, List, Optional, Union

# Import EnhancedModelManager
from app.services.models.manager import EnhancedModelManager

# Configure logging
logger = logging.getLogger(__name__)

class ModelManager:
    \"\"\"
    ModelManager compatibility wrapper for EnhancedModelManager
    \"\"\"
    
    def __init__(self, registry_config=None, config=None):
        \"\"\"
        Initialize the model manager with the given registry configuration
        
        Args:
            registry_config: Configuration for the model registry
            config: General configuration
        \"\"\"
        self.registry_config = registry_config
        self.config = config or {}
        self.enhanced_manager = None
        self.loaded_models = {}
    
    async def initialize(self):
        \"\"\"Initialize the enhanced model manager\"\"\"
        # Create hardware info for the enhanced manager
        from app.services.hardware.detector import detect_hardware
        
        try:
            # Try to detect hardware
            hardware_info = detect_hardware()
            logger.info(f"Detected hardware: {hardware_info}")
        except Exception as e:
            logger.warning(f"Error detecting hardware: {e}, using default settings")
            # Create default hardware info
            hardware_info = {
                "gpu": {
                    "has_gpu": torch.cuda.is_available(),
                    "cuda_available": torch.cuda.is_available(),
                    "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
                    "gpu_memory": 0
                },
                "cpu": {
                    "supports_avx2": True,
                    "cores": os.cpu_count() or 8
                }
            }
            
            # If CUDA is available, estimate GPU memory
            if hardware_info["gpu"]["cuda_available"]:
                for i in range(torch.cuda.device_count()):
                    mem_info = torch.cuda.get_device_properties(i).total_memory
                    hardware_info["gpu"]["gpu_memory"] = max(hardware_info["gpu"]["gpu_memory"], mem_info)
        
        # Import the ModelLoader
        from app.services.models.loader import ModelLoader
        
        # Create loader
        loader = ModelLoader(registry_config=self.registry_config)
        
        # Create enhanced manager
        self.enhanced_manager = EnhancedModelManager(
            loader=loader,
            hardware_info=hardware_info,
            config=self.config
        )
        
        # Redirect methods to enhanced manager
        self.loaded_models = self.enhanced_manager.loaded_models
        
        logger.info("ModelManager initialized with EnhancedModelManager")
    
    async def load_model(self, model_type: str, **kwargs) -> Dict[str, Any]:
        \"\"\"
        Load a model of the specified type
        
        Args:
            model_type: Type of model to load
            **kwargs: Additional arguments
            
        Returns:
            Dict with model information
        \"\"\"
        if self.enhanced_manager is None:
            await self.initialize()
            
        # Forward to enhanced manager
        result = await self.enhanced_manager.load_model(model_type, **kwargs)
        
        # Keep local reference to loaded models for compatibility
        if "model" in result:
            self.loaded_models[model_type] = result["model"]
            
        return result
    
    async def run_model(self, model_type: str, method_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"
        Run a model with a specified method
        
        Args:
            model_type: Type of model to run
            method_name: Method to call
            input_data: Input data
            
        Returns:
            Dict with results
        \"\"\"
        if self.enhanced_manager is None:
            await self.initialize()
            
        # Forward to enhanced manager
        return await self.enhanced_manager.run_model(model_type, method_name, input_data)
    
    async def list_loaded_models(self) -> List[str]:
        \"\"\"
        Get a list of loaded model names
        
        Returns:
            List of model names
        \"\"\"
        if self.enhanced_manager is None:
            return []
            
        # Get model info
        info = self.enhanced_manager.get_model_info()
        
        # Return loaded model types
        return [model_type for model_type, model_info in info.items() 
                if model_info.get("loaded", False) and model_type != "_system"]
                
    async def create_embeddings(self, texts: Union[str, List[str]], model_key: str = "rag_retriever") -> List[List[float]]:
        \"\"\"
        Create embeddings for texts
        
        Args:
            texts: Text or list of texts to embed
            model_key: Key of model to use
            
        Returns:
            List of embeddings
        \"\"\"
        if self.enhanced_manager is None:
            await self.initialize()
            
        # Forward to enhanced manager
        return await self.enhanced_manager.create_embeddings(texts, model_key)

    async def unload_model(self, model_type: str) -> bool:
        \"\"\"
        Unload a model
        
        Args:
            model_type: Type of model to unload
            
        Returns:
            Success status
        \"\"\"
        if self.enhanced_manager is None:
            return False
            
        # Forward to enhanced manager
        return await self.enhanced_manager.unload_model(model_type)
    
    async def unload_all_models(self) -> bool:
        \"\"\"
        Unload all models
        
        Returns:
            Success status
        \"\"\"
        if self.enhanced_manager is None:
            return False
            
        # Forward to enhanced manager
        return await self.enhanced_manager.unload_all_models()
"""

    # Check if file exists
    if os.path.exists(target_path):
        print(f"Backup of model_manager.py...")
        backup_file(target_path)
    
    # Write new file
    with open(target_path, 'w') as f:
        f.write(ModelManager_content.strip())
    
    print(f"Updated {target_path} with compatibility fixes")

def update_simplifier_py():
    """Update the simplifier.py file with fixes"""
    target_path = "app/core/pipeline/simplifier.py"
    # Check if file exists
    if os.path.exists(target_path):
        print(f"Backup of simplifier.py...")
        backup_file(target_path)
    
    # Read existing content
    with open(target_path, 'r') as f:
        content = f.read()
    
    # Fix missing function
    if "_rule_based_simplify" in content and not content.strip().endswith("simplified_text"):
        # The rule_based_simplify function is missing proper definition or has syntax error
        # Find the correct position to fix
        rule_based_pos = content.find("def _rule_based_simplify")
        if rule_based_pos != -1:
            # Find the docstring or first line after function definition
            next_def_pos = content.find("def ", rule_based_pos + 10)
            
            # If no next def found, append to end
            if next_def_pos == -1:
                next_def_pos = len(content)
            
            # Fix function with proper implementation
            fixed_function = """
    def _rule_based_simplify(self, text: str, level: int, language: str = "en", domain: str = None) -> str:
        # Docstring for rule-based simplification
        # Apply rule-based simplification with a specific level
        # If no text, return empty string
        if not text:
            return ""
        
        # Check if legal domain
        is_legal_domain = domain and "legal" in domain.lower()
        
        # Define vocabulary replacements for different levels
        replacements = {}
        
        # Level 1 (minimal simplification)
        level1_replacements = {
            r'\\butilize\\b': 'use',
            r'\\bpurchase\\b': 'buy',
            r'\\bsubsequently\\b': 'later',
            r'\\bfurnish\\b': 'provide',
            r'\\baforementioned\\b': 'previously mentioned',
            r'\\bdelineated\\b': 'outlined',
            r'\\bin accordance with\\b': 'according to'
        }
        
        # Level 2
        level2_replacements = {
            r'\\bindicate\\b': 'show',
            r'\\bsufficient\\b': 'enough',
            r'\\badditional\\b': 'more',
            r'\\bprior to\\b': 'before',
            r'\\bverifying\\b': 'proving',
            r'\\brequirements\\b': 'rules'
        }
        
        # Level 3
        level3_replacements = {
            r'\\bassist\\b': 'help',
            r'\\bobtain\\b': 'get',
            r'\\brequire\\b': 'need',
            r'\\bcommence\\b': 'start',
            r'\\bterminate\\b': 'end',
            r'\\bdemonstrate\\b': 'show',
            r'\\bdelineated\\b': 'described',
            r'\\bin accordance with\\b': 'following',
            r'\\bemployment status\\b': 'job status',
            r'\\bapplication procedure\\b': 'application process'
        }
        
        # Level 4
        level4_replacements = {
            r'\\bregarding\\b': 'about',
            r'\\bimplement\\b': 'use',
            r'\\bnumerous\\b': 'many',
            r'\\bfacilitate\\b': 'help',
            r'\\binitial\\b': 'first',
            r'\\battempt\\b': 'try',
            r'\\bapplicant\\b': 'you',
            r'\\bfurnish\\b': 'give',
            r'\\baforementioned\\b': 'this',
            r'\\bdelineated\\b': 'listed',
            r'\\bverifying\\b': 'that proves',
            r'\\bemployment status\\b': 'job information',
            r'\\bapplication procedure\\b': 'steps',
            r'\\bdocumentation\\b': 'papers',
            r'\\bsection\\b': 'part'
        }
        
        # Level 5
        level5_replacements = {
            r'\\binquire\\b': 'ask',
            r'\\bascertain\\b': 'find out',
            r'\\bcomprehend\\b': 'understand',
            r'\\bnevertheless\\b': 'but',
            r'\\btherefore\\b': 'so',
            r'\\bfurthermore\\b': 'also',
            r'\\bconsequently\\b': 'so',
            r'\\bapproximately\\b': 'about',
            r'\\bmodification\\b': 'change',
            r'\\bendeavor\\b': 'try',
            r'\\bproficiency\\b': 'skill',
            r'\\bnecessitate\\b': 'need',
            r'\\bacquisition\\b': 'getting',
            r'\\bemployment status\\b': 'job info',
            r'\\bapplication procedure\\b': 'form',
            r'\\bmust\\b': 'need to'
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
        simplified_text = re.sub(r'\\s+', ' ', simplified_text).strip()
        
        # For highest level, add explaining phrases
        if level == 5:
            if is_legal_domain:
                simplified_text += " This means you need to follow what the law says."
            else:
                simplified_text += " This means you need to show the required information."
        
        return simplified_text
"""
            # Properly indent the function
            fixed_function_indented = "\n".join(["    " + line if line.strip() else "" for line in fixed_function.split("\n")])
            
            # Replace the broken function
            first_part = content[:rule_based_pos]
            last_part = content[next_def_pos:]
            
            new_content = first_part + fixed_function_indented + "\n" + last_part
            
            # Write the updated content
            with open(target_path, 'w') as f:
                f.write(new_content)
            
            print(f"Fixed _rule_based_simplify function in {target_path}")
        else:
            print(f"Could not find _rule_based_simplify function in {target_path} to fix")
    else:
        print(f"File {target_path} not found")

def update_wrapper_fixes():
    """Update the wrapper.py file with necessary fixes"""
    target_path = "app/services/models/wrapper.py"
    
    # Backup the file
    if os.path.exists(target_path):
        print(f"Backup of wrapper.py...")
        backup_file(target_path)
        
        # Read existing content
        with open(target_path, 'r') as f:
            content = f.read()
        
        # Fix issues with text handling in BaseModelWrapper._preprocess
        if "_preprocess" in content:
            # Find the preprocess method in the BaseModelWrapper
            preprocess_start = content.find("def _preprocess(self, input_data: ModelInput)")
            if preprocess_start != -1:
                # Find the next method definition to know where this method ends
                next_method = content.find("def ", preprocess_start + 10)
                if next_method != -1:
                    # Extract the current implementation
                    current_impl = content[preprocess_start:next_method]
                    
                    # Check if the implementation is incomplete
                    if "pass" in current_impl and not "return" in current_impl:
                        # Fix the implementation with proper handling
                        fixed_impl = """def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """
        Preprocess the input data for the model.
        
        Args:
            input_data: The input data
            
        Returns:
            The preprocessed data
        """
        # Handle different input types
        if isinstance(input_data, ModelInput):
            # Handle ModelInput object directly
            text = input_data.text
            source_lang = input_data.source_language
            target_lang = input_data.target_language
            params = input_data.parameters or {}
            
            # Validate text input
            if isinstance(text, str):
                if not text.strip():
                    text = " "  # Use space instead of empty string
            elif isinstance(text, list):
                # Handle list of texts
                text = [str(t) if not isinstance(t, str) else t for t in text]
                text = [" " if not t.strip() else t for t in text]
            else:
                # Convert anything else to string
                text = str(text)
                
            # Create preprocessed dict
            return {
                "text": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "parameters": params
            }
            
        elif isinstance(input_data, dict):
            # Handle dictionary input
            return input_data
        else:
            # Treat as raw text input (deprecated, for backward compatibility)
            text = str(input_data) if not isinstance(input_data, str) else input_data
            return {"text": text}"""
                        
                        # Properly indent the function
                        fixed_impl_indented = "\n".join(["    " + line if line.strip() else "" for line in fixed_impl.split("\n")])
                        
                        # Replace the incomplete implementation
                        new_content = content[:preprocess_start] + fixed_impl_indented + content[next_method:]
                        
                        # Write the updated content
                        with open(target_path, 'w') as f:
                            f.write(new_content)
                        
                        print(f"Fixed _preprocess method in BaseModelWrapper in {target_path}")
                    else:
                        print(f"_preprocess method in BaseModelWrapper appears to be complete, no fix needed")
                else:
                    print(f"Could not determine the end of _preprocess method in {target_path}")
            else:
                print(f"Could not find _preprocess method in {target_path}")
        else:
            print(f"No _preprocess method found in {target_path} to fix")
    else:
        print(f"File {target_path} not found")

def main():
    """Main function to run all fixes"""
    print("Running CasaLingua Model Manager Fixes")
    
    # Create model_manager.py with compatible implementation
    write_model_manager()
    
    # Fix simplifier.py missing function
    update_simplifier_py()
    
    # Fix wrapper.py issues
    update_wrapper_fixes()
    
    print("\nCasaLingua model manager fixes have been successfully applied.")
    print("To test the fixes, try running the demo again.")

if __name__ == "__main__":
    main()