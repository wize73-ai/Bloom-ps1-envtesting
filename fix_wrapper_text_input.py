#!/usr/bin/env python3
"""
Fix for the TranslationModelWrapper input type errors

This script adds proper input validation and type conversion to the
TranslationModelWrapper._preprocess method to prevent text input type errors.
"""

import os
import sys
import re
import shutil
from typing import Dict, Any, List, Optional

def apply_fix():
    """Apply the fix to add proper type validation to TranslationModelWrapper."""
    print("Adding type validation to TranslationModelWrapper._preprocess method...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    wrapper_py_path = os.path.join(script_dir, "app/services/models/wrapper.py")
    wrapper_py_backup = os.path.join(script_dir, "app/services/models/wrapper.py.bak3")
    
    # Create backup
    if os.path.exists(wrapper_py_path):
        shutil.copy2(wrapper_py_path, wrapper_py_backup)
        print(f"Created backup: {wrapper_py_backup}")
    else:
        print(f"Error: {wrapper_py_path} does not exist")
        return False
    
    # Read the wrapper.py file
    with open(wrapper_py_path, 'r') as f:
        content = f.read()
    
    # Find the TranslationModelWrapper._preprocess method
    preprocess_pattern = r'def _preprocess\(self, (.*?)(\).*?):'
    preprocess_match = re.search(preprocess_pattern, content)
    
    if preprocess_match:
        old_param_signature = preprocess_match.group(1)
        old_closing_paren = preprocess_match.group(2)
        
        # Check if the signature is already using ModelInput
        if "input_data: ModelInput" in old_param_signature:
            print("TranslationModelWrapper._preprocess already has proper ModelInput signature.")
            
            # Still need to add type validation - add it right after the method definition
            method_body_pattern = r'def _preprocess\(self, .*?\).*?:[\s\n]+(.+)'
            method_body_match = re.search(method_body_pattern, content, re.DOTALL)
            
            if method_body_match:
                old_method_body = method_body_match.group(1)
                
                # Input validation code to add
                validation_code = """
        # Ensure input_data is a dictionary if it's not already ModelInput
        if isinstance(input_data, dict):
            # Extract text from the input data
            if 'text' in input_data:
                texts = input_data['text']
            else:
                raise ValueError("input_data dictionary must contain 'text' key")
                
            # Ensure texts is a valid text type
            if isinstance(texts, (str, list)):
                # If it's a string, make sure it's not empty
                if isinstance(texts, str) and not texts.strip():
                    texts = " "  # Use a space instead of empty string
                # If it's a list, ensure all elements are strings
                if isinstance(texts, list):
                    texts = [str(t) if not isinstance(t, str) else t for t in texts]
                    # Ensure no empty strings
                    texts = [" " if not t.strip() else t for t in texts]
            else:
                # Convert to string if it's not a string or list
                texts = str(texts)
        elif hasattr(input_data, 'text'):
            # Handle ModelInput or similar object with text attribute
            texts = input_data.text
            # Apply the same validations
            if isinstance(texts, (str, list)):
                if isinstance(texts, str) and not texts.strip():
                    texts = " "
                if isinstance(texts, list):
                    texts = [str(t) if not isinstance(t, str) else t for t in texts]
                    texts = [" " if not t.strip() else t for t in texts]
            else:
                texts = str(texts)
        else:
            # Direct text input (deprecated) - ensure it's a valid string
            texts = str(input_data) if not isinstance(input_data, str) else input_data
            if not texts.strip():
                texts = " "
                
"""
                # Add validation code to the beginning of the method body
                new_method_body = validation_code + old_method_body
                
                # Replace the method body
                content = content.replace(old_method_body, new_method_body, 1)
            else:
                print("Could not find method body to update.")
                return False
        else:
            # Update to proper ModelInput signature
            new_param_signature = "input_data: ModelInput"
            content = content.replace(
                f"def _preprocess(self, {old_param_signature}{old_closing_paren}:", 
                f"def _preprocess(self, {new_param_signature}{old_closing_paren}:"
            )
            
            # Find the method body to update with type validation
            method_body_pattern = r'def _preprocess\(self, .*?\).*?:[\s\n]+(.+)'
            method_body_match = re.search(method_body_pattern, content, re.DOTALL)
            
            if method_body_match:
                old_method_body = method_body_match.group(1)
                
                # Input validation code to add
                validation_code = """
        # Ensure input_data.text is a valid text type
        if hasattr(input_data, 'text'):
            texts = input_data.text
            # Apply validations
            if isinstance(texts, (str, list)):
                if isinstance(texts, str) and not texts.strip():
                    texts = " "  # Use a space instead of empty string
                if isinstance(texts, list):
                    texts = [str(t) if not isinstance(t, str) else t for t in texts]
                    # Ensure no empty strings
                    texts = [" " if not t.strip() else t for t in texts]
            else:
                texts = str(texts)
        else:
            # Direct text input (deprecated) - ensure it's a valid string
            texts = str(input_data) if not isinstance(input_data, str) else input_data
            if not texts.strip():
                texts = " "
                
"""
                # Add validation code to the beginning of the method body
                new_method_body = validation_code + old_method_body
                
                # Replace the method body
                content = content.replace(old_method_body, new_method_body, 1)
            else:
                print("Could not find method body to update.")
                return False
    else:
        print("Could not find TranslationModelWrapper._preprocess method in the file")
        return False
    
    # Write the modified file
    with open(wrapper_py_path, 'w') as f:
        f.write(content)
    
    print("Successfully added type validation to TranslationModelWrapper._preprocess method!")
    print("Please restart the server for changes to take effect.")
    return True

if __name__ == "__main__":
    apply_fix()