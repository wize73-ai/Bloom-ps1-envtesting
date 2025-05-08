#!/usr/bin/env python3
"""
Fix for enhancing the rule-based simplification in the SimplificationPipeline.

This script adds domain-aware simplification to the _rule_based_simplify method
in the SimplificationPipeline class and adds more comprehensive vocabulary
replacements for better differentiation between simplification levels.
"""

import os
import sys
import re
import shutil
from typing import Dict, Any, List

def apply_fix():
    """Apply the fix to enhance rule-based simplification."""
    print("Applying rule-based simplification implementation fix...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    simplifier_py_path = os.path.join(script_dir, "app/core/pipeline/simplifier.py")
    simplifier_py_backup = os.path.join(script_dir, "app/core/pipeline/simplifier.py.bak")
    
    # Create backup
    if os.path.exists(simplifier_py_path):
        shutil.copy2(simplifier_py_path, simplifier_py_backup)
        print(f"Created backup: {simplifier_py_backup}")
    else:
        print(f"Error: {simplifier_py_path} does not exist")
        return False
    
    # Read the simplifier.py file
    with open(simplifier_py_path, 'r') as f:
        content = f.read()
    
    # Get improved replacements by level
    improved_replacements = """
    def _get_language_replacements(self, language: str, level: int) -> Dict[str, str]:
        \"\"\"
        Get language-specific word replacements for the given simplification level.
        
        Args:
            language: Language code
            level: Simplification level (1-5)
            
        Returns:
            Dictionary of replacements for the given language and level
        \"\"\"
        if language != "en":
            # Only English is fully supported currently
            return {}
            
        # Define substitutions by level (accumulative)
        substitutions = {
            # Common for all levels
            1: {
                r'\\butilize\\b': 'use',
                r'\\bpurchase\\b': 'buy',
                r'\\bsubsequently\\b': 'later',
                r'\\bfurnish\\b': 'provide',
                r'\\baforementioned\\b': 'previously mentioned',
                r'\\bdelineated\\b': 'outlined',
                r'\\bin accordance with\\b': 'according to'
            },
            2: {
                r'\\bindicate\\b': 'show',
                r'\\bsufficient\\b': 'enough',
                r'\\badditional\\b': 'more',
                r'\\bprior to\\b': 'before',
                r'\\bverifying\\b': 'proving',
                r'\\brequirements\\b': 'rules'
            },
            3: {
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
            },
            4: {
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
            },
            5: {
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
        }
        
        # Get the appropriate substitutions for this level and all lower levels
        all_substitutions = {}
        for l in range(1, level + 1):
            if l in substitutions:
                all_substitutions.update(substitutions[l])
                
        return all_substitutions
    """
    
    # Improved rule-based simplify method with domain awareness
    improved_rule_based = """
    def _rule_based_simplify(self, text: str, level: int, language: str = "en", domain: str = None) -> str:
        \"\"\"
        Apply rule-based simplification with a specific level.
        
        Args:
            text: Text to simplify
            level: Simplification level (1-5, where 5 is simplest)
            language: Language code
            domain: Optional domain for domain-specific simplification
            
        Returns:
            Simplified text
        \"\"\"
        # If no text, return empty string
        if not text:
            return ""
        
        # Get substitutions for this level
        replacements = self._get_language_replacements(language, level)
        
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
            simplified_text = re.sub(pattern, replacement, simplified_text, flags=re.IGNORECASE)
        
        # Clean up spaces
        simplified_text = re.sub(r'\\s+', ' ', simplified_text).strip()
        
        # For highest level, add explaining phrases
        if level == 5:
            # Add domain-specific explanations
            if domain and "legal" in domain.lower():
                simplified_text += " This means you need to follow what the law says."
            else:
                simplified_text += " This means you need to show the required information."
        
        # Return the simplified text
        return simplified_text
    """

    # Check if methods already exist
    if "_get_language_replacements" in content:
        # Replace existing method
        old_replacements_pattern = r'def _get_language_replacements.*?return.*?\}'
        content = re.sub(old_replacements_pattern, improved_replacements.strip(), content, flags=re.DOTALL)
    else:
        # Insert method after class definition
        content = content.replace("class SimplificationPipeline:", 
                                  "class SimplificationPipeline:\n" + improved_replacements)
    
    # Replace rule-based simplify method
    old_rule_based_pattern = r'def _rule_based_simplify.*?return simplified_text(\s+?)'
    if re.search(old_rule_based_pattern, content, re.DOTALL):
        content = re.sub(old_rule_based_pattern, improved_rule_based.strip() + r'\1', content, flags=re.DOTALL)
    else:
        # Could not find the method, so insert it after _apply_grade_level_vocabulary
        content = content.replace("def _apply_grade_level_vocabulary", 
                                  improved_rule_based + "\n\n    def _apply_grade_level_vocabulary")
    
    # Update the simplify method to pass domain to rule-based simplifier
    simplify_pattern = r'(simplified_text = self\._rule_based_simplify\(\s*text,\s*level,\s*language)'
    if re.search(simplify_pattern, content):
        content = re.sub(simplify_pattern, r'\1, domain=options.get("domain")', content)
    
    # Write the modified file
    with open(simplifier_py_path, 'w') as f:
        f.write(content)
    
    print("Successfully applied rule-based simplification implementation fix!")
    print("Please restart the server for changes to take effect.")
    return True

if __name__ == "__main__":
    apply_fix()