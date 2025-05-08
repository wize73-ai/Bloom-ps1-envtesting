#!/usr/bin/env python3
"""
Simple script to directly enhance the rule-based simplification in the app.
"""

import os
import sys
import re
import shutil

def apply_enhanced_simplifier():
    """Apply enhanced rule-based simplification."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    simplifier_path = os.path.join(project_root, "app/core/pipeline/simplifier.py")
    backup_path = os.path.join(project_root, "app/core/pipeline/simplifier.py.bak2")
    
    # Create backup
    shutil.copy2(simplifier_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the file
    with open(simplifier_path, 'r') as f:
        content = f.read()
    
    # Find _rule_based_simplify method
    rule_based_pattern = r'def _rule_based_simplify\(self,.*?return simplified_text'
    rule_based_match = re.search(rule_based_pattern, content, re.DOTALL)
    
    if rule_based_match:
        # Extract the method
        rule_based_method = rule_based_match.group(0)
        
        # Create enhanced method with domain awareness
        enhanced_method = """def _rule_based_simplify(self, text: str, level: int, language: str = "en", domain: str = None) -> str:
        """
        enhanced_method += """
        Apply rule-based simplification with a specific level.
        
        Args:
            text: Text to simplify
            level: Simplification level (1-5, where 5 is simplest)
            language: Language code
            domain: Optional domain for domain-specific simplification
            
        Returns:
            Simplified text
        """
        enhanced_method += """
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
        
        return simplified_text"""
        
        # Replace escaped backslashes with single backslashes
        enhanced_method = enhanced_method.replace('\\\\b', '\\b')
        enhanced_method = enhanced_method.replace('\\\\s', '\\s')
        
        # Replace the method in the content
        new_content = content.replace(rule_based_method, enhanced_method)
        
        # Fix calls to _rule_based_simplify
        simplify_pattern = r'simplified_text = self\._rule_based_simplify\(\s*text,\s*level,\s*language\s*\)'
        if re.search(simplify_pattern, new_content):
            new_content = re.sub(
                simplify_pattern, 
                'simplified_text = self._rule_based_simplify(text, level, language, domain=options.get("domain"))', 
                new_content
            )
        
        # Write the modified content
        with open(simplifier_path, 'w') as f:
            f.write(new_content)
        
        print("Successfully enhanced rule-based simplification!")
        return True
    else:
        print("Could not find _rule_based_simplify method in the file")
        return False

if __name__ == "__main__":
    apply_enhanced_simplifier()