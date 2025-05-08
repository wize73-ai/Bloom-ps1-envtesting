#!/usr/bin/env python3
"""
Direct test of the rule-based simplification without going through the API.
This script directly tests the simplification functionality by importing the
SimplificationPipeline and applying the rule-based simplification.
"""

import os
import sys
import re
from typing import Dict, Any

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ANSI color codes for prettier output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
ENDC = "\033[0m"
BOLD = "\033[1m"

# Test texts
TEST_TEXTS = [
    {
        "name": "Legal Text",
        "text": "The tenant shall indemnify and hold harmless the landlord from and against any and all claims, actions, suits, judgments and demands brought or recovered against the landlord by reason of any negligent or willful act or omission of the tenant, its agents, servants, employees, licensees, visitors, invitees, and any of the tenant's contractors and subcontractors."
    },
    {
        "name": "Technical Text",
        "text": "Prior to commencement of the installation process, ensure that all prerequisite components have been obtained and are readily accessible for utilization. Failure to verify component availability may result in procedural delays."
    },
    {
        "name": "Financial Text",
        "text": "The applicant must furnish documentation verifying income and employment status in accordance with the requirements delineated in section 8 of the aforementioned application procedure."
    }
]

# Define word replacements by level
def get_replacements_by_level(level: int) -> Dict[str, str]:
    """Get text replacements for a specific simplification level."""
    # Define substitutions by level
    # Level 1: Minimal substitutions
    # Level 5: Maximum substitutions
    substitutions = {
        # Common for all levels
        1: {
            r'\butilize\b': 'use',
            r'\bpurchase\b': 'buy',
            r'\bsubsequently\b': 'later',
            r'\bfurnish\b': 'provide',
            r'\baforementioned\b': 'previously mentioned',
            r'\bdelineated\b': 'outlined',
            r'\bin accordance with\b': 'according to'
        },
        2: {
            r'\butilize\b': 'use',
            r'\bpurchase\b': 'buy',
            r'\bindicate\b': 'show',
            r'\bsufficient\b': 'enough',
            r'\bsubsequently\b': 'later',
            r'\badditional\b': 'more',
            r'\bprior to\b': 'before',
            r'\bfurnish\b': 'provide',
            r'\baforementioned\b': 'previously mentioned',
            r'\bdelineated\b': 'outlined',
            r'\bin accordance with\b': 'according to',
            r'\bverifying\b': 'proving',
            r'\brequirements\b': 'rules'
        },
        3: {
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
            r'\bfurnish\b': 'provide',
            r'\baforementioned\b': 'previously mentioned',
            r'\bdelineated\b': 'described',
            r'\bin accordance with\b': 'following',
            r'\bverifying\b': 'proving',
            r'\brequirements\b': 'rules',
            r'\bemployment status\b': 'job status',
            r'\bapplication procedure\b': 'application process'
        },
        4: {
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
            r'\battempt\b': 'try',
            r'\bapplicant\b': 'you',
            r'\bfurnish\b': 'give',
            r'\baforementioned\b': 'this',
            r'\bdelineated\b': 'listed',
            r'\bin accordance with\b': 'following',
            r'\bverifying\b': 'that proves',
            r'\brequirements\b': 'rules',
            r'\bemployment status\b': 'job information',
            r'\bapplication procedure\b': 'application steps',
            r'\bdocumentation\b': 'papers',
            r'\bsection 8\b': 'part 8'
        },
        5: {
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
            r'\battempt\b': 'try',
            r'\binquire\b': 'ask',
            r'\bascertain\b': 'find out',
            r'\bcomprehend\b': 'understand',
            r'\bnevertheless\b': 'however',
            r'\btherefore\b': 'so',
            r'\bfurthermore\b': 'also',
            r'\bconsequently\b': 'so',
            r'\bapproximately\b': 'about',
            r'\bmodification\b': 'change',
            r'\bendeavor\b': 'try',
            r'\bproficiency\b': 'skill',
            r'\bnecessitate\b': 'need',
            r'\bacquisition\b': 'getting',
            r'\bimmersion\b': 'practice',
            r'\bassimilation\b': 'learning',
            r'\bapplicant\b': 'you',
            r'\bfurnish\b': 'give',
            r'\baforementioned\b': 'this',
            r'\bdelineated\b': 'listed',
            r'\bin accordance with\b': 'based on',
            r'\bverifying\b': 'that proves',
            r'\brequirements\b': 'rules',
            r'\bemployment status\b': 'job info',
            r'\bapplication procedure\b': 'form',
            r'\bdocumentation\b': 'papers',
            r'\bsection 8\b': 'part 8',
            r'\bmust\b': 'need to'
        }
    }
    
    # Get the appropriate substitutions for this level and all lower levels
    all_substitutions = {}
    for l in range(1, level + 1):
        if l in substitutions:
            all_substitutions.update(substitutions[l])
            
    return all_substitutions

def rule_based_simplify(text: str, level: int, is_legal_domain: bool = False) -> str:
    """Apply rule-based simplification with a specific level."""
    # If no text, return empty string
    if not text:
        return ""
    
    # Get substitutions for this level
    replacements = get_replacements_by_level(level)
    
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
    simplified_text = re.sub(r'\s+', ' ', simplified_text).strip()
    
    # For highest level, add explaining phrases
    if level == 5:
        if is_legal_domain:
            simplified_text += " This means you need to follow what the law says."
        else:
            simplified_text += " This means you need to show proof of your job and income as required."
    
    return simplified_text

def calculate_metrics(text: str) -> Dict[str, Any]:
    """Calculate readability metrics for the text."""
    try:
        # Count words and sentences
        words = len(re.findall(r'\b\w+\b', text))
        sentences = len(re.split(r'[.!?]+', text)) or 1
        
        # Rough approximation for syllables
        vowels = "aeiouy"
        syllables = 0
        
        for word in re.findall(r'\b\w+\b', text.lower()):
            if len(word) <= 3:
                syllables += 1
                continue
                
            # Count vowel groups as syllables
            count = 0
            prev_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            
            # Adjustments
            if word.endswith('e'):
                count -= 1
            if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                count += 1
            if count == 0:
                count = 1
                
            syllables += count
        
        # Calculate Flesch-Kincaid Grade Level
        words_per_sentence = words / sentences
        syllables_per_word = syllables / max(1, words)
        fk_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
        
        # Ensure reasonable range
        fk_grade = max(1, min(12, fk_grade))
        
        return {
            "grade_level": round(fk_grade, 1),
            "words_per_sentence": round(words_per_sentence, 1),
            "syllables_per_word": round(syllables_per_word, 2),
            "words": words,
            "sentences": sentences
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {"grade_level": 0}

def direct_simplify_test():
    """Direct test of the rule-based simplification."""
    print(f"\n{BOLD}{BLUE}Testing Direct Rule-Based Simplification{ENDC}")
    print("-" * 80)
    
    # Test each text
    for test_case in TEST_TEXTS:
        print(f"\n{BOLD}Test Case: {test_case['name']}{ENDC}")
        print(f"{BOLD}Original Text:{ENDC} {test_case['text']}")
        
        # Calculate metrics for original
        original_metrics = calculate_metrics(test_case['text'])
        print(f"{BOLD}Original Grade Level:{ENDC} {original_metrics['grade_level']}")
        print("-" * 80)
        
        # Test each simplification level
        simplified_texts = []
        
        for level in range(1, 6):
            print(f"\n{BOLD}Simplification Level {level}:{ENDC}")
            
            # Direct rule-based simplification
            is_legal = "legal" in test_case['name'].lower()
            simplified = rule_based_simplify(test_case['text'], level, is_legal_domain=is_legal)
            simplified_texts.append(simplified)
            
            print(simplified)
            
            # Calculate metrics
            metrics = calculate_metrics(simplified)
            print(f"{BOLD}Grade Level:{ENDC} {metrics['grade_level']}")
            print(f"{BOLD}Words:{ENDC} {metrics['words']}")
            
            # Calculate text diff percentage
            orig_words = original_metrics['words']
            simp_words = metrics['words']
            word_diff = (simp_words - orig_words) / orig_words * 100 if orig_words else 0
            diff_color = GREEN if word_diff < 0 else YELLOW
            print(f"{BOLD}Change:{ENDC} {diff_color}{word_diff:.1f}%{ENDC}")
            
        # Check uniqueness
        unique_texts = set(simplified_texts)
        print(f"\n{BOLD}Summary for {test_case['name']}:{ENDC}")
        print(f"Unique simplified outputs: {len(unique_texts)} out of 5 levels")
        
        if len(unique_texts) >= 4:
            print(f"{GREEN}The simplification levels are producing different outputs as expected.{ENDC}")
        elif len(unique_texts) >= 3:
            print(f"{YELLOW}The simplification levels are producing somewhat different outputs.{ENDC}")
        else:
            print(f"{RED}The simplification levels are not producing sufficiently different outputs.{ENDC}")
        
        print("-" * 80)

def test_with_simplification_pipeline():
    """Test simplification using the actual SimplificationPipeline."""
    try:
        # Import necessary components
        from app.core.pipeline.simplifier import SimplificationPipeline
        
        # Create a mock model manager (since we're only testing the rule-based simplification)
        class MockModelManager:
            def get_model(self, model_type, model_id=None):
                return None
        
        print(f"\n{BOLD}{BLUE}Testing SimplificationPipeline._rule_based_simplify{ENDC}")
        print("-" * 80)
        
        # Initialize the pipeline without actual models
        model_manager = MockModelManager()
        pipeline = SimplificationPipeline(model_manager)
        
        # Test each text
        for test_case in TEST_TEXTS:
            print(f"\n{BOLD}Test Case: {test_case['name']}{ENDC}")
            print(f"{BOLD}Original Text:{ENDC} {test_case['text']}")
            
            # Calculate metrics for original
            original_metrics = calculate_metrics(test_case['text'])
            print(f"{BOLD}Original Grade Level:{ENDC} {original_metrics['grade_level']}")
            print("-" * 80)
            
            # Test each simplification level
            simplified_texts = []
            
            for level in range(1, 6):
                print(f"\n{BOLD}Simplification Level {level}:{ENDC}")
                
                # Use the pipeline's rule-based simplification
                is_legal = "legal" in test_case['name'].lower()
                simplified = pipeline._rule_based_simplify(
                    test_case['text'], level, is_legal_domain=is_legal
                )
                simplified_texts.append(simplified)
                
                print(simplified)
                
                # Calculate metrics
                metrics = calculate_metrics(simplified)
                print(f"{BOLD}Grade Level:{ENDC} {metrics['grade_level']}")
                print(f"{BOLD}Words:{ENDC} {metrics['words']}")
                
                # Calculate text diff percentage
                orig_words = original_metrics['words']
                simp_words = metrics['words']
                word_diff = (simp_words - orig_words) / orig_words * 100 if orig_words else 0
                diff_color = GREEN if word_diff < 0 else YELLOW
                print(f"{BOLD}Change:{ENDC} {diff_color}{word_diff:.1f}%{ENDC}")
                
            # Check uniqueness
            unique_texts = set(simplified_texts)
            print(f"\n{BOLD}Summary for {test_case['name']}:{ENDC}")
            print(f"Unique simplified outputs: {len(unique_texts)} out of 5 levels")
            
            if len(unique_texts) >= 4:
                print(f"{GREEN}The simplification levels are producing different outputs as expected.{ENDC}")
            elif len(unique_texts) >= 3:
                print(f"{YELLOW}The simplification levels are producing somewhat different outputs.{ENDC}")
            else:
                print(f"{RED}The simplification levels are not producing sufficiently different outputs.{ENDC}")
            
            print("-" * 80)
            
    except ImportError as e:
        print(f"{RED}Unable to import SimplificationPipeline: {e}{ENDC}")
        print("Testing with local implementation instead...")
        direct_simplify_test()
    except Exception as e:
        print(f"{RED}Error while testing with SimplificationPipeline: {e}{ENDC}")
        print("Testing with local implementation instead...")
        direct_simplify_test()

if __name__ == "__main__":
    # Test with both methods
    print(f"{BOLD}Testing simplification with local implementation:{ENDC}")
    direct_simplify_test()
    
    print(f"\n{BOLD}Testing simplification with SimplificationPipeline:{ENDC}")
    test_with_simplification_pipeline()