"""
Direct test script for veracity auditor functionality.

This script bypasses the API and directly tests the VeracityAuditor class.
"""

import asyncio
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

async def main():
    from app.audit.veracity import VeracityAuditor
    
    print("Creating VeracityAuditor instance...")
    auditor = VeracityAuditor()
    await auditor.initialize()
    
    # Test cases for translation verification
    test_translations = [
        {
            "source": "I have 5 apples and 3 oranges.",
            "target": "Tengo 5 manzanas y 3 naranjas.",
            "source_lang": "en",
            "target_lang": "es",
            "expected_issues": []
        },
        {
            "source": "I have 5 apples and 3 oranges.",
            "target": "Tengo manzanas y naranjas.",  # Missing numbers
            "source_lang": "en",
            "target_lang": "es",
            "expected_issues": ["missing_numbers"]
        },
        {
            "source": "The red house costs $500,000.",
            "target": "La casa roja cuesta $.",  # Missing numbers
            "source_lang": "en",
            "target_lang": "es",
            "expected_issues": ["missing_numbers"]
        },
        {
            "source": "How many people attended the conference?",
            "target": "How many people attended the conference?",  # Untranslated
            "source_lang": "en", 
            "target_lang": "es",
            "expected_issues": ["untranslated"]
        }
    ]
    
    print("\nRunning translation verification tests...")
    for i, test in enumerate(test_translations):
        print(f"\nTest {i+1}: {test['source']} -> {test['target']}")
        
        result = await auditor.verify_translation(
            test["source"],
            test["target"],
            test["source_lang"],
            test["target_lang"]
        )
        
        print(f"  Verified: {result.get('verified', False)}")
        print(f"  Score: {result.get('score', 0.0):.2f}")
        print(f"  Confidence: {result.get('confidence', 0.0):.2f}")
        
        if result.get("issues"):
            issue_types = [issue["type"] for issue in result["issues"]]
            print(f"  Issues: {', '.join(issue_types)}")
            
            # Check if expected issues were found
            found_expected = all(issue in issue_types for issue in test["expected_issues"])
            print(f"  Found expected issues: {found_expected}")
        else:
            print("  No issues found")
            if test["expected_issues"]:
                print(f"  FAILED: Expected to find issues: {', '.join(test['expected_issues'])}")
    
    # Test the check() method which routes to appropriate verification
    print("\nTesting the general check() method...")
    
    # Translation check
    translation_options = {
        "source_language": "en",
        "target_language": "es",
        "operation": "translation"
    }
    
    trans_result = await auditor.check(
        "I have 5 apples and 3 oranges.",
        "Tengo 5 manzanas y 3 naranjas.",
        translation_options
    )
    
    print("\nGeneral check for translation:")
    print(f"  Verified: {trans_result.get('verified', False)}")
    print(f"  Score: {trans_result.get('score', 0.0):.2f}")
    
    # Simplification check
    simplify_options = {
        "source_language": "en",
        "operation": "simplification",
        "level": 3
    }
    
    complex_text = "The implementation of the proposed algorithm demonstrated a significant reduction in computational complexity while maintaining the accuracy of the results."
    simple_text = "The new algorithm reduced computing complexity while keeping accurate results."
    
    simplify_result = await auditor.check(
        complex_text,
        simple_text,
        simplify_options
    )
    
    print("\nGeneral check for simplification:")
    print(f"  Verified: {simplify_result.get('verified', False)}")
    print(f"  Score: {simplify_result.get('score', 0.0):.2f}")
    
    # Get overall quality statistics
    print("\nQuality statistics:")
    stats = auditor.get_quality_statistics()
    print(f"  Total checks: {stats['overall']['total_count']}")
    print(f"  Verified rate: {stats['overall']['verification_rate']:.2f}")
    
    if stats['overall']['top_issues']:
        top_issues = ", ".join([f"{i['type']} ({i['count']})" for i in stats['overall']['top_issues']])
        print(f"  Top issues: {top_issues}")
    
if __name__ == "__main__":
    asyncio.run(main())