#!/usr/bin/env python3
"""
Fix for integrating rule-based simplification with the API endpoint.

This script modifies the simplification endpoint to properly use our enhanced
rule-based simplification functionality when handling API requests.
"""

import os
import sys
import shutil
import re

def apply_fix():
    """Apply the fix to integrate rule-based simplification with the API endpoint."""
    print("Applying rule-based simplification integration fix...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    pipeline_py_path = os.path.join(script_dir, "app/api/routes/pipeline.py")
    pipeline_py_backup = os.path.join(script_dir, "app/api/routes/pipeline.py.bak")
    
    # Create backup
    if os.path.exists(pipeline_py_path):
        shutil.copy2(pipeline_py_path, pipeline_py_backup)
        print(f"Created backup: {pipeline_py_backup}")
    else:
        print(f"Error: {pipeline_py_path} does not exist")
        return False
    
    # Read the pipeline.py file
    with open(pipeline_py_path, 'r') as f:
        content = f.read()
    
    # Find the simplify_text function
    simplify_pattern = r'async def simplify_text\(.*?\):\s*""".*?""".*?return\s+response'
    match = re.search(simplify_pattern, content, re.DOTALL)
    
    if not match:
        print("Error: Could not find simplify_text function in pipeline.py")
        return False
    
    # Get the function content
    func_content = match.group(0)
    
    # Check if the function already has the enhanced rule-based logic
    if "rule_based_result" in func_content:
        print("Enhanced rule-based logic already present in simplify_text function")
        return True
    
    # Find where to insert the rule-based simplification code
    result_pattern = r'(\s+# Create result model\n\s+result = SimplifyResult\(\n.*?\s+\))'
    result_match = re.search(result_pattern, func_content, re.DOTALL)
    
    if not result_match:
        print("Error: Could not find result creation in simplify_text function")
        return False
    
    # Extract the result creation code
    result_creation = result_match.group(1)
    
    # Create insertion point before the result creation
    insertion_point = func_content.find(result_creation)
    
    # Create the enhanced rule-based simplification check
    enhanced_check = (
        '\n        # Check if simplified text is unchanged from original\n'
        '        if simplify_result.get("simplified_text") == simplify_request.text:\n'
        '            # Fall back to enhanced rule-based simplification\n'
        '            try:\n'
        '                # Parse the target level\n'
        '                level = 3  # Default to middle level\n'
        '                if simplify_request.target_level.isdigit():\n'
        '                    level = int(simplify_request.target_level)\n'
        '                    level = max(1, min(5, level))  # Ensure level is between 1-5\n'
        '                elif simplify_request.target_level.lower() == "simple":\n'
        '                    level = 4\n'
        '                elif simplify_request.target_level.lower() == "medium":\n'
        '                    level = 3\n'
        '                elif simplify_request.target_level.lower() == "complex":\n'
        '                    level = 2\n'
        '                \n'
        '                # Import the rule-based simplification function\n'
        '                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))\n'
        '                from app.core.pipeline.simplifier import SimplificationPipeline\n'
        '                \n'
        '                # Determine if legal domain based on options\n'
        '                is_legal_domain = False\n'
        '                if isinstance(options, dict) and options.get("domain"):\n'
        '                    is_legal_domain = "legal" in options["domain"].lower()\n'
        '                \n'
        '                # Create a temporary simplification pipeline\n'
        '                simplifier = SimplificationPipeline(processor.model_manager)\n'
        '                \n'
        '                # Apply rule-based simplification\n'
        '                rule_based_text = simplifier._rule_based_simplify(\n'
        '                    simplify_request.text, level, is_legal_domain=is_legal_domain\n'
        '                )\n'
        '                \n'
        '                # Update the result with rule-based simplified text\n'
        '                if rule_based_text and rule_based_text != simplify_request.text:\n'
        '                    simplify_result["simplified_text"] = rule_based_text\n'
        '                    simplify_result["model_used"] = f"rule_based_simplifier_level_{level}"\n'
        '                    logger.info(f"Applied rule-based simplification at level {level}")\n'
        '            except Exception as e:\n'
        '                logger.error(f"Error applying rule-based simplification: {str(e)}", exc_info=True)\n'
        '                # Continue with original result if rule-based fails\n'
    )
    
    # Create the modified function with enhanced rule-based simplification
    modified_func = func_content[:insertion_point] + enhanced_check + func_content[insertion_point:]
    
    # Replace the function in the content
    modified_content = content.replace(func_content, modified_func)
    
    # Write back the modified file
    with open(pipeline_py_path, 'w') as f:
        f.write(modified_content)
    
    print("Successfully applied rule-based simplification integration fix!")
    print("Please restart the server for changes to take effect.")
    return True

if __name__ == "__main__":
    apply_fix()