#!/usr/bin/env python3
"""
Apply MBART Spanish-to-English Translation Fix

This script applies the fix for Spanish to English translation in MBART models.
It implements the _get_mbart_lang_code method and improves handling of the
forced_bos_token_id parameter that's essential for correct target language generation.

Author: CasaLingua Development Team
"""

import os
import sys
import logging
import shutil
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_if_fix_already_applied():
    """Check if the fix is already applied by looking for _get_mbart_lang_code method"""
    wrapper_path = os.path.join('app', 'services', 'models', 'wrapper.py')
    
    try:
        with open(wrapper_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check if the method exists
            if '_get_mbart_lang_code' in content:
                logger.info("✅ Fix already applied (_get_mbart_lang_code method found)")
                return True
            else:
                logger.info("❌ Fix not applied (_get_mbart_lang_code method not found)")
                return False
    except Exception as e:
        logger.error(f"Error checking for existing fix: {e}")
        return False

def backup_file(file_path):
    """Create a backup of the file"""
    try:
        if os.path.exists(file_path):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{file_path}.bak_{timestamp}"
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return True
        else:
            logger.error(f"File not found: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return False

def apply_fix():
    """Apply the fix by running the script from fix_mbart_tokenizer.py"""
    try:
        wrapper_path = os.path.join('app', 'services', 'models', 'wrapper.py')
        
        # Create a backup first
        if not backup_file(wrapper_path):
            logger.error("Failed to create backup, aborting fix")
            return False
        
        # Import and run the patch function
        try:
            from scripts.fix_mbart_tokenizer import patch_translation_model_wrapper
            
            # Apply the patch
            success = patch_translation_model_wrapper()
            
            if success:
                logger.info("✅ Successfully applied MBART tokenizer fix")
                return True
            else:
                logger.error("❌ Failed to apply MBART tokenizer fix")
                return False
        except ImportError:
            logger.error("Could not import fix_mbart_tokenizer module")
            
            # If direct import fails, try manual patching
            return apply_manual_fix()
    except Exception as e:
        logger.error(f"Error applying fix: {e}")
        return False

def apply_manual_fix():
    """Apply the fix manually by modifying the wrapper.py file directly"""
    try:
        wrapper_path = os.path.join('app', 'services', 'models', 'wrapper.py')
        
        # Read the existing file
        with open(wrapper_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the TranslationModelWrapper class
        class_start = content.find("class TranslationModelWrapper(BaseModelWrapper):")
        if class_start == -1:
            logger.error("Could not find TranslationModelWrapper class in wrapper.py")
            return False
        
        # Find the first method after the class definition
        first_method = content.find("def ", class_start)
        if first_method == -1:
            logger.error("Could not find methods in TranslationModelWrapper class")
            return False
        
        # Add the _get_mbart_lang_code method right after the class declaration
        mbart_lang_code_method = """
    def _get_mbart_lang_code(self, language_code: str) -> str:
        \"\"\"Convert ISO language code to MBART language code format\"\"\"
        # MBART-50 uses language codes like "en_XX", "es_XX", etc.
        if language_code in ["zh", "zh-cn", "zh-CN"]:
            return "zh_CN"
        elif language_code in ["zh-tw", "zh-TW"]:
            return "zh_TW"
        else:
            # Just the base language code for most languages
            base_code = language_code.split("-")[0].lower()
            return f"{base_code}_XX"
            """
        
        # Insert the method at the appropriate location
        new_content = content[:first_method] + mbart_lang_code_method + content[first_method:]
        
        # Write the modified content back to the file
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.info("✅ Successfully added _get_mbart_lang_code method")
        
        # Now verify that the method was added
        if check_if_fix_already_applied():
            return True
        else:
            logger.error("Failed to verify the fix application")
            return False
    except Exception as e:
        logger.error(f"Error applying manual fix: {e}")
        return False

def test_fix():
    """Test the fixed implementation"""
    try:
        logger.info("Testing Spanish to English translation with fixed implementation...")
        
        # Run the test script
        test_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_mbart_wrapper_direct.py')
        if os.path.exists(test_script_path):
            # Execute the test script
            logger.info("Running test script...")
            import subprocess
            result = subprocess.run([sys.executable, test_script_path], capture_output=True, text=True)
            
            # Check if the test was successful
            if "SUCCESS: Translation appears to be in English" in result.stdout:
                logger.info("✅ Test completed successfully: Spanish to English translation now works correctly")
                return True
            else:
                logger.error(f"❌ Test failed:\n{result.stdout}\n{result.stderr}")
                return False
        else:
            logger.error(f"Test script not found: {test_script_path}")
            return False
    except Exception as e:
        logger.error(f"Error testing fix: {e}")
        return False

def main():
    """Main function that applies the MBART fix"""
    logger.info("Starting MBART Spanish-to-English translation fix")
    
    # Check if fix is already applied
    if check_if_fix_already_applied():
        logger.info("Fix already applied, no action needed")
    else:
        # Apply the fix
        if apply_fix():
            # Test the fix
            if test_fix():
                logger.info("✅ MBART Spanish-to-English translation fix successfully applied and tested")
            else:
                logger.warning("⚠️ MBART fix applied but test failed - manual intervention may be needed")
        else:
            logger.error("❌ Failed to apply MBART fix")

if __name__ == "__main__":
    main()