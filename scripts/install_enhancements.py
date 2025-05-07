#!/usr/bin/env python3
"""
Install Enhancements Script

This script installs the enhanced model prompting system and
fixes for metrics and audit reporting in the running server.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import requests
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def main():
    """Main function to install enhancements."""
    logger.info("Starting enhancement installation")
    
    # Check server status
    server_url = "http://localhost:8000"
    try:
        response = requests.get(f"{server_url}/health")
        if response.status_code != 200:
            logger.error(f"Server not responding correctly: {response.status_code}")
            sys.exit(1)
        logger.info("Server is running and responding")
    except Exception as e:
        logger.error(f"Error connecting to server: {str(e)}")
        sys.exit(1)
    
    # Install metrics fixes
    try:
        from app.audit.metrics_fix import setup_enhanced_metrics
        result = await setup_enhanced_metrics()
        logger.info(f"Enhanced metrics setup: {'Success' if result else 'Failed'}")
    except Exception as e:
        logger.error(f"Error setting up enhanced metrics: {str(e)}")
    
    # Test language detection
    try:
        logger.info("Testing language detection endpoint")
        response = requests.post(
            f"{server_url}/pipeline/detect",
            json={"text": "This is a test of the enhanced language detection system.", "detailed": True}
        )
        if response.status_code == 200:
            logger.info("Language detection test successful")
            result = response.json()
            if "data" in result:
                logger.info(f"Detected language: {result['data'].get('detected_language')}, "
                          f"Confidence: {result['data'].get('confidence')}")
        else:
            logger.error(f"Language detection test failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Error testing language detection: {str(e)}")
    
    # Test simplification
    try:
        logger.info("Testing simplification endpoint")
        response = requests.post(
            f"{server_url}/pipeline/simplify",
            json={
                "text": "The implementation of the fiscal policies necessitated extensive deliberation concerning potential macroeconomic ramifications.",
                "language": "en",
                "level": 4,
                "domain": "financial"
            }
        )
        if response.status_code == 200:
            logger.info("Simplification test successful")
            result = response.json()
            if "data" in result:
                logger.info(f"Simplified text: {result['data'].get('simplified_text')}")
        else:
            logger.error(f"Simplification test failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Error testing simplification: {str(e)}")
    
    logger.info("Enhancement installation complete")
    logger.info("Note: To fully apply enhancements, the server should be restarted")

if __name__ == "__main__":
    asyncio.run(main())