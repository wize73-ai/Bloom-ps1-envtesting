# app/utils/logging.py
"""
Logging Utilities for CasaLingua

This module provides comprehensive logging configuration for the
entire application, supporting console, file, and structured logging.
"""

import os
import sys
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from app.ui.console import setup_console_logging, setup_file_logging
from app.utils.config import get_config_value

class ConfigFilter(logging.Filter):
    """
    Filter to truncate large configuration dumps in log messages.
    """
    def filter(self, record):
        if record.module == "config" and hasattr(record, "msg"):
            # Check if this is a large config dump
            msg = record.getMessage()
            if len(msg) > 500 and ("config" in msg.lower() and "{" in msg and "}" in msg):
                # Truncate the message
                record.msg = f"{msg.split('{', 1)[0]}... (content truncated)"
        return True


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    This formatter outputs log records as JSON objects, making them
    easily parseable by log aggregation systems.
    """
    
    def format(self, record):
        """Format the record as a JSON object."""
        # Apply config filter to truncate large config dumps
        config_filter = ConfigFilter()
        config_filter.filter(record)
        
        # Create base log record
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "thread": record.thread,
            "process": record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in log_record:
                continue
            
            # Skip non-serializable objects
            try:
                json.dumps({key: value})
                log_record[key] = value
            except (TypeError, OverflowError, ValueError):
                try:
                    log_record[key] = str(value)
                except Exception:
                    log_record[key] = "<unserializable>"
        
        return json.dumps(log_record)

def configure_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Configure comprehensive logging for the application.
    
    Args:
        config: Application configuration
        
    Returns:
        logging.Logger: The configured root logger
    """
    # Get logging configuration
    log_level_str = get_config_value(config, "log_level", "INFO")
    try:
        log_level = getattr(logging, log_level_str)
    except AttributeError:
        log_level = logging.INFO
    
    log_dir = Path(get_config_value(config, "log_dir", "logs"))
    environment = get_config_value(config, "environment", "development")
    
    # Create log directory
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create log directory {log_dir}: {e}", file=sys.stderr)
    
    # Get or create root logger
    logger = logging.getLogger("casalingua")
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create a config filter instance to be reused
    config_filter = ConfigFilter()
    
    # Always set up console logging
    console_level_str = "DEBUG" if get_config_value(config, "debug", False) else log_level_str
    try:
        console_level = getattr(logging, console_level_str)
    except AttributeError:
        console_level = logging.INFO
    console_logger = setup_console_logging(console_level)
    
    # Apply config filter to all handlers of the console logger
    for handler in console_logger.handlers:
        handler.addFilter(config_filter)
    
    # Set up file logging
    if environment != "development" or get_config_value(config, "log_to_file", False):
        log_file = log_dir / "casalingua.log"
        setup_file_logging(logger, str(log_file), log_level_str)
        
        # Apply config filter to all handlers
        for handler in logger.handlers:
            handler.addFilter(config_filter)
    
    # Set up JSON logging for production
    if environment == "production" or get_config_value(config, "structured_logging", False):
        json_file = log_dir / "casalingua.json"
        json_handler = logging.handlers.RotatingFileHandler(
            str(json_file),
            maxBytes=10485760,  # 10 MB
            backupCount=10
        )
        json_handler.setLevel(log_level)
        json_handler.setFormatter(JSONFormatter())
        # Note: JSONFormatter already applies the ConfigFilter internally
        logger.addHandler(json_handler)
    
    # Set up module-specific loggers
    for module in ["api", "core", "services", "audit", "ui"]:
        module_logger = logging.getLogger(f"casalingua.{module}")
        module_logger.setLevel(log_level)
        module_logger.propagate = True  # Let the root logger handle it
    
    # Apply config filter to the config logger specifically
    config_logger = logging.getLogger("casalingua.config")
    config_logger.addFilter(config_filter)
    
    logger.info(f"Logging configured - Level: {log_level_str}, Environment: {environment}")
    return logger

def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        logging.Logger: Logger for the module
    """
    return logging.getLogger(f"casalingua.{module_name}")