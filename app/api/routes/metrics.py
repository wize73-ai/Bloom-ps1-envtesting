"""
Memory metrics endpoint for CasaLingua API
Provides detailed memory usage by models and system
"""

import os
import gc
import json
import time
import psutil
import logging
import torch
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from app.utils.logging import get_logger
from app.api.schemas.system import MetricsResponse

# Initialize router
router = APIRouter(prefix="/health", tags=["health"])
logger = get_logger("casalingua.api.metrics")

# @router.get("/metrics", response_model=Dict[str, Any])
# async def get_metrics(request: Request) -> Dict[str, Any]:
#     """
#     Get detailed memory metrics about the running server and loaded models
#     
#     Returns:
#         Dict with memory usage metrics
#     """
#     metrics = {
#         "timestamp": time.time(),
#         "system_memory": {},
#         "model_memory": {},
#         "process_memory": None,
#     }
#     
#     # Get system memory
#     try:
#         vm = psutil.virtual_memory()
#         metrics["system_memory"] = {
#             "total": round(vm.total / (1024**3), 2),  # GB
#             "available": round(vm.available / (1024**3), 2),  # GB
#             "used": round(vm.used / (1024**3), 2),  # GB
#             "percent": vm.percent
#         }
#     except Exception as e:
#         logger.warning(f"Failed to get system memory: {str(e)}")
#     
#     # Get process memory
#     try:
#         process = psutil.Process(os.getpid())
#         metrics["process_memory"] = round(process.memory_info().rss / (1024**3), 2)  # GB
#     except Exception as e:
#         logger.warning(f"Failed to get process memory: {str(e)}")
#     
#     # Get GPU memory if available
#     if torch.cuda.is_available():
#         try:
#             cuda_device = torch.cuda.current_device()
#             metrics["gpu_memory"] = {
#                 "total": round(torch.cuda.get_device_properties(cuda_device).total_memory / (1024**3), 2),  # GB
#                 "allocated": round(torch.cuda.memory_allocated(cuda_device) / (1024**3), 2),  # GB
#                 "cached": round(torch.cuda.memory_reserved(cuda_device) / (1024**3), 2),  # GB
#                 "available": round((torch.cuda.get_device_properties(cuda_device).total_memory - 
#                                   torch.cuda.memory_allocated(cuda_device)) / (1024**3), 2)  # GB
#             }
#             metrics["gpu_memory"]["used"] = round(metrics["gpu_memory"]["allocated"], 2)
#         except Exception as e:
#             logger.warning(f"Failed to get GPU memory: {str(e)}")
#     
#     # Get model memory from app state if available
#     try:
#         app_state = request.app.state
#         if hasattr(app_state, "model_manager"):
#             model_manager = app_state.model_manager
#             if hasattr(model_manager, "loaded_models"):
#                 loaded_models = model_manager.loaded_models
#                 for model_id, model_data in loaded_models.items():
#                     # Estimate model memory usage if possible
#                     model_obj = model_data.get("model")
#                     if model_obj:
#                         try:
#                             # Different methods to estimate memory
#                             if hasattr(model_obj, "get_memory_footprint"):
#                                 # Some models have this method
#                                 mem_bytes = model_obj.get_memory_footprint()
#                                 metrics["model_memory"][model_id] = round(mem_bytes / (1024**3), 2)  # GB
#                             elif hasattr(torch, "hf_device_map"):
#                                 # For models with device map
#                                 metrics["model_memory"][model_id] = "Distributed across devices"
#                             else:
#                                 # Use PyTorch's memory_allocated before and after collection
#                                 torch.cuda.empty_cache()
#                                 gc.collect()
#                                 before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
#                                 
#                                 # Force Python garbage collection
#                                 gc.collect()
#                                 
#                                 # For non-CUDA models or as fallback
#                                 metrics["model_memory"][model_id] = "Active"
#                         except Exception as e:
#                             metrics["model_memory"][model_id] = "Unknown"
#                             logger.warning(f"Failed to estimate memory for model {model_id}: {str(e)}")
#     except Exception as e:
#         logger.warning(f"Failed to get model memory: {str(e)}")
#     
#     return metrics