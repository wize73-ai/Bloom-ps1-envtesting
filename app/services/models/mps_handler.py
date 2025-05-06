"""
Helper functions for handling MPS (Metal Performance Shaders) device on Apple Silicon
"""

import logging
import torch

logger = logging.getLogger(__name__)

def init_mps_device():
    """Initialize MPS device with dummy tensors to prevent cache errors"""
    if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
        logger.warning("MPS is not available, cannot initialize")
        return False
    
    try:
        logger.info("Initializing MPS device with dummy tensors")
        # Create some dummy tensors and perform operations
        # This ensures the MPS cache is properly initialized
        dummy1 = torch.zeros(1, 1).to("mps")
        dummy2 = torch.ones(1, 1).to("mps")
        result = dummy1 + dummy2
        # Clean up
        del dummy1, dummy2, result
        logger.info("MPS device successfully initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize MPS device: {str(e)}")
        return False

def move_to_mps(tensor):
    """Safely move a tensor to MPS device"""
    if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
        logger.warning("MPS is not available, using tensor as-is")
        return tensor
    
    try:
        # First make sure MPS is initialized
        if tensor.device.type != "mps":
            init_mps_device()
        
        # Move tensor to MPS
        return tensor.to("mps")
    except Exception as e:
        logger.warning(f"Failed to move tensor to MPS: {str(e)}")
        return tensor

def move_inputs_to_device(inputs, device):
    """Move all tensors in inputs dictionary to the specified device"""
    if not inputs:
        return inputs
    
    # First handle MPS specifically
    if device == "mps":
        try:
            init_mps_device()
        except Exception as e:
            logger.warning(f"Failed to initialize MPS before moving inputs: {str(e)}")
    
    # Move each tensor
    try:
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor) and hasattr(inputs[key], "to"):
                inputs[key] = inputs[key].to(device)
    except Exception as e:
        logger.warning(f"Error moving inputs to {device}: {str(e)}")
    
    return inputs