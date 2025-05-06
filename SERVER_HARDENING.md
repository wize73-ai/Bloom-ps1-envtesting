# CasaLingua Server Hardening Recommendations

## Overview

This document outlines recommendations for hardening the CasaLingua server to address observed issues with model loading and stability.

## Critical Issues

1. **Model Loading Cascade Failures**
   - Problem: When certain API calls trigger model loading, multiple models attempt to load simultaneously, causing resource contention and failures
   - Impact: Server becomes unstable or unresponsive, leading to cascading model reload attempts

2. **Insufficient Model Loading Control**
   - Problem: No mechanism to ensure models load completely before accepting requests
   - Impact: Premature requests can interrupt loading or cause incomplete model initialization

3. **Inadequate Memory Management**
   - Problem: Large models (especially MBART) can cause memory pressure when loaded alongside other models
   - Impact: Memory pressure leads to poor performance or OOM (out of memory) errors

## Recommended Hardening Measures

### 1. Implement Model Loading Queue

```python
class ModelLoadingQueue:
    """Manages sequential model loading to prevent resource contention"""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.currently_loading = None
        self._worker_task = None
    
    async def start_worker(self):
        """Start the worker that processes the queue"""
        self._worker_task = asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Process model loading requests sequentially"""
        while True:
            model_id, model_config, callback = await self.queue.get()
            self.currently_loading = model_id
            logger.info(f"Starting to load model: {model_id}")
            
            try:
                result = await self._load_model(model_id, model_config)
                if callback:
                    await callback(model_id, result, None)
            except Exception as e:
                logger.error(f"Error loading model {model_id}: {str(e)}")
                if callback:
                    await callback(model_id, None, e)
            finally:
                self.currently_loading = None
                self.queue.task_done()
    
    async def queue_model_loading(self, model_id, model_config, callback=None):
        """Add a model to the loading queue"""
        await self.queue.put((model_id, model_config, callback))
        queue_size = self.queue.qsize()
        logger.info(f"Model {model_id} queued for loading. Queue size: {queue_size}")
        return queue_size
```

### 2. Add Model Readiness Checks and Request Throttling

```python
class EnhancedModelManager:
    # Add these methods to the existing manager
    
    def is_model_ready(self, model_id):
        """Check if a model is fully loaded and ready"""
        if model_id not in self.loaded_models:
            return False
            
        model_data = self.loaded_models[model_id]
        return (model_data.get("status") == "loaded" and
                model_data.get("model") is not None and
                model_data.get("loading_complete", False))
    
    async def ensure_model_ready(self, model_id, timeout=60):
        """Wait until a model is ready, with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_model_ready(model_id):
                return True
            await asyncio.sleep(0.5)
        return False
    
    def get_model_load_status(self):
        """Get loading status of all models"""
        return {
            model_id: {
                "status": data.get("status", "unknown"),
                "loading_progress": data.get("loading_progress", 0),
                "loading_started": data.get("loading_started"),
                "loading_complete": data.get("loading_complete", False)
            }
            for model_id, data in self.loaded_models.items()
        }
```

### 3. Implement Memory Usage Monitoring and Protection

```python
def check_memory_availability(required_mb=0):
    """Check if there's enough memory available"""
    try:
        import psutil
        vm = psutil.virtual_memory()
        available_mb = vm.available / (1024 * 1024)
        
        # Also check GPU memory if available
        gpu_available_mb = 0
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gpu_available_mb = (torch.cuda.get_device_properties(device).total_memory - 
                              torch.cuda.memory_allocated(device)) / (1024 * 1024)
            
        # Use minimum of CPU and GPU memory if both are used
        if gpu_available_mb > 0:
            available_mb = min(available_mb, gpu_available_mb)
            
        logger.info(f"Memory check: {available_mb:.0f}MB available, {required_mb}MB required")
        return available_mb > required_mb, available_mb
    except Exception as e:
        logger.warning(f"Error checking memory: {str(e)}")
        return True, None  # Assume sufficient if we can't check
```

### 4. Add API Circuit Breakers

```python
class CircuitBreaker:
    """Implements circuit breaker pattern for model API calls"""
    
    def __init__(self, name, failure_threshold=5, reset_timeout=300):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        
    def can_execute(self):
        """Check if operation can be executed"""
        current_time = time.time()
        
        # Reset after timeout
        if self.state == "open" and current_time - self.last_failure_time > self.reset_timeout:
            logger.info(f"Circuit breaker {self.name} reset to half-open after timeout")
            self.state = "half-open"
            
        return self.state != "open"
        
    def record_success(self):
        """Record successful execution"""
        if self.state == "half-open":
            logger.info(f"Circuit breaker {self.name} closing after successful execution")
            self.state = "closed"
            self.failure_count = 0
            
    def record_failure(self):
        """Record failed execution"""
        current_time = time.time()
        self.last_failure_time = current_time
        self.failure_count += 1
        
        if self.state == "closed" and self.failure_count >= self.failure_threshold:
            logger.warning(f"Circuit breaker {self.name} opening after {self.failure_count} failures")
            self.state = "open"
        elif self.state == "half-open":
            logger.warning(f"Circuit breaker {self.name} re-opening after failure in half-open state")
            self.state = "open"
```

### 5. Lazy Loading Configuration

Add a configuration option to enable lazy loading of models:

```json
{
    "model_loading": {
        "strategy": "lazy",   // Options: "eager", "lazy", "on_demand"
        "preload_models": ["language_detection"],  // Models to always load
        "load_timeout": 120,  // Seconds to wait for a model to load
        "queue_enabled": true,  // Use loading queue
        "retry": {
            "max_attempts": 3,
            "backoff_factor": 2
        },
        "memory_protection": {
            "enabled": true,
            "minimum_required": 2048  // Minimum MB required
        }
    }
}
```

### 6. Health Check Enhancement

Update the health check endpoint to include model readiness:

```python
@router.get("/health/detailed", response_model=Dict[str, Any])
async def get_detailed_health(request: Request):
    """Get detailed health information"""
    app = request.app
    model_manager = app.state.model_manager if hasattr(app.state, "model_manager") else None
    
    health_data = {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0,
        "models": [],
        "system": {}
    }
    
    # Add model status
    if model_manager:
        model_status = model_manager.get_model_load_status()
        for model_id, status in model_status.items():
            health_data["models"].append({
                "name": model_id,
                "status": status["status"],
                "loading_progress": status["loading_progress"],
                "ready": model_manager.is_model_ready(model_id)
            })
            
        # Check for loading queue
        if hasattr(model_manager, "loading_queue"):
            health_data["model_queue"] = {
                "size": model_manager.loading_queue.queue.qsize(),
                "currently_loading": model_manager.loading_queue.currently_loading
            }
    
    # Add system info
    try:
        import psutil
        vm = psutil.virtual_memory()
        health_data["system"] = {
            "memory_total": f"{vm.total / (1024**3):.1f} GB",
            "memory_available": f"{vm.available / (1024**3):.1f} GB",
            "memory_used": f"{vm.used / (1024**3):.1f} GB",
            "memory_percent": vm.percent,
            "cpu_usage": psutil.cpu_percent()
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            health_data["system"]["gpu"] = {
                "count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": f"{torch.cuda.memory_allocated() / (1024**3):.1f} GB",
                "memory_reserved": f"{torch.cuda.memory_reserved() / (1024**3):.1f} GB"
            }
    except Exception as e:
        health_data["system"] = {"error": str(e)}
    
    return health_data
```

### 7. Model Unloading Strategy

Implement strategic model unloading for memory management:

```python
async def unload_unused_models(self, threshold_minutes=30):
    """Unload models that haven't been used recently"""
    current_time = time.time()
    candidates_for_unloading = []
    
    # Find unused models
    for model_id, model_data in self.loaded_models.items():
        last_used = model_data.get("last_used_time", 0)
        if current_time - last_used > threshold_minutes * 60:
            # Don't unload certain critical models
            if model_id not in self.config.get("always_loaded_models", []):
                candidates_for_unloading.append((model_id, current_time - last_used))
    
    # Sort by least recently used
    candidates_for_unloading.sort(key=lambda x: x[1], reverse=True)
    
    # Unload up to max_unload models
    max_unload = min(2, len(candidates_for_unloading))
    for i in range(max_unload):
        model_id = candidates_for_unloading[i][0]
        await self.unload_model(model_id)
        logger.info(f"Unloaded unused model: {model_id}")
```

## Implementation Plan

1. **Short-term Fixes (Immediate)**
   - Implement the model loading queue to prevent simultaneous loading
   - Add basic memory protection checks before model loading
   - Update health endpoint with model readiness information

2. **Medium-term Improvements (Next Release)**
   - Add circuit breakers for API endpoints
   - Implement the full lazy loading configuration
   - Add timeout handling for loading operations

3. **Long-term Stability (Future Releases)**
   - Implement automatic model unloading strategy
   - Add memory usage monitoring dashboard
   - Implement adaptive request throttling based on server load

## Monitoring Recommendations

1. Add specific logging for model operations:
   ```python
   # Add to logging.py
   model_logger = logging.getLogger("casalingua.models")
   handler = logging.FileHandler("logs/models.log")
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   handler.setFormatter(formatter)
   model_logger.addHandler(handler)
   model_logger.setLevel(logging.DEBUG)
   ```

2. Implement regular memory snapshots to track usage patterns:
   ```python
   # Add scheduled task
   @app.on_event("startup")
   async def start_memory_monitoring():
       asyncio.create_task(monitor_memory())
       
   async def monitor_memory():
       while True:
           # Get memory usage
           vm = psutil.virtual_memory()
           allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
           
           # Log to metrics file
           metrics = {
               "timestamp": time.time(),
               "memory": {
                   "system_available": vm.available / (1024**3),
                   "gpu_allocated": allocated / (1024**3) if allocated else 0
               }
           }
           
           with open(f"logs/metrics/memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
               json.dump(metrics, f)
               
           await asyncio.sleep(300)  # Check every 5 minutes
   ```

These recommendations should help stabilize the CasaLingua server and prevent the model loading cascades that trigger server instability.