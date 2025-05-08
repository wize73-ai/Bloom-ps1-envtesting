"""
Enhanced Model Manager for CasaLingua
Provides advanced model management with hardware-aware loading
"""

import os
import logging
import asyncio
import torch
import gc
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# Configure logging
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

# Import model loader
from app.services.models.loader import ModelLoader

class ModelType(str, Enum):
    """Model types enum"""
    TRANSLATION = "translation"
    MULTIPURPOSE = "multipurpose" 
    VERIFICATION = "verification"
    LANGUAGE_DETECTION = "language_detection"
    NER_DETECTION = "ner_detection"
    SIMPLIFIER = "simplifier"
    RAG_GENERATOR = "rag_generator"
    RAG_RETRIEVER = "rag_retriever"
    ANONYMIZER = "anonymizer"

class EnhancedModelManager:
    """Enhanced model manager with hardware-aware capabilities"""
    
    def __init__(self, loader: ModelLoader, hardware_info: Dict[str, Any], config: Dict[str, Any] = None):
        """
        Initialize enhanced model manager
        
        Args:
            loader (ModelLoader): Model loader instance
            hardware_info (Dict[str, Any]): Hardware information
            config (Dict[str, Any], optional): Application configuration
        """
        self.loader = loader
        self.hardware_info = hardware_info
        self.config = config or {}
        self.loaded_models = {}
        self.model_metadata = {}
        
        # Device and precision configuration
        self.device = self._determine_device()
        self.precision = self._determine_precision()
        
        console.print(Panel(
            f"[bold cyan]Enhanced Model Manager Initialized[/bold cyan]\n"
            f"Device: [yellow]{self.device}[/yellow], Precision: [green]{self.precision}[/green]",
            border_style="blue"
        ))
    
    def _determine_device(self) -> str:
        """
        Determine the appropriate device for model execution
        
        Returns:
            str: Device string ("cuda", "mps", "cpu")
        """
        # Check if device is specified in config
        if self.config.get("device"):
            return self.config["device"]
        
        # Check hardware information
        gpu_info = self.hardware_info.get("gpu", {})
        has_gpu = gpu_info.get("has_gpu", False)
        
        if has_gpu:
            if gpu_info.get("cuda_available", False):
                return "cuda"
            elif gpu_info.get("mps_available", False):
                return "mps"
        
        # Default to CPU
        return "cpu"
    
    def _determine_precision(self) -> str:
        """
        Determine the appropriate precision for models
        
        Returns:
            str: Precision string ("float32", "float16", "int8")
        """
        # Check if precision is specified in config
        if self.config.get("precision"):
            return self.config["precision"]
        
        # Base decision on hardware
        if self.device == "cuda":
            # For CUDA, use float16 if enough memory
            gpu_memory = self.hardware_info.get("gpu", {}).get("gpu_memory", 0)
            if gpu_memory > 4 * 1024**3:  # > 4GB
                return "float16"
            else:
                return "int8"
        elif self.device == "mps":
            # For MPS (Apple Silicon), use float16
            return "float16"
        else:
            # For CPU, use int8 if AVX2 available, otherwise float32
            cpu_info = self.hardware_info.get("cpu", {})
            if cpu_info.get("supports_avx2", False):
                return "int8"
            else:
                return "float32"
    
    async def load_model(self, model_type: Union[ModelType, str]) -> Dict[str, Any]:
        """
        Load a model with the appropriate settings based on hardware
        
        Args:
            model_type (Union[ModelType, str]): Model type
            
        Returns:
            Dict[str, Any]: Model information
        """
        # Convert enum to string if needed
        if isinstance(model_type, ModelType):
            model_type_str = model_type.value
        else:
            model_type_str = model_type
            
        console.print(f"[bold cyan]Enhanced loading of model:[/bold cyan] [yellow]{model_type_str}[/yellow]")
        
        # Check if model is already loaded
        if model_type_str in self.loaded_models:
            console.print(f"[green]✓[/green] Model [yellow]{model_type_str}[/yellow] already loaded from cache")
            return {
                "model": self.loaded_models[model_type_str],
                "tokenizer": self.model_metadata.get(model_type_str, {}).get("tokenizer"),
                "type": model_type_str
            }
        
        # Prepare kwargs for loading
        kwargs = {
            "device": self.device,
            "precision": self.precision
        }
        
        # Load model using loader
        try:
            # Log start of loading instead of using Progress
            logger.info(f"Loading {model_type_str} model with {self.precision} precision...")
            console.print(f"[bold blue]Loading {model_type_str} model with {self.precision} precision...[/bold blue]")
            
            # Load model using async loader
            start_time = time.time()
            model_info = await self.loader.load_model_async(model_type_str, **kwargs)
            load_time = time.time() - start_time
            
            # Extract model and tokenizer
            model = model_info["model"]
            tokenizer = model_info.get("tokenizer")
            config = model_info.get("config")
            
            # Store model and metadata
            self.loaded_models[model_type_str] = model
            self.model_metadata[model_type_str] = {
                "tokenizer": tokenizer,
                "config": config,
                "loaded_at": asyncio.get_event_loop().time()
            }
            
            # Log completion
            logger.info(f"Model {model_type_str} loaded successfully in {load_time:.2f}s")
            console.print(Panel(f"[bold green]✓ Successfully loaded model:[/bold green] [yellow]{model_type_str}[/yellow] in {load_time:.2f}s", border_style="green"))
            
            return {
                "model": model,
                "tokenizer": tokenizer,
                "type": model_type_str
            }
        except Exception as e:
            console.print(Panel(f"[bold red]⚠ Error loading model:[/bold red] [yellow]{model_type_str}[/yellow]\n{str(e)}", border_style="red"))
            logger.error(f"Error loading model {model_type_str}: {e}")
            raise
    
    async def unload_all_models(self, timeout_per_model: float = 30.0) -> None:
        """
        Unload all models to free memory with timeout protection
        
        Args:
            timeout_per_model (float): Maximum time in seconds to wait for each model to unload
        """
        # Log to file rather than console for better reliability during shutdown
        logger.info("Unloading all models...")
        
        # Try console output but catch any errors - terminal might be unstable during shutdown
        try:
            console.print("[bold cyan]Unloading all models...[/bold cyan]")
        except Exception as console_err:
            logger.warning(f"Console error during shutdown: {console_err}")
        
        try:
            # Get list of loaded models
            model_types = list(self.loaded_models.keys())
            
            if not model_types:
                logger.info("No models currently loaded")
                try:
                    console.print("[yellow]No models currently loaded[/yellow]")
                except Exception:
                    pass
                return
            
            # Track failed unloads
            failed_unloads = []
            
            # Check if app is shutting down - use simpler logging if we're in shutdown mode
            is_shutdown_mode = asyncio.current_task().get_name().startswith("shutdown") if hasattr(asyncio.current_task(), "get_name") else False
            
            # Simplified approach without Rich progress bars during shutdown
            if is_shutdown_mode:
                logger.info("Using simplified unloading during shutdown")
                
                # Unload each model without fancy progress tracking
                for model_type in model_types:
                    logger.info(f"Unloading {model_type}...")
                    
                    try:
                        # Create a task to unload the model with timeout
                        unload_task = asyncio.create_task(self._unload_model_with_timeout(model_type, timeout_per_model))
                        
                        # Wait for the task to complete
                        success = await unload_task
                        
                        if not success:
                            logger.warning(f"Model {model_type} unload timed out after {timeout_per_model}s")
                            failed_unloads.append((model_type, "timeout"))
                        else:
                            logger.info(f"Model {model_type} unloaded successfully")
                            
                        # Remove from local cache regardless of unload success
                        if model_type in self.loaded_models:
                            del self.loaded_models[model_type]
                        if model_type in self.model_metadata:
                            del self.model_metadata[model_type]
                    
                    except Exception as model_error:
                        logger.error(f"Error unloading model {model_type}: {str(model_error)}", exc_info=True)
                        failed_unloads.append((model_type, str(model_error)))
            else:
                # Normal operation with Rich progress tracking
                try:
                    # Use a try-except block for the entire Progress context manager
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[bold blue]{task.description}"),
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TimeElapsedColumn(),
                        console=console
                    ) as progress:
                        # Create overall task
                        task = progress.add_task(f"Unloading {len(model_types)} models", total=len(model_types))
                        
                        # Unload each model with timeout protection
                        for model_type in model_types:
                            try:
                                progress.update(task, description=f"Unloading {model_type}...")
                            except Exception as progress_error:
                                logger.warning(f"Progress update error: {progress_error}")
                            
                            try:
                                # Create a task to unload the model with timeout
                                unload_task = asyncio.create_task(self._unload_model_with_timeout(model_type, timeout_per_model))
                                
                                # Wait for the task to complete
                                success = await unload_task
                                
                                if not success:
                                    logger.warning(f"Model {model_type} unload timed out after {timeout_per_model}s")
                                    try:
                                        console.print(f"[yellow]⚠ Model {model_type} unload timed out after {timeout_per_model}s[/yellow]")
                                    except Exception:
                                        pass
                                    failed_unloads.append((model_type, "timeout"))
                                
                                # Remove from local cache regardless of unload success
                                if model_type in self.loaded_models:
                                    del self.loaded_models[model_type]
                                if model_type in self.model_metadata:
                                    del self.model_metadata[model_type]
                            
                            except Exception as model_error:
                                logger.error(f"Error unloading model {model_type}: {str(model_error)}", exc_info=True)
                                try:
                                    console.print(f"[red]Error unloading {model_type}: {str(model_error)}[/red]")
                                except Exception:
                                    pass
                                failed_unloads.append((model_type, str(model_error)))
                            
                            # Update progress (with error handling)
                            try:
                                progress.update(task, advance=1)
                            except Exception as progress_error:
                                logger.warning(f"Progress advance error: {progress_error}")
                except Exception as progress_manager_error:
                    # If the progress manager fails, fall back to simple console output
                    logger.warning(f"Progress manager error: {progress_manager_error}. Falling back to simple unloading.")
                    
                    # Simpler fallback unloading
                    for model_type in model_types:
                        logger.info(f"Unloading {model_type}...")
                        
                        try:
                            # Directly unload without fancy progress tracking
                            success = await self._unload_model_with_timeout(model_type, timeout_per_model)
                            
                            if not success:
                                logger.warning(f"Model {model_type} unload timed out after {timeout_per_model}s")
                                failed_unloads.append((model_type, "timeout"))
                                
                            # Remove from local cache
                            if model_type in self.loaded_models:
                                del self.loaded_models[model_type]
                            if model_type in self.model_metadata:
                                del self.model_metadata[model_type]
                                
                        except Exception as model_error:
                            logger.error(f"Error unloading model {model_type}: {str(model_error)}")
                            failed_unloads.append((model_type, str(model_error)))
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Additional CUDA cleanup
                for device_id in range(torch.cuda.device_count()):
                    try:
                        torch.cuda.reset_peak_memory_stats(device_id)
                        torch.cuda.synchronize(device_id)
                    except Exception as cuda_error:
                        logger.warning(f"Error resetting CUDA device {device_id}: {str(cuda_error)}")
            
            # Report results
            if failed_unloads:
                logger.warning(f"Models unloaded with {len(failed_unloads)} issues")
                for model, reason in failed_unloads:
                    logger.warning(f"Unload issue: {model}: {reason}")
                
                # Try console output, but don't fail if it errors
                try:
                    console.print(Panel(
                        f"[bold yellow]Models unloaded with {len(failed_unloads)} issues:[/bold yellow]\n" +
                        "\n".join([f"[yellow]- {model}: {reason}[/yellow]" for model, reason in failed_unloads]),
                        border_style="yellow"
                    ))
                except Exception:
                    pass
            else:
                logger.info("All models unloaded successfully")
                try:
                    console.print(Panel("[bold green]✓ All models unloaded successfully[/bold green]", border_style="green"))
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error unloading models: {e}", exc_info=True)
            try:
                console.print(Panel(f"[bold red]⚠ Error unloading models:[/bold red]\n{str(e)}", border_style="red"))
            except Exception:
                pass
    
    async def unload_model(self, model_type: Union[ModelType, str]) -> None:
        """
        Unload a specific model to free memory
        
        Args:
            model_type (Union[ModelType, str]): Model type
        """
        # Convert enum to string if needed
        if isinstance(model_type, ModelType):
            model_type_str = model_type.value
        else:
            model_type_str = model_type
        
        # Log to file first for reliability
        logger.info(f"Unloading model: {model_type_str}")
        
        # Try console output but handle errors gracefully
        try:
            console.print(f"[cyan]Unloading model: [yellow]{model_type_str}[/yellow][/cyan]")
        except Exception as console_err:
            logger.warning(f"Console error while unloading {model_type_str}: {console_err}")
        
        try:
            # Use loader to unload model
            success = self.loader.unload_model(model_type_str)
            
            if success:
                # Remove from local cache
                if model_type_str in self.loaded_models:
                    del self.loaded_models[model_type_str]
                if model_type_str in self.model_metadata:
                    del self.model_metadata[model_type_str]
                
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Model {model_type_str} unloaded successfully")
                try:
                    console.print(f"[green]✓ Model {model_type_str} unloaded successfully[/green]")
                except Exception:
                    pass
            else:
                logger.warning(f"Failed to unload model {model_type_str}")
                try:
                    console.print(f"[yellow]⚠ Failed to unload model {model_type_str}[/yellow]")
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error unloading model {model_type_str}: {e}", exc_info=True)
            try:
                console.print(f"[red]⚠ Error unloading model {model_type_str}: {str(e)}[/red]")
            except Exception:
                pass
    
    async def _unload_model_with_timeout(self, model_type: str, timeout_seconds: float) -> bool:
        """
        Unload a model with timeout protection
        
        Args:
            model_type (str): Type of model to unload
            timeout_seconds (float): Maximum time to wait for unload
            
        Returns:
            bool: True if model was unloaded successfully, False if timeout occurred
        """
        try:
            # Get or create event loop - handle case where this is called in thread without loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If there's no event loop in the current thread
                logger.warning("No event loop in current thread, creating new loop for model unloading")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Use run_in_executor to perform the unload in a separate thread
            # This allows us to set a timeout since unload operations can block
            unload_task = loop.run_in_executor(
                None, 
                self.loader.unload_model,
                model_type
            )
            
            # Wait for the unload to complete with timeout
            try:
                success = await asyncio.wait_for(unload_task, timeout=timeout_seconds)
                return success
            except asyncio.TimeoutError:
                logger.warning(f"Unloading model {model_type} timed out after {timeout_seconds}s")
                # Just return False to indicate timeout, the caller will handle cleanup
                return False
            except asyncio.CancelledError:
                logger.warning(f"Unloading task for model {model_type} was cancelled")
                return False
                
        except Exception as e:
            logger.error(f"Error in _unload_model_with_timeout for {model_type}: {str(e)}", exc_info=True)
            # Don't re-raise to make shutdown more robust
            return False
    
    async def reload_model(self, model_type: Union[ModelType, str]) -> Dict[str, Any]:
        """
        Reload a previously unloaded model
        
        Args:
            model_type (Union[ModelType, str]): Model type
            
        Returns:
            Dict[str, Any]: Model information
        """
        # Convert enum to string if needed
        if isinstance(model_type, ModelType):
            model_type_str = model_type.value
        else:
            model_type_str = model_type
            
        console.print(f"[bold cyan]Reloading model: [yellow]{model_type_str}[/yellow][/bold cyan]")
        
        # Ensure model is unloaded first
        await self.unload_model(model_type_str)
        
        # Load the model
        return await self.load_model(model_type_str)
    
    # Add a wrapper cache to avoid recreating wrappers for the same model
    _wrapper_cache = {}
    
    async def run_model(self, model_type: str, method_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a model with the specified method and input data
        
        Args:
            model_type (str): Model type
            method_name (str): Method to call on the model wrapper
            input_data (Dict[str, Any]): Input data for the model
            
        Returns:
            Dict[str, Any]: Model output
        """
        # Use a more detailed log level to reduce verbose logging
        logger.debug(f"Running model {model_type}.{method_name}")
        
        # Ensure the model is loaded
        if model_type not in self.loaded_models:
            logger.info(f"Model {model_type} not loaded, loading now")
            model_info = await self.load_model(model_type)
            
            if not model_info.get("model"):
                raise ValueError(f"Failed to load model {model_type}")
        
        # Get the model and its wrapper
        model = self.loaded_models[model_type]
        tokenizer = self.model_metadata.get(model_type, {}).get("tokenizer")
        
        # Create input for the model wrapper
        from app.services.models.wrapper_base import ModelInput
        
        # Extract common fields from input_data
        text = input_data.get("text", "")
        source_language = input_data.get("source_language")
        target_language = input_data.get("target_language")
        context = input_data.get("context", [])
        parameters = input_data.get("parameters", {})
        
        # Create ModelInput instance
        model_input = ModelInput(
            text=text,
            source_language=source_language,
            target_language=target_language,
            context=context,
            parameters=parameters
        )
        
        # Use cached wrapper if available to avoid recreating it each time
        cache_key = f"{model_type}_{self.device}_{self.precision}"
        if cache_key in self._wrapper_cache:
            logger.debug(f"Using cached wrapper for {model_type}")
            wrapper = self._wrapper_cache[cache_key]
        else:
            # Import wrapper factory function
            from app.services.models.wrapper import create_model_wrapper
            
            # Create wrapper for the model
            logger.debug(f"Creating new wrapper for {model_type}")
            wrapper = create_model_wrapper(
                model_type,
                model,
                tokenizer,
                {"task": model_type, "device": self.device, "precision": self.precision}
            )
            
            # Cache the wrapper for future use
            self._wrapper_cache[cache_key] = wrapper
        
        # Call the appropriate method
        if method_name == "process":
            try:
                # Check if the process method is async
                if asyncio.iscoroutinefunction(wrapper.process):
                    # Asynchronous processing with proper await
                    logger.debug(f"Process method for {model_type} is async, awaiting result")
                    result = await wrapper.process(model_input)
                else:
                    # Synchronous processing
                    logger.debug(f"Process method for {model_type} is synchronous")
                    result = wrapper.process(model_input)
                    
                # Check if result is already a dictionary
                if isinstance(result, dict):
                    return result
                    
                # Return all fields from ModelOutput including enhanced metrics
                return {
                    "result": getattr(result, "result", None),
                    "metadata": getattr(result, "metadata", {}),
                    "metrics": getattr(result, "metrics", {}),
                    "performance_metrics": getattr(result, "performance_metrics", {}),
                    "memory_usage": getattr(result, "memory_usage", {}),
                    "operation_cost": getattr(result, "operation_cost", None),
                    "accuracy_score": getattr(result, "accuracy_score", None),
                    "truth_score": getattr(result, "truth_score", None)
                }
            except Exception as e:
                logger.error(f"Error in model processing: {str(e)}", exc_info=True)
                return {
                    "result": f"Error: {str(e)}",
                    "metadata": {
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                }
        elif method_name == "process_async":
            # Asynchronous processing
            result = await wrapper.process_async(model_input)
            
            # Check if result is already a dictionary
            if isinstance(result, dict):
                return result
                
            # Return all fields from ModelOutput including enhanced metrics
            return {
                "result": getattr(result, "result", None),
                "metadata": getattr(result, "metadata", {}),
                "metrics": getattr(result, "metrics", {}),
                "performance_metrics": getattr(result, "performance_metrics", {}),
                "memory_usage": getattr(result, "memory_usage", {}),
                "operation_cost": getattr(result, "operation_cost", None),
                "accuracy_score": getattr(result, "accuracy_score", None),
                "truth_score": getattr(result, "truth_score", None)
            }
        else:
            raise ValueError(f"Unknown method {method_name}")
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all loaded models
        
        Returns:
            Dict[str, Dict[str, Any]]: Model information
        """
        # Get information from loader
        loader_info = self.loader.get_model_info()
        
        # Enhance with local metadata
        info = {}
        for model_type, model_info in loader_info.items():
            metadata = self.model_metadata.get(model_type, {})
            model_config = metadata.get("config")
            
            # Estimate model memory usage
            memory_usage = self._estimate_model_memory_usage(model_type)
            
            info[model_type] = {
                "loaded": model_type in self.loaded_models,
                "model_name": model_config.model_name if model_config else model_info.get("model_name", "unknown"),
                "device": self.device,
                "precision": self.precision,
                "memory_usage": memory_usage,
                "loaded_at": metadata.get("loaded_at", None)
            }
        
        # Create a table to display model information
        table = Table(title="[bold]Loaded Models[/bold]")
        table.add_column("Model Type", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Model Name", style="yellow")
        table.add_column("Device", style="magenta")
        table.add_column("Precision", style="blue")
        table.add_column("Mem Usage", style="red")
        
        for model_type, model_info in info.items():
            status = "[green]✓ Loaded[/green]" if model_info.get("loaded", False) else "[dim]Not Loaded[/dim]"
            memory_usage = f"{model_info.get('memory_usage', {}).get('estimated_gb', 0):.2f} GB" if model_info.get("loaded", False) else "-"
            
            table.add_row(
                model_type,
                status,
                model_info.get("model_name", "unknown"),
                model_info.get("device", "unknown"),
                model_info.get("precision", "unknown"),
                memory_usage
            )
        
        # Add system memory information at the bottom
        system_memory = self._get_system_memory_usage()
        total_gpu_memory = self._get_total_gpu_memory_usage()
        
        console.print(table)
        
        # Print memory pressure information
        memory_table = Table(title="[bold]System Memory Pressure[/bold]")
        memory_table.add_column("Resource", style="cyan")
        memory_table.add_column("Used", style="yellow")
        memory_table.add_column("Total", style="blue")
        memory_table.add_column("Percentage", style="red")
        
        # Add RAM usage
        ram_percent = system_memory["used"] / system_memory["total"] * 100 if system_memory["total"] > 0 else 0
        memory_table.add_row(
            "RAM",
            f"{system_memory['used'] / (1024**3):.2f} GB",
            f"{system_memory['total'] / (1024**3):.2f} GB",
            f"{ram_percent:.1f}%" 
        )
        
        # Add GPU usage if available
        if total_gpu_memory["has_gpu"]:
            gpu_percent = total_gpu_memory["allocated"] / total_gpu_memory["total"] * 100 if total_gpu_memory["total"] > 0 else 0
            memory_table.add_row(
                "GPU Memory",
                f"{total_gpu_memory['allocated'] / (1024**3):.2f} GB",
                f"{total_gpu_memory['total'] / (1024**3):.2f} GB", 
                f"{gpu_percent:.1f}%"
            )
        
        console.print(memory_table)
        
        # Include memory pressure in the returned info
        info["_system"] = {
            "memory_pressure": {
                "ram": system_memory,
                "gpu": total_gpu_memory
            },
            "timestamp": time.time()
        }
            
        return info

    async def create_embeddings(self, texts: Union[str, List[str]], model_key: str = "embedding_model") -> List[List[float]]:
        """
        Create embeddings for the given texts using a specified model.
        
        Args:
            texts: Text or list of texts to embed
            model_key: Key of the model to use (default: "rag_retriever")
            
        Returns:
            List of embeddings as float lists
        """
        logger.info(f"Creating embeddings using model: {model_key}")
        
        # Ensure the model is loaded
        if model_key not in self.loaded_models:
            logger.info(f"Model {model_key} not loaded, loading now")
            model_info = await self.load_model(model_key)
            
            if not model_info.get("model"):
                logger.error(f"Failed to load model {model_key} for embeddings")
                # Return simple random embeddings as fallback
                import numpy as np
                if isinstance(texts, str):
                    return [np.random.rand(384).tolist()]  # Simple 384-dim embedding
                else:
                    return [np.random.rand(384).tolist() for _ in texts]
        
        # Convert single text to list
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = texts
        
        # Get the model and prepare for embedding creation
        model = self.loaded_models[model_key]
        
        try:
            # Use sentence-transformers if available
            if hasattr(model, "encode") and callable(model.encode):
                # Use the encode method directly
                import numpy as np
                if len(texts_list) == 0:
                    return []
                
                embeddings = model.encode(texts_list, convert_to_tensor=True, show_progress_bar=False)
                # Convert to numpy arrays and then to lists
                if hasattr(embeddings, "cpu") and callable(embeddings.cpu):
                    embeddings = embeddings.cpu().numpy()
                
                # Convert to list of lists
                embeddings_list = embeddings.tolist() if hasattr(embeddings, "tolist") else [emb.tolist() for emb in embeddings]
                
                return embeddings_list
            else:
                # Use the RAG wrapper approach
                from app.services.models.wrapper import ModelInput, create_model_wrapper
                
                # Get tokenizer
                tokenizer = self.model_metadata.get(model_key, {}).get("tokenizer")
                
                # Create wrapper for the model
                wrapper = create_model_wrapper(
                    model_key,
                    model,
                    tokenizer,
                    {"task": model_key, "device": self.device, "precision": self.precision}
                )
                
                # Create embeddings one by one
                embeddings = []
                for text in texts_list:
                    # Create input
                    model_input = ModelInput(text=text)
                    
                    # Get embedding
                    result = wrapper.process(model_input)
                    
                    if isinstance(result.result, list):
                        embeddings.append(result.result)
                    else:
                        # Try to get embedding from result
                        if hasattr(result, "embedding") and result.embedding is not None:
                            embeddings.append(result.embedding)
                        else:
                            # If all else fails, create a dummy embedding
                            import numpy as np
                            embeddings.append(np.random.rand(384).tolist())  # Simple 384-dim embedding
                
                return embeddings
        except Exception as e:
            # Log error and return fallback embeddings
            logger.error(f"Error creating embeddings: {str(e)}", exc_info=True)
            
            # Create fallback embeddings
            import numpy as np
            return [np.random.rand(384).tolist() for _ in texts_list]
        
    def _estimate_model_memory_usage(self, model_type: str) -> Dict[str, Any]:
        """
        Estimate memory usage for a loaded model
        
        Args:
            model_type: Type of model
            
        Returns:
            Dict containing memory usage estimates
        """
        if model_type not in self.loaded_models:
            return {"estimated_bytes": 0, "estimated_gb": 0}
            
        model = self.loaded_models[model_type]
        
        try:
            # Calculate parameter-based memory usage
            params_count = sum(p.numel() for p in model.parameters())
            
            # Estimate memory based on precision
            precision_multiplier = 4  # default for float32
            if self.precision == "float16":
                precision_multiplier = 2
            elif self.precision == "int8":
                precision_multiplier = 1
                
            # Parameter memory + overhead (50% more for buffers, optimizer states, etc)
            estimated_bytes = params_count * precision_multiplier * 1.5
            
            # Add CUDA memory for GPU models
            if self.device.startswith("cuda") and torch.cuda.is_available():
                for param in model.parameters():
                    if param.device.type == "cuda":
                        # Add device-specific memory if available
                        device_id = param.device.index
                        if device_id is not None:
                            allocated = torch.cuda.memory_allocated(device_id)
                            reserved = torch.cuda.memory_reserved(device_id)
                            # Use the larger of allocated or param-based estimation
                            estimated_bytes = max(estimated_bytes, allocated)
            
            return {
                "params_count": params_count,
                "precision_bytes": precision_multiplier,
                "estimated_bytes": estimated_bytes,
                "estimated_gb": estimated_bytes / (1024**3)
            }
        except Exception as e:
            logger.debug(f"Error estimating model memory for {model_type}: {e}")
            return {"estimated_bytes": 0, "estimated_gb": 0, "error": str(e)}
            
    def _get_system_memory_usage(self) -> Dict[str, Any]:
        """
        Get current system memory usage
        
        Returns:
            Dict with memory usage information
        """
        import psutil
        
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "free": memory.free,
            "percent": memory.percent
        }
        
    def _get_total_gpu_memory_usage(self) -> Dict[str, Any]:
        """
        Get total GPU memory usage across all devices
        
        Returns:
            Dict with GPU memory usage information
        """
        result = {
            "has_gpu": False,
            "total": 0,
            "allocated": 0,
            "reserved": 0,
            "free": 0,
            "devices": {}
        }
        
        try:
            if not torch.cuda.is_available():
                return result
                
            result["has_gpu"] = True
            
            # Track totals across all devices
            total_allocated = 0
            total_reserved = 0
            total_memory = 0
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                
                total_allocated += allocated
                total_reserved += reserved
                total_memory += device_props.total_memory
                
                result["devices"][i] = {
                    "name": device_props.name,
                    "total": device_props.total_memory,
                    "allocated": allocated,
                    "reserved": reserved,
                    "free": device_props.total_memory - reserved
                }
            
            result["total"] = total_memory
            result["allocated"] = total_allocated
            result["reserved"] = total_reserved
            result["free"] = total_memory - total_reserved
            
        except Exception as e:
            logger.debug(f"Error getting GPU memory usage: {e}")
            
        return result