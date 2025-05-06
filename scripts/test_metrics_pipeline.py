#!/usr/bin/env python3
"""
Test enhanced metrics propagation through the translation pipeline.
"""
import sys
import os
import asyncio
import json
from typing import Dict, Any
from dataclasses import asdict
from rich.console import Console
from rich.panel import Panel

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required modules
from app.services.models.wrapper import ModelOutput, ModelInput
from app.services.models.manager import EnhancedModelManager
from app.services.models.loader import ModelLoader
from app.utils.config import load_config
from app.core.pipeline.translator import TranslationPipeline
from app.core.pipeline.processor import UnifiedProcessor

# Create rich console
console = Console()

async def main():
    """Test metrics propagation through the pipeline layers"""
    console.print(Panel("[bold blue]Testing Enhanced Metrics Propagation[/bold blue]", border_style="blue"))
    
    # Load the configuration
    console.print("[yellow]Loading configuration...[/yellow]")
    config = load_config()
    registry_config = load_config("config/model_registry.json")
    
    # Create model loader
    console.print("[yellow]Creating model loader...[/yellow]")
    model_loader = ModelLoader(config, registry_config)
    
    # Create hardware info dict
    hardware_info = {
        "cpu": {
            "cores": os.cpu_count(),
            "utilization": 0.5  # Just a placeholder
        },
        "gpu": {
            "has_gpu": False,
            "cuda_available": False,
            "mps_available": False
        },
        "memory": {
            "total": 16 * 1024 * 1024 * 1024,  # 16 GB
            "available": 8 * 1024 * 1024 * 1024  # 8 GB
        }
    }
    
    # Create model manager
    console.print("[yellow]Creating model manager...[/yellow]")
    model_manager = EnhancedModelManager(model_loader, hardware_info, config)
    
    # Mock model processing for testing
    class MockModelWrapper:
        def process(self, input_data):
            # Create a fake model output with metrics
            return ModelOutput(
                result="Hola, esto es una prueba de la API de traducción.",
                metadata={"model": "mock_model"},
                metrics={"total_time": 0.5},
                performance_metrics={
                    "preprocess_time": 0.1,
                    "inference_time": 0.3,
                    "postprocess_time": 0.1,
                    "total_time": 0.5,
                    "tokens_processed": {
                        "input_tokens_estimate": 10,
                        "output_tokens_estimate": 12
                    },
                    "throughput": {
                        "tokens_per_second": 44.0,
                        "chars_per_second": 220.0
                    }
                },
                memory_usage={
                    "before": {"total": 16000000000, "used": 8000000000, "free": 8000000000, "percent": 50.0},
                    "after": {"total": 16000000000, "used": 8100000000, "free": 7900000000, "percent": 50.6},
                    "difference": {"total": 0, "used": 100000000, "free": -100000000, "percent": 0.6}
                },
                operation_cost=0.0005,
                accuracy_score=0.95,
                truth_score=0.92
            )
    
    # Override the run_model method for testing
    async def mock_run_model(self, model_type, method_name, input_data):
        wrapper = MockModelWrapper()
        result = wrapper.process(input_data)
        return {
            "result": result.result,
            "metadata": result.metadata,
            "metrics": result.metrics,
            "performance_metrics": result.performance_metrics,
            "memory_usage": result.memory_usage,
            "operation_cost": result.operation_cost,
            "accuracy_score": result.accuracy_score,
            "truth_score": result.truth_score
        }
    
    # Patch the model manager
    model_manager.run_model = mock_run_model.__get__(model_manager)
    
    # Create translation pipeline
    console.print("[yellow]Creating translation pipeline...[/yellow]")
    translation_pipeline = TranslationPipeline(model_manager, config, registry_config)
    
    # Create unified processor
    console.print("[yellow]Creating unified processor...[/yellow]")
    
    # Create a mock metrics object
    class MockMetrics:
        async def record_pipeline_execution(self, **kwargs):
            pass
            
        async def record_language_operation(self, **kwargs):
            pass
    
    processor = UnifiedProcessor(model_manager, config, MockMetrics())
    processor.translation_pipeline = translation_pipeline
    processor.initialized = True
    
    # Test 1: Test ModelWrapper metrics
    console.print("\n[bold]Test 1: ModelWrapper Metrics[/bold]")
    model_input = ModelInput(
        text="Hello, this is a test of the translation API.",
        source_language="en",
        target_language="es"
    )
    
    console.print("[yellow]Running mock model...[/yellow]")
    model_result = await model_manager.run_model("translation", "process", model_input)
    
    console.print(Panel(
        f"[bold cyan]Model Result Contains:[/bold cyan]\n"
        f"Performance Metrics: {bool(model_result.get('performance_metrics'))}\n"
        f"Memory Usage: {bool(model_result.get('memory_usage'))}\n"
        f"Operation Cost: {model_result.get('operation_cost')}\n"
        f"Accuracy Score: {model_result.get('accuracy_score')}\n"
        f"Truth Score: {model_result.get('truth_score')}",
        title="Model Output",
        border_style="green"
    ))
    
    # Test 2: Translation Pipeline metrics propagation
    console.print("\n[bold]Test 2: Translation Pipeline Metrics[/bold]")
    
    console.print("[yellow]Calling translation pipeline translate_text...[/yellow]")
    pipeline_result = await translation_pipeline.translate_text(
        text="Hello, this is a test of the translation API.",
        source_language="en",
        target_language="es"
    )
    
    console.print(Panel(
        f"[bold cyan]Pipeline Result Contains:[/bold cyan]\n"
        f"Performance Metrics: {bool(pipeline_result.get('performance_metrics'))}\n"
        f"Memory Usage: {bool(pipeline_result.get('memory_usage'))}\n"
        f"Operation Cost: {pipeline_result.get('operation_cost')}\n"
        f"Accuracy Score: {pipeline_result.get('accuracy_score')}\n"
        f"Truth Score: {pipeline_result.get('truth_score')}",
        title="Translation Pipeline Output",
        border_style="green"
    ))
    
    # Test 3: Processor metrics propagation
    console.print("\n[bold]Test 3: Processor Metrics[/bold]")
    
    console.print("[yellow]Calling processor process_translation...[/yellow]")
    processor_result = await processor.process_translation(
        text="Hello, this is a test of the translation API.",
        source_language="en",
        target_language="es"
    )
    
    console.print(Panel(
        f"[bold cyan]Processor Result Contains:[/bold cyan]\n"
        f"Performance Metrics: {bool(processor_result.get('performance_metrics'))}\n"
        f"Memory Usage: {bool(processor_result.get('memory_usage'))}\n"
        f"Operation Cost: {processor_result.get('operation_cost')}\n"
        f"Accuracy Score: {processor_result.get('accuracy_score')}\n"
        f"Truth Score: {processor_result.get('truth_score')}",
        title="Processor Output",
        border_style="green"
    ))
    
    # Print full processor result for inspection
    console.print("\n[bold]Processor Full Output[/bold]")
    console.print(Panel(
        json.dumps(processor_result, indent=2, default=str),
        title="Full Processor Response",
        border_style="blue"
    ))
    
    # Verify metrics propagation
    console.print("\n[bold]Verification Summary[/bold]")
    
    # Model stage verification
    console.print(
        "[green]✓[/green]" if model_result.get('performance_metrics') else "[red]✗[/red]",
        "Model outputs performance_metrics"
    )
    console.print(
        "[green]✓[/green]" if model_result.get('memory_usage') else "[red]✗[/red]",
        "Model outputs memory_usage"
    )
    console.print(
        "[green]✓[/green]" if model_result.get('operation_cost') is not None else "[red]✗[/red]",
        "Model outputs operation_cost"
    )
    
    # Pipeline stage verification
    console.print(
        "[green]✓[/green]" if pipeline_result.get('performance_metrics') else "[red]✗[/red]",
        "Pipeline outputs performance_metrics"
    )
    console.print(
        "[green]✓[/green]" if pipeline_result.get('memory_usage') else "[red]✗[/red]",
        "Pipeline outputs memory_usage"
    )
    console.print(
        "[green]✓[/green]" if pipeline_result.get('operation_cost') is not None else "[red]✗[/red]",
        "Pipeline outputs operation_cost"
    )
    
    # Processor stage verification
    console.print(
        "[green]✓[/green]" if processor_result.get('performance_metrics') else "[red]✗[/red]",
        "Processor outputs performance_metrics"
    )
    console.print(
        "[green]✓[/green]" if processor_result.get('memory_usage') else "[red]✗[/red]",
        "Processor outputs memory_usage"
    )
    console.print(
        "[green]✓[/green]" if processor_result.get('operation_cost') is not None else "[red]✗[/red]",
        "Processor outputs operation_cost"
    )
    
    # Overall success
    success = all([
        model_result.get('performance_metrics'),
        model_result.get('memory_usage'),
        model_result.get('operation_cost') is not None,
        pipeline_result.get('performance_metrics'),
        pipeline_result.get('memory_usage'),
        pipeline_result.get('operation_cost') is not None,
        processor_result.get('performance_metrics'),
        processor_result.get('memory_usage'),
        processor_result.get('operation_cost') is not None
    ])
    
    if success:
        console.print(Panel(
            "[bold green]✓ Enhanced metrics are successfully propagated through all pipeline layers![/bold green]",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[bold red]✗ Enhanced metrics propagation has issues[/bold red]",
            border_style="red"
        ))
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)