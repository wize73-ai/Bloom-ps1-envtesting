#!/usr/bin/env python3
"""
CasaLingua Cloud Cost Estimator

This script provides a realistic cost estimation for running CasaLingua in a cloud environment
when integrated with Bloom Housing. It includes:
- Cost analysis for various deployment scenarios
- Usage-based pricing models
- Comparative cost analysis between providers
- ROI calculations based on accessibility improvements

Usage:
    python casalingua_cost_estimator.py
"""

import os
import sys
import json
import random
import asyncio
import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.layout import Layout
from rich import box
from rich.live import Live
from rich.bar import Bar

# Initialize console
console = Console()

# Constants for Bloom Housing traffic
ANNUAL_HITS = 1_800_000  # 1.8 million hits per year
MONTHLY_HITS = ANNUAL_HITS / 12
DAILY_HITS = ANNUAL_HITS / 365

# Translation needs distribution
LANGUAGE_DISTRIBUTION = {
    "es": 0.65,  # Spanish
    "zh": 0.12,  # Chinese
    "vi": 0.08,  # Vietnamese
    "tl": 0.05,  # Tagalog
    "ru": 0.04,  # Russian
    "ko": 0.03,  # Korean
    "ar": 0.02,  # Arabic
    "fr": 0.01,  # French
}

# User interaction types
INTERACTION_TYPES = {
    "form_translation": 0.35,      # Forms that need translation
    "document_viewing": 0.30,      # PDF/documents that need translation
    "listing_browsing": 0.20,      # Property listings needing translation
    "simplification": 0.10,        # Legal text simplification
    "chat_support": 0.05,          # Real-time chat support
}

# Breakdown by content size
CONTENT_SIZE_DISTRIBUTION = {
    "small": 0.60,    # Small snippets/forms (1-100 tokens)
    "medium": 0.30,   # Medium texts (101-500 tokens)
    "large": 0.09,    # Large documents (501-2000 tokens)
    "xlarge": 0.01,   # Very large documents (2001+ tokens)
}

# Token counts by content size (average)
TOKEN_COUNTS = {
    "small": 50,
    "medium": 250,
    "large": 1000,
    "xlarge": 3000,
}

# Cloud provider pricing
@dataclass
class CloudPricing:
    """Class for storing cloud provider pricing information"""
    provider_name: str
    compute_cost_per_hour: Dict[str, float]  # Machine type -> hourly cost
    storage_cost_per_gb_month: float
    network_cost_per_gb: float
    ml_inference_cost_per_1k_tokens: float
    gpu_cost_per_hour: Dict[str, float]  # GPU type -> hourly cost
    memory_gb_per_machine: Dict[str, int]  # Machine type -> GB of RAM
    managed_service_markup: float = 0.3  # 30% markup for managed services
    
    @property
    def ml_inference_cost_per_token(self) -> float:
        return self.ml_inference_cost_per_1k_tokens / 1000

# Define cloud provider pricing
AWS_PRICING = CloudPricing(
    provider_name="AWS",
    compute_cost_per_hour={
        "small": 0.0416,     # t4g.medium - $0.0416/hour
        "medium": 0.1664,    # t4g.xlarge - $0.1664/hour
        "large": 0.3328,     # t4g.2xlarge - $0.3328/hour
        "xlarge": 0.6656,    # t4g.4xlarge - $0.6656/hour
    },
    storage_cost_per_gb_month=0.08,  # EBS gp3 storage
    network_cost_per_gb=0.09,        # Data transfer out
    ml_inference_cost_per_1k_tokens=0.0004,  # Custom pricing for SageMaker
    gpu_cost_per_hour={
        "t4": 0.526,         # g4dn.xlarge with T4 GPU - $0.526/hour
        "a10g": 1.324,       # g5.xlarge with A10G GPU - $1.324/hour
    },
    memory_gb_per_machine={
        "small": 4,
        "medium": 16, 
        "large": 32,
        "xlarge": 64,
    }
)

GCP_PRICING = CloudPricing(
    provider_name="GCP",
    compute_cost_per_hour={
        "small": 0.0356,     # e2-standard-2 - $0.0356/hour
        "medium": 0.1424,    # e2-standard-8 - $0.1424/hour
        "large": 0.2848,     # e2-standard-16 - $0.2848/hour
        "xlarge": 0.5696,    # e2-standard-32 - $0.5696/hour
    },
    storage_cost_per_gb_month=0.04,  # Standard persistent disk
    network_cost_per_gb=0.08,        # Data transfer out
    ml_inference_cost_per_1k_tokens=0.00042,  # Custom pricing for Vertex AI
    gpu_cost_per_hour={
        "t4": 0.35,          # T4 GPU add-on - $0.35/hour
        "a100": 2.934,       # A100 GPU add-on - $2.934/hour
    },
    memory_gb_per_machine={
        "small": 4,
        "medium": 16, 
        "large": 32,
        "xlarge": 64,
    }
)

AZURE_PRICING = CloudPricing(
    provider_name="Azure",
    compute_cost_per_hour={
        "small": 0.052,      # B2ms - $0.052/hour
        "medium": 0.208,     # B8ms - $0.208/hour
        "large": 0.416,      # B16ms - $0.416/hour
        "xlarge": 0.832,     # B20ms - $0.832/hour
    },
    storage_cost_per_gb_month=0.0575,  # Standard SSD E10
    network_cost_per_gb=0.087,         # Data transfer out
    ml_inference_cost_per_1k_tokens=0.00038,  # Custom pricing for Azure ML
    gpu_cost_per_hour={
        "t4": 0.40,          # NC4as_T4_v3 - $0.40/hour
        "a100": 3.06,        # NC24ads_A100_v4 - $3.06/hour
    },
    memory_gb_per_machine={
        "small": 8,
        "medium": 32, 
        "large": 64,
        "xlarge": 128,
    }
)

# Usage scenarios
@dataclass
class UsageScenario:
    """Class for defining usage scenarios"""
    name: str
    description: str
    annual_volume: int
    avg_tokens_per_request: int
    requires_gpu: bool = False
    requires_verification: bool = False
    response_time_secs: float = 1.0
    
    @property
    def daily_volume(self) -> int:
        return int(self.annual_volume / 365)
    
    @property
    def monthly_volume(self) -> int:
        return int(self.annual_volume / 12)
    
    @property
    def total_annual_tokens(self) -> int:
        return self.annual_volume * self.avg_tokens_per_request

# Define Bloom Housing integration scenarios based on 1.8M annual hits
BLOOM_SCENARIOS = [
    UsageScenario(
        name="Housing Application Form Translation",
        description="Translation of housing application forms and input fields",
        annual_volume=int(ANNUAL_HITS * INTERACTION_TYPES["form_translation"]),
        avg_tokens_per_request=150,
        requires_gpu=False,
        requires_verification=True,
        response_time_secs=0.8
    ),
    UsageScenario(
        name="Property Listing Translation",
        description="Translation of property descriptions, amenities, and requirements",
        annual_volume=int(ANNUAL_HITS * INTERACTION_TYPES["listing_browsing"]),
        avg_tokens_per_request=200,
        requires_gpu=False,
        requires_verification=False,
        response_time_secs=1.0
    ),
    UsageScenario(
        name="Legal Document Simplification",
        description="Simplification of lease agreements and legal housing documents",
        annual_volume=int(ANNUAL_HITS * INTERACTION_TYPES["simplification"]),
        avg_tokens_per_request=800,
        requires_gpu=True,
        requires_verification=True,
        response_time_secs=2.5
    ),
    UsageScenario(
        name="Support Chat Translation",
        description="Real-time translation for support chat with housing providers",
        annual_volume=int(ANNUAL_HITS * INTERACTION_TYPES["chat_support"]),
        avg_tokens_per_request=100,
        requires_gpu=False,
        requires_verification=False,
        response_time_secs=0.5
    ),
    UsageScenario(
        name="Document/PDF Translation",
        description="Full translation of housing-related documents and PDFs",
        annual_volume=int(ANNUAL_HITS * INTERACTION_TYPES["document_viewing"]),
        avg_tokens_per_request=1200,
        requires_gpu=True,
        requires_verification=True,
        response_time_secs=3.0
    )
]

# Deployment scenarios
@dataclass
class DeploymentScenario:
    """Class for defining deployment scenarios"""
    name: str
    description: str
    machine_size: str
    replica_count: int
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    storage_gb: int = 100
    includes_redundancy: bool = False
    uses_managed_service: bool = False
    
    def get_monthly_compute_cost(self, provider: CloudPricing) -> float:
        """Calculate monthly compute cost"""
        base_cost = provider.compute_cost_per_hour[self.machine_size] * 24 * 30 * self.replica_count
        
        # Add GPU costs if applicable
        if self.gpu_type and self.gpu_count > 0:
            gpu_cost = provider.gpu_cost_per_hour[self.gpu_type] * self.gpu_count * 24 * 30 * self.replica_count
            base_cost += gpu_cost
        
        # Add managed service markup if applicable
        if self.uses_managed_service:
            base_cost *= (1 + provider.managed_service_markup)
            
        return base_cost
    
    def get_monthly_storage_cost(self, provider: CloudPricing) -> float:
        """Calculate monthly storage cost"""
        return self.storage_gb * provider.storage_cost_per_gb_month * self.replica_count
    
    def get_monthly_network_cost(self, provider: CloudPricing, monthly_gb_transfer: float) -> float:
        """Calculate monthly network cost"""
        return monthly_gb_transfer * provider.network_cost_per_gb
    
    def get_total_monthly_cost(self, provider: CloudPricing, monthly_gb_transfer: float) -> float:
        """Calculate total monthly infrastructure cost"""
        compute_cost = self.get_monthly_compute_cost(provider)
        storage_cost = self.get_monthly_storage_cost(provider)
        network_cost = self.get_monthly_network_cost(provider, monthly_gb_transfer)
        
        return compute_cost + storage_cost + network_cost
    
    def can_support_scenario(self, scenario: UsageScenario) -> bool:
        """Check if this deployment can handle the given scenario"""
        if scenario.requires_gpu and (not self.gpu_type or self.gpu_count == 0):
            return False
        
        # Check if we have enough memory to handle the model
        # This is a simplified check based on token size
        required_memory_gb = 4  # Base memory
        if scenario.avg_tokens_per_request > 500:
            required_memory_gb += 4
        if scenario.avg_tokens_per_request > 1000:
            required_memory_gb += 8
            
        # Convert machine size to memory
        if scenario.requires_gpu:
            # GPU memory is typically additional to system memory
            return True
        else:
            # Check system memory
            return required_memory_gb <= AZURE_PRICING.memory_gb_per_machine[self.machine_size]

# Define deployment scenarios
DEPLOYMENT_SCENARIOS = [
    DeploymentScenario(
        name="Small Deployment",
        description="Basic setup with minimal resources for testing/low traffic",
        machine_size="small",
        replica_count=2,
        storage_gb=100,
        includes_redundancy=False,
        uses_managed_service=False
    ),
    DeploymentScenario(
        name="Medium Deployment",
        description="Standard production setup for moderate traffic",
        machine_size="medium",
        replica_count=2,
        gpu_type="t4",
        gpu_count=1,
        storage_gb=200,
        includes_redundancy=True,
        uses_managed_service=False
    ),
    DeploymentScenario(
        name="Large Deployment",
        description="High-performance setup for significant traffic",
        machine_size="large",
        replica_count=3,
        gpu_type="t4",
        gpu_count=2,
        storage_gb=500,
        includes_redundancy=True,
        uses_managed_service=True
    ),
    DeploymentScenario(
        name="Premium Deployment",
        description="Enterprise-grade setup with high availability and best performance",
        machine_size="xlarge",
        replica_count=4,
        gpu_type="a100",
        gpu_count=1,
        storage_gb=1000,
        includes_redundancy=True,
        uses_managed_service=True
    )
]

class CostEstimator:
    """Class for estimating cloud costs for CasaLingua"""
    
    def __init__(self):
        self.providers = [AWS_PRICING, GCP_PRICING, AZURE_PRICING]
        self.scenarios = BLOOM_SCENARIOS
        self.deployments = DEPLOYMENT_SCENARIOS
        
    def estimate_monthly_token_cost(self, scenario: UsageScenario, provider: CloudPricing) -> float:
        """Estimate the monthly cost for ML inference tokens"""
        monthly_tokens = scenario.total_annual_tokens / 12
        return monthly_tokens * provider.ml_inference_cost_per_token
    
    def estimate_monthly_data_transfer(self, scenarios: List[UsageScenario]) -> float:
        """Estimate monthly data transfer volume in GB"""
        total_monthly_tokens = sum(scenario.total_annual_tokens / 12 for scenario in scenarios)
        
        # Roughly estimate: 1000 tokens ≈ 1.5 KB for requests, 3 KB for responses
        total_kb = total_monthly_tokens * 4.5 / 1000
        total_gb = total_kb / (1024 * 1024)
        
        # Add overhead for HTTP/API communication
        return total_gb * 1.2  # 20% overhead
    
    def get_deployment_recommendation(self, scenarios: List[UsageScenario]) -> DeploymentScenario:
        """Get recommended deployment based on usage scenarios"""
        total_annual_volume = sum(scenario.annual_volume for scenario in scenarios)
        requires_gpu = any(scenario.requires_gpu for scenario in scenarios)
        avg_response_time = sum(scenario.response_time_secs * scenario.annual_volume 
                               for scenario in scenarios) / total_annual_volume
        
        if total_annual_volume < 500_000:
            # Low volume
            if requires_gpu:
                return self.deployments[1]  # Medium deployment
            else:
                return self.deployments[0]  # Small deployment
        elif total_annual_volume < 2_000_000:
            # Medium volume
            if avg_response_time > 2.0 or requires_gpu:
                return self.deployments[2]  # Large deployment
            else:
                return self.deployments[1]  # Medium deployment
        else:
            # High volume
            return self.deployments[3]  # Premium deployment
    
    def estimate_total_monthly_cost(self, scenarios: List[UsageScenario], 
                                   deployment: DeploymentScenario,
                                   provider: CloudPricing) -> Dict[str, float]:
        """Estimate total monthly cost for scenarios with a given deployment"""
        # Calculate infrastructure costs
        monthly_data_transfer = self.estimate_monthly_data_transfer(scenarios)
        compute_cost = deployment.get_monthly_compute_cost(provider)
        storage_cost = deployment.get_monthly_storage_cost(provider)
        network_cost = deployment.get_monthly_network_cost(provider, monthly_data_transfer)
        
        # Calculate token/inference costs
        token_costs = sum(self.estimate_monthly_token_cost(scenario, provider) 
                          for scenario in scenarios)
        
        # Calculate verification costs (if applicable)
        verification_costs = 0.0
        for scenario in scenarios:
            if scenario.requires_verification:
                # Verification costs approximately 20% more in tokens
                monthly_tokens = scenario.total_annual_tokens / 12
                verification_tokens = monthly_tokens * 0.2
                verification_costs += verification_tokens * provider.ml_inference_cost_per_token
        
        # Calculate support costs (estimate 10% of infrastructure)
        infrastructure_cost = compute_cost + storage_cost + network_cost
        support_cost = infrastructure_cost * 0.1
        
        # Return breakdown
        return {
            "compute": compute_cost,
            "storage": storage_cost,
            "network": network_cost,
            "token_inference": token_costs,
            "verification": verification_costs,
            "support": support_cost,
            "total": (compute_cost + storage_cost + network_cost + 
                     token_costs + verification_costs + support_cost)
        }
    
    def get_roi_metrics(self, annual_cost: float, scenarios: List[UsageScenario]) -> Dict[str, Any]:
        """Calculate ROI metrics for accessibility improvements"""
        # Total users impacted
        total_requests = sum(scenario.annual_volume for scenario in scenarios)
        
        # Estimate number of unique users (typically 5-10% of total hits for housing sites)
        unique_users = int(total_requests * 0.08)  # 8% unique users
        
        # Estimate percentage of limited English proficiency (LEP) users
        lep_percentage = sum(LANGUAGE_DISTRIBUTION.values())
        lep_users = int(unique_users * lep_percentage)
        
        # Estimate improvement metrics
        completion_rate_improvement = 0.25  # 25% improvement in form completion
        error_reduction = 0.35  # 35% reduction in errors
        support_call_reduction = 0.40  # 40% reduction in support calls
        
        # Estimate monetary impact
        avg_cost_per_support_call = 12.00  # $12 per support call
        avg_cost_per_incomplete_application = 35.00  # $35 per incomplete application
        avg_cost_per_application_error = 28.00  # $28 per application with errors
        
        # Assume 30% of LEP users would have needed support calls without translation
        support_calls_avoided = int(lep_users * 0.3 * support_call_reduction)
        support_cost_saved = support_calls_avoided * avg_cost_per_support_call
        
        # Assume 40% of LEP users would have had incomplete applications without translation
        incomplete_applications_avoided = int(lep_users * 0.4 * completion_rate_improvement)
        incomplete_app_cost_saved = incomplete_applications_avoided * avg_cost_per_incomplete_application
        
        # Assume 25% of LEP users would have had application errors without translation
        errors_avoided = int(lep_users * 0.25 * error_reduction)
        error_cost_saved = errors_avoided * avg_cost_per_application_error
        
        # Calculate total annual savings
        total_annual_savings = support_cost_saved + incomplete_app_cost_saved + error_cost_saved
        
        # Calculate ROI
        roi = ((total_annual_savings - annual_cost) / annual_cost) * 100
        
        # Calculate payback period (in months)
        monthly_savings = total_annual_savings / 12
        monthly_cost = annual_cost / 12
        if monthly_savings > 0:
            payback_period = annual_cost / monthly_savings
        else:
            payback_period = float('inf')
        
        return {
            "lep_users_served": lep_users,
            "support_calls_avoided": support_calls_avoided,
            "incomplete_applications_avoided": incomplete_applications_avoided,
            "errors_avoided": errors_avoided,
            "annual_cost_savings": total_annual_savings,
            "roi_percentage": roi,
            "payback_period_months": payback_period
        }

    def display_cost_breakdown(self, costs: Dict[str, float]) -> Table:
        """Create a Rich table with cost breakdown"""
        table = Table(title="Monthly Cost Breakdown", box=box.ROUNDED)
        
        table.add_column("Cost Category", style="cyan")
        table.add_column("Amount (USD)", style="green", justify="right")
        table.add_column("Percentage", style="yellow", justify="right")
        
        total = costs["total"]
        
        for category, amount in costs.items():
            if category != "total":
                percentage = (amount / total) * 100 if total > 0 else 0
                table.add_row(
                    category.replace("_", " ").title(),
                    f"${amount:.2f}",
                    f"{percentage:.1f}%"
                )
        
        table.add_row("Total", f"${total:.2f}", "100.0%", style="bold")
        
        return table
    
    def display_roi_metrics(self, roi_data: Dict[str, Any]) -> Table:
        """Create a Rich table with ROI metrics"""
        table = Table(title="ROI Metrics", box=box.ROUNDED)
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Limited English Users Served", f"{roi_data['lep_users_served']:,}")
        table.add_row("Support Calls Avoided", f"{roi_data['support_calls_avoided']:,}")
        table.add_row("Incomplete Applications Avoided", f"{roi_data['incomplete_applications_avoided']:,}")
        table.add_row("Application Errors Avoided", f"{roi_data['errors_avoided']:,}")
        table.add_row("Annual Cost Savings", f"${roi_data['annual_cost_savings']:,.2f}")
        table.add_row("ROI", f"{roi_data['roi_percentage']:.1f}%")
        table.add_row("Payback Period", f"{roi_data['payback_period_months']:.1f} months")
        
        return table
    
    def display_provider_comparison(self, scenarios: List[UsageScenario], 
                                  deployment: DeploymentScenario) -> Table:
        """Create a Rich table comparing costs across cloud providers"""
        table = Table(title=f"Cloud Provider Comparison - {deployment.name}", box=box.ROUNDED)
        
        table.add_column("Provider", style="cyan")
        table.add_column("Monthly Cost", style="green", justify="right")
        table.add_column("Annual Cost", style="green", justify="right")
        table.add_column("Infrastructure", style="yellow", justify="right")
        table.add_column("ML Services", style="yellow", justify="right")
        
        for provider in self.providers:
            costs = self.estimate_total_monthly_cost(scenarios, deployment, provider)
            monthly_total = costs["total"]
            annual_total = monthly_total * 12
            
            # Split between infrastructure and ML services
            infrastructure = costs["compute"] + costs["storage"] + costs["network"] + costs["support"]
            ml_services = costs["token_inference"] + costs["verification"]
            
            table.add_row(
                provider.provider_name,
                f"${monthly_total:.2f}",
                f"${annual_total:.2f}",
                f"${infrastructure:.2f}",
                f"${ml_services:.2f}"
            )
        
        return table
    
    def display_scenarios_breakdown(self, scenarios: List[UsageScenario]) -> Table:
        """Create a Rich table with scenario details"""
        table = Table(title="Usage Scenarios Breakdown", box=box.ROUNDED)
        
        table.add_column("Scenario", style="cyan")
        table.add_column("Annual Volume", style="green", justify="right")
        table.add_column("Avg Tokens", style="yellow", justify="right")
        table.add_column("Total Annual Tokens", style="magenta", justify="right")
        table.add_column("GPU?", justify="center")
        table.add_column("Verify?", justify="center")
        
        for scenario in scenarios:
            table.add_row(
                scenario.name,
                f"{scenario.annual_volume:,}",
                f"{scenario.avg_tokens_per_request:,}",
                f"{scenario.total_annual_tokens:,}",
                "✓" if scenario.requires_gpu else "✗",
                "✓" if scenario.requires_verification else "✗"
            )
        
        return table

    async def run_demo(self):
        """Run the cost estimation demo"""
        console.clear()
        console.rule("[bold blue]CasaLingua Cloud Cost Estimator[/]")
        console.print("[bold cyan]Estimating operational costs for Bloom Housing integration[/]")
        console.print(f"Based on annual traffic of [bold]{ANNUAL_HITS:,}[/] hits")
        console.print("")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(description="Analyzing usage patterns...", total=None)
            await asyncio.sleep(1.5)
            
            progress.update(task, description="Calculating infrastructure requirements...")
            await asyncio.sleep(1.5)
            
            progress.update(task, description="Estimating costs across providers...")
            await asyncio.sleep(1.5)
            
            progress.update(task, description="Computing ROI metrics...")
            await asyncio.sleep(1.5)
        
        # Display usage scenarios
        console.print(self.display_scenarios_breakdown(self.scenarios))
        console.print("")
        
        # Get recommended deployment
        recommended_deployment = self.get_deployment_recommendation(self.scenarios)
        console.print(f"[bold green]Recommended Deployment:[/] {recommended_deployment.name}")
        console.print(f"[dim]{recommended_deployment.description}[/]")
        console.print("")
        
        # Display provider comparison
        console.print(self.display_provider_comparison(self.scenarios, recommended_deployment))
        console.print("")
        
        # Select the cheapest provider for detailed analysis
        provider_costs = [
            (provider, self.estimate_total_monthly_cost(self.scenarios, recommended_deployment, provider))
            for provider in self.providers
        ]
        cheapest_provider = min(provider_costs, key=lambda x: x[1]["total"])[0]
        detailed_costs = self.estimate_total_monthly_cost(self.scenarios, recommended_deployment, cheapest_provider)
        
        # Display detailed cost breakdown
        console.print(f"[bold]Detailed Monthly Cost Breakdown for {cheapest_provider.provider_name}[/]")
        console.print(self.display_cost_breakdown(detailed_costs))
        console.print("")
        
        # Calculate and display ROI metrics
        annual_cost = detailed_costs["total"] * 12
        roi_metrics = self.get_roi_metrics(annual_cost, self.scenarios)
        console.print(self.display_roi_metrics(roi_metrics))
        console.print("")
        
        # Create cost by scenario visualization
        console.print("[bold]Cost Distribution by Scenario[/]")
        
        # Calculate token costs per scenario for chosen provider
        scenario_costs = []
        for scenario in self.scenarios:
            monthly_tokens = scenario.total_annual_tokens / 12
            token_cost = monthly_tokens * cheapest_provider.ml_inference_cost_per_token
            if scenario.requires_verification:
                verification_tokens = monthly_tokens * 0.2
                verification_cost = verification_tokens * cheapest_provider.ml_inference_cost_per_token
            else:
                verification_cost = 0
                
            # Estimate scenario's portion of infrastructure cost (based on token proportion)
            total_tokens = sum(s.total_annual_tokens for s in self.scenarios)
            token_proportion = scenario.total_annual_tokens / total_tokens
            infra_cost = (detailed_costs["compute"] + detailed_costs["storage"] + 
                        detailed_costs["network"] + detailed_costs["support"]) * token_proportion
            
            total_scenario_cost = token_cost + verification_cost + infra_cost
            scenario_costs.append((scenario.name, total_scenario_cost))
        
        # Create a simple bar chart visualization with Rich
        max_cost = max(cost for _, cost in scenario_costs)
        for name, cost in scenario_costs:
            bar_width = int((cost / max_cost) * 40)  # Scale to max 40 chars
            bar = Bar(size=40, begin=0, end=max_cost, color="green")
            console.print(f"{name:<32} ${cost:>8.2f}  {bar}")
        
        console.print("")
        console.print("[bold]Key Insights:[/]")
        console.print(f"• Expected monthly cost using {cheapest_provider.provider_name}: [bold]${detailed_costs['total']:.2f}[/]")
        console.print(f"• Expected annual cost: [bold]${annual_cost:.2f}[/]")
        console.print(f"• Return on Investment (ROI): [bold]{roi_metrics['roi_percentage']:.1f}%[/]")
        console.print(f"• Payback period: [bold]{roi_metrics['payback_period_months']:.1f} months[/]")
        console.print(f"• Limited English users served annually: [bold]{roi_metrics['lep_users_served']:,}[/]")
        
        console.print("")
        console.print("[dim]Note: This is a cost estimation for planning purposes. Actual costs may vary based on implementation details and usage patterns.[/]")
        console.print("")

async def main():
    """Main function to run the cost estimation demo"""
    try:
        estimator = CostEstimator()
        await estimator.run_demo()
    except KeyboardInterrupt:
        console.print("[bold red]Demo interrupted by user[/]")
    except Exception as e:
        console.print(f"[bold red]Error during demo: {str(e)}[/]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())