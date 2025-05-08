#!/usr/bin/env python3
"""
CasaLingua Complete Demo

This comprehensive demo showcases:
1. Translation and simplification functionality
2. Performance metrics and metadata visualization
3. Cloud cost estimation for integration with Bloom Housing
4. ROI analysis for language accessibility

The demo runs for approximately 3 minutes with interactive
visualizations and transitions between sections.

Usage:
    python casalingua_complete_demo.py
"""

import os
import sys
import time
import json
import asyncio
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.layout import Layout
    from rich import box
    from rich.live import Live
    from rich.text import Text
    from rich.style import Style
    from rich.bar import Bar
except ImportError:
    print("This demo requires the rich library. Please install it with:")
    print("pip install rich")
    sys.exit(1)

# Add parent directory to path so we can import app modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Import required libraries for API calls
import requests
import aiohttp
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("casalingua.demo")

# API endpoint configuration
API_ENDPOINT = "http://localhost:8000"  # Default CasaLingua API endpoint

# Define API interface classes
class LiveAPIClient:
    """Client for interacting with a running CasaLingua service"""
    
    def __init__(self, base_url=API_ENDPOINT):
        self.base_url = base_url
        self.session = None
    
    async def initialize(self):
        """Initialize the API client"""
        self.session = aiohttp.ClientSession()
        # Check if service is available
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status != 200:
                    print(f"Warning: CasaLingua API returned status {response.status}")
                else:
                    result = await response.json()
                    print(f"✓ Connected to CasaLingua API - Status: {result.get('status', 'OK')}")
        except Exception as e:
            print(f"Warning: Could not connect to CasaLingua API at {self.base_url}: {e}")
            print("The demo will run with simulated functionality.")
    
    async def close(self):
        """Close the API client session"""
        if self.session:
            await self.session.close()
    
    async def translate_text(self, text, source_language, target_language):
        """Call the translation API endpoint"""
        try:
            # Make sure text input is properly formatted as a string
            if not isinstance(text, str):
                text = str(text)
                
            async with self.session.post(
                f"{self.base_url}/pipeline/translate",
                json={
                    "text": text,
                    "source_language": source_language,
                    "target_language": target_language
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # Extract the relevant data from the API response
                    return {
                        "translated_text": result.get("result", {}).get("translated_text", 
                                                                      f"[API Error: No translation returned]"),
                        "model_used": result.get("metadata", {}).get("model_used", "unknown"),
                        "cached": result.get("metadata", {}).get("cached", False)
                    }
                else:
                    logger.error(f"Translation API error: {response.status}")
                    return {
                        "translated_text": f"[API Error: {response.status}]",
                        "model_used": "error",
                        "cached": False
                    }
        except Exception as e:
            logger.error(f"Translation API call failed: {e}")
            return {
                "translated_text": f"[API Error: {str(e)}]",
                "model_used": "error",
                "cached": False
            }
    
    async def simplify_text(self, text, target_grade_level):
        """Call the simplification API endpoint"""
        try:
            # Make sure text input is properly formatted as a string
            if not isinstance(text, str):
                text = str(text)
                
            # Map target_grade_level to the expected format
            # The API expects level 1-5 or simple/medium/complex
            level = target_grade_level
            if isinstance(target_grade_level, int):
                # Use as is if it's already a number
                pass
            elif target_grade_level.lower() == "simple":
                level = "simple"
            elif target_grade_level.lower() == "medium":
                level = "medium"
            elif target_grade_level.lower() == "complex":
                level = "complex"
            
            async with self.session.post(
                f"{self.base_url}/pipeline/simplify",
                json={
                    "text": text,
                    "target_level": level,
                    "language": "en",
                    "preserve_formatting": True
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # Extract result from the response structure
                    if "data" in result and "simplified_text" in result["data"]:
                        return {
                            "simplified_text": result["data"]["simplified_text"],
                            "model_used": result["data"].get("model_used", "simplifier")
                        }
                    else:
                        logger.error(f"Unexpected response format: {result}")
                        return {
                            "simplified_text": f"[API Error: Unexpected response format]",
                            "model_used": "error"
                        }
                else:
                    error_text = await response.text()
                    logger.error(f"Simplification API error: {response.status} - {error_text}")
                    return {
                        "simplified_text": f"[API Error: {response.status}]",
                        "model_used": "error"
                    }
        except Exception as e:
            logger.error(f"Simplification API call failed: {e}")
            return {
                "simplified_text": f"[API Error: {str(e)}]",
                "model_used": "error"
            }
    
    async def audit_translation(self, source_text, translation, source_lang, target_lang):
        """Call the verification API endpoint"""
        try:
            async with self.session.post(
                f"{self.base_url}/verification/translation",
                json={
                    "source_text": source_text,
                    "translation": translation,
                    "source_language": source_lang,
                    "target_language": target_lang
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "score": result.get("result", {}).get("score", 0.5),
                        "issues": result.get("result", {}).get("issues", [])
                    }
                else:
                    logger.error(f"Verification API error: {response.status}")
                    return {
                        "score": 0.5,
                        "issues": [{"type": "api_error", "severity": "high", 
                                  "description": f"API returned status {response.status}"}]
                    }
        except Exception as e:
            logger.error(f"Verification API call failed: {e}")
            return {
                "score": 0.5,
                "issues": [{"type": "api_error", "severity": "high", 
                          "description": f"API call failed: {str(e)}"}]
            }
    
    async def check_health(self):
        """Check the health of the CasaLingua service"""
        try:
            async with self.session.get(f"{self.base_url}/health/detailed") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "error", "message": f"API returned status {response.status}"}
        except Exception as e:
            return {"status": "error", "message": f"Health check failed: {str(e)}"}
    
    async def list_loaded_models(self):
        """Get list of loaded models from the API"""
        try:
            async with self.session.get(f"{self.base_url}/admin/models/info") as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("models", {}).keys()
                else:
                    return ["Error fetching models"]
        except Exception as e:
            return ["API error", str(e)]


# If the API client fails, use these simulation classes for fallback
class SimulatedAPIClient:
    """Simulated client when the real API isn't available"""
    
    def __init__(self, **kwargs):
        print("Using simulated CasaLingua API")
    
    async def initialize(self):
        await asyncio.sleep(0.5)
        print("✓ Initialized simulated CasaLingua API")
    
    async def close(self):
        pass
    
    async def translate_text(self, text, source_language, target_language):
        await asyncio.sleep(1)  # Simulate processing time
        return {
            "translated_text": f"[Translated to {target_language}]: {text[:30]}...",
            "model_used": random.choice(["mbart-large", "mt5-base", "nllb-medium"]),
            "cached": random.choice([True, False])
        }
    
    async def simplify_text(self, text, target_grade_level):
        await asyncio.sleep(1.5)  # Simulate processing time
        return {
            "simplified_text": f"[Simplified to {target_grade_level} level]: {text[:20]}...",
            "model_used": "t5-base-simplifier"
        }
    
    async def audit_translation(self, source_text, translation, source_lang, target_lang):
        await asyncio.sleep(0.8)
        score = random.uniform(0.75, 0.98)
        issues = []
        if score < 0.85:
            issues = [{"type": "accuracy", "severity": "medium", "description": "Potential meaning loss"}]
        return {"score": score, "issues": issues}
    
    async def check_health(self):
        await asyncio.sleep(0.3)
        return {
            "status": "ok",
            "uptime": "1h 23m 45s",
            "models": {"loaded": 4, "total": 6},
            "memory": {"available": "4.2GB", "total": "8GB"},
            "requests": {"total": 128, "success_rate": 0.98}
        }
    
    async def list_loaded_models(self):
        return ["mbart-large", "mt5-base", "t5-simplifier", "xlm-roberta-detector"]


# Function to get the appropriate client
def get_api_client(simulate=False):
    """Return either a live API client or simulated client"""
    if simulate:
        return SimulatedAPIClient()
    return LiveAPIClient()

# Initialize console for pretty output
console = Console()
# Already have logger set up above
# logger = get_logger("casalingua.demo")

# -----------------------------------------------------------------------------
# SECTION 1: Sample texts and configuration for the demo
# -----------------------------------------------------------------------------

# Housing-specific sample texts for demonstration
HOUSING_SENTENCES = [
    "The lease agreement requires all tenants to maintain their units in good condition.",
    "Rent payments are due on the first day of each month, with a five-day grace period.",
    "Upon moving out, tenants must return all keys to the property management office.",
    "The security deposit will be returned within 30 days after the final inspection.",
    "Maintenance requests should be submitted through the resident portal.",
]

COMPLEX_HOUSING_TEXTS = [
    "The lessor shall be permitted to access the aforementioned premises for inspection purposes upon providing the lessee with a minimum of twenty-four (24) hours advance written notification, except in cases of emergency wherein immediate ingress may be necessitated.",
    
    "Tenant hereby acknowledges and agrees that, pursuant to the terms and conditions set forth in this agreement, failure to remit monthly rental payments in a timely manner may result in the assessment of late fees, which shall accrue at a rate of five percent (5%) of the outstanding balance per diem, commencing on the sixth (6th) day following the payment due date.",
    
    "The security deposit, in the amount specified in Section 1.4 of this agreement, shall be held in escrow by the property management entity and shall be disbursed to the tenant within thirty (30) calendar days subsequent to the termination of tenancy, less any deductions for damages exceeding normal wear and tear, outstanding rental obligations, or cleaning expenses necessitated by the tenant's occupancy.",
    
    "Notwithstanding anything to the contrary contained herein, the Landlord reserves the right to terminate this Lease Agreement prior to the expiration of the primary term in the event that the Tenant violates any of the covenants, terms, or conditions specified herein, particularly those pertaining to timely payment of rent, proper maintenance of the premises, or adherence to community regulations.",
]

LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
    "de": "German", 
    "zh": "Chinese",
    "vi": "Vietnamese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic"
}

API_BASE_URL = "http://localhost:8000"

# -----------------------------------------------------------------------------
# SECTION 2: Cloud cost estimation data and classes
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# SECTION 3: Performance metrics and visualization
# -----------------------------------------------------------------------------

class MetricsDisplay:
    """Class for handling and displaying metrics"""
    
    def __init__(self):
        self.metrics = {
            "translation_time": [],
            "simplification_time": [],
            "veracity_time": [],
            "token_counts": [],
            "model_usage": {},
            "language_pairs": {},
            "quality_scores": [],
        }
    
    def add_translation_metric(self, source_lang: str, target_lang: str, 
                              time_taken: float, model_used: str, 
                              token_count: int, quality_score: float = None):
        """Add metrics from a translation operation"""
        self.metrics["translation_time"].append(time_taken)
        
        # Track model usage
        if model_used not in self.metrics["model_usage"]:
            self.metrics["model_usage"][model_used] = 0
        self.metrics["model_usage"][model_used] += 1
        
        # Track language pair
        lang_pair = f"{source_lang}-{target_lang}"
        if lang_pair not in self.metrics["language_pairs"]:
            self.metrics["language_pairs"][lang_pair] = 0
        self.metrics["language_pairs"][lang_pair] += 1
        
        # Track token count
        self.metrics["token_counts"].append(token_count)
        
        # Track quality if provided
        if quality_score is not None:
            self.metrics["quality_scores"].append(quality_score)
    
    def add_simplification_metric(self, time_taken: float, token_count: int):
        """Add metrics from a simplification operation"""
        self.metrics["simplification_time"].append(time_taken)
        self.metrics["token_counts"].append(token_count)
    
    def add_veracity_metric(self, time_taken: float):
        """Add metrics from a veracity audit operation"""
        self.metrics["veracity_time"].append(time_taken)
    
    def get_avg_translation_time(self) -> float:
        """Get average translation time in seconds"""
        if not self.metrics["translation_time"]:
            return 0.0
        return sum(self.metrics["translation_time"]) / len(self.metrics["translation_time"])
    
    def get_avg_simplification_time(self) -> float:
        """Get average simplification time in seconds"""
        if not self.metrics["simplification_time"]:
            return 0.0
        return sum(self.metrics["simplification_time"]) / len(self.metrics["simplification_time"])
    
    def get_avg_quality_score(self) -> float:
        """Get average quality score"""
        if not self.metrics["quality_scores"]:
            return 0.0
        return sum(self.metrics["quality_scores"]) / len(self.metrics["quality_scores"])
    
    def get_most_used_model(self) -> str:
        """Get the most frequently used model"""
        if not self.metrics["model_usage"]:
            return "none"
        return max(self.metrics["model_usage"].items(), key=lambda x: x[1])[0]
    
    def get_metrics_table(self) -> Table:
        """Generate a Rich table with metrics information"""
        table = Table(title="Processing Metrics", box=box.ROUNDED)
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Add translation metrics
        if self.metrics["translation_time"]:
            avg_time = self.get_avg_translation_time()
            table.add_row("Avg. Translation Time", f"{avg_time:.3f} seconds")
        
        # Add simplification metrics
        if self.metrics["simplification_time"]:
            avg_time = self.get_avg_simplification_time()
            table.add_row("Avg. Simplification Time", f"{avg_time:.3f} seconds")
        
        # Add token metrics
        if self.metrics["token_counts"]:
            avg_tokens = sum(self.metrics["token_counts"]) / len(self.metrics["token_counts"])
            table.add_row("Avg. Token Count", f"{avg_tokens:.1f} tokens")
        
        # Add quality metrics
        if self.metrics["quality_scores"]:
            avg_quality = self.get_avg_quality_score()
            table.add_row("Avg. Quality Score", f"{avg_quality:.2f}")
        
        # Add model usage
        if self.metrics["model_usage"]:
            most_used = self.get_most_used_model()
            table.add_row("Most Used Model", most_used)
        
        # Add operation counts
        total_ops = (len(self.metrics["translation_time"]) + 
                    len(self.metrics["simplification_time"]) + 
                    len(self.metrics["veracity_time"]))
        table.add_row("Total Operations", str(total_ops))
        
        return table
    
    def get_language_table(self) -> Table:
        """Generate a Rich table with language pair information"""
        table = Table(title="Language Pair Usage", box=box.ROUNDED)
        
        table.add_column("Language Pair", style="yellow")
        table.add_column("Count", style="green")
        
        for lang_pair, count in self.metrics["language_pairs"].items():
            table.add_row(lang_pair, str(count))
        
        return table

class EnhancedMetadata:
    """Class for enhancing and displaying metadata from operations"""
    
    def __init__(self):
        self.current_metadata = {}
    
    def update_metadata(self, metadata: Dict[str, Any]):
        """Update current metadata with new information"""
        self.current_metadata = metadata
    
    def get_metadata_table(self) -> Table:
        """Generate a Rich table with current metadata"""
        table = Table(title="Operation Metadata", box=box.ROUNDED)
        
        table.add_column("Parameter", style="blue")
        table.add_column("Value", style="white")
        
        # Select the most relevant metadata fields to display
        important_fields = [
            "model_used", "timestamp", "duration_ms", "source_language", 
            "target_language", "token_count", "quality_score", 
            "confidence", "cached", "method"
        ]
        
        for field in important_fields:
            if field in self.current_metadata:
                value = self.current_metadata[field]
                # Format the value based on its type
                if isinstance(value, float):
                    if field == "duration_ms":
                        formatted_value = f"{value:.2f} ms"
                    else:
                        formatted_value = f"{value:.3f}"
                elif field in ["source_language", "target_language"] and value in LANGUAGES:
                    formatted_value = f"{value} ({LANGUAGES[value]})"
                else:
                    formatted_value = str(value)
                
                table.add_row(field.replace("_", " ").title(), formatted_value)
        
        return table

# -----------------------------------------------------------------------------
# SECTION 4: Cost estimator functionality
# -----------------------------------------------------------------------------

class CostEstimator:
    """Class for estimating cloud costs for CasaLingua"""
    
    def __init__(self):
        self.providers = [AWS_PRICING, GCP_PRICING, AZURE_PRICING]
        self.scenarios = BLOOM_SCENARIOS
        self.deployments = DEPLOYMENT_SCENARIOS
    
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

# -----------------------------------------------------------------------------
# SECTION 5: Main CasaLingua demo class
# -----------------------------------------------------------------------------

class CasaLinguaDemo:
    """CasaLingua comprehensive demo with functionality and cost estimation"""
    
    def __init__(self, duration=180, simulate=False):  # 3 minutes default
        """Initialize the demo with specified duration in seconds"""
        self.duration = duration
        self.start_time = None
        self.end_time = None
        self.simulate = simulate
        
        # API client for real or simulated functionality
        self.api_client = None
        
        # Metrics and visualization components
        self.enhanced_metrics = MetricsDisplay()
        self.metadata_display = EnhancedMetadata()
        
        # Cost estimation components
        self.cost_estimator = CostEstimator()

    async def initialize(self):
        """Initialize all necessary components"""
        with console.status("[bold green]Initializing CasaLingua components..."):
            # Initialize the API client
            try:
                # Try to connect to the real CasaLingua API first
                self.api_client = get_api_client(simulate=self.simulate)
                await self.api_client.initialize()
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to connect to CasaLingua API: {e}[/yellow]")
                console.print("[yellow]Falling back to simulation mode[/yellow]")
                self.api_client = get_api_client(simulate=True)
                await self.api_client.initialize()
        
        console.print("[bold green]✓[/] CasaLingua components initialized successfully")

    async def perform_translation(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Perform translation and collect metrics"""
        start_time = time.time()
        
        # Perform translation through the API client
        result = await self.api_client.translate_text(
            text=text,
            source_language=source_lang,
            target_language=target_lang
        )
        
        # Calculate time taken
        time_taken = time.time() - start_time
        
        # Estimate token count (simplistic model)
        token_count = len(text.split())
        
        # Add metrics
        self.enhanced_metrics.add_translation_metric(
            source_lang=source_lang,
            target_lang=target_lang,
            time_taken=time_taken,
            model_used=result.get('model_used', 'default'),
            token_count=token_count
        )
        
        # Create and update metadata
        metadata = {
            "model_used": result.get('model_used', 'mbart'),
            "timestamp": datetime.now().isoformat(),
            "duration_ms": time_taken * 1000,
            "source_language": source_lang,
            "target_language": target_lang,
            "token_count": token_count,
            "method": "translate_text",
            "cached": result.get('cached', False)
        }
        self.metadata_display.update_metadata(metadata)
        
        return result

    async def perform_simplification(self, text: str, target_level: str = "simple") -> Dict[str, Any]:
        """Perform text simplification and collect metrics"""
        start_time = time.time()
        
        # Perform simplification through the API client
        result = await self.api_client.simplify_text(
            text=text,
            target_grade_level=target_level
        )
        
        # Calculate time taken
        time_taken = time.time() - start_time
        
        # Estimate token count
        token_count = len(text.split())
        
        # Add metrics
        self.enhanced_metrics.add_simplification_metric(
            time_taken=time_taken,
            token_count=token_count
        )
        
        # Create and update metadata
        metadata = {
            "model_used": result.get('model_used', 'simplifier'),
            "timestamp": datetime.now().isoformat(),
            "duration_ms": time_taken * 1000,
            "source_language": "en",
            "token_count": token_count,
            "target_complexity": target_level,
            "method": "simplify_text",
            "cached": result.get('cached', False)
        }
        self.metadata_display.update_metadata(metadata)
        
        return result

    async def perform_verification(self, source_text: str, translated_text: str, 
                                source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Perform translation verification and collect metrics"""
        start_time = time.time()
        
        # Perform verification through the API client
        result = await self.api_client.audit_translation(
            source_text=source_text,
            translation=translated_text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        # Calculate time taken
        time_taken = time.time() - start_time
        
        # Add metrics
        self.enhanced_metrics.add_veracity_metric(time_taken=time_taken)
        
        # Add quality score to translation metrics
        quality_score = result.get('score', 0.0)
        self.enhanced_metrics.add_translation_metric(
            source_lang=source_lang,
            target_lang=target_lang,
            time_taken=0,  # Already counted
            model_used="verification",
            token_count=0,  # Already counted
            quality_score=quality_score
        )
        
        # Create and update metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "duration_ms": time_taken * 1000,
            "quality_score": quality_score,
            "source_language": source_lang,
            "target_language": target_lang,
            "method": "verify_translation",
            "issues_found": len(result.get('issues', []))
        }
        self.metadata_display.update_metadata(metadata)
        
        return result

    async def demonstrate_translation_with_metrics(self):
        """Demonstrate translation with enhanced metrics display"""
        # Select a random sentence and language
        text = random.choice(HOUSING_SENTENCES)
        target_lang = random.choice(list(LANGUAGES.keys()))
        
        # Create a layout for this demonstration
        layout = Layout()
        layout.split_column(
            Layout(Panel(f"[bold]Source Text (English):[/]\n[italic]{text}[/]", 
                       box=box.ROUNDED, style="green"), name="source"),
            Layout(name="result"),
            Layout(name="metadata", size=12)
        )
        
        layout["result"].split_row(
            Layout(name="translation", ratio=3),
            Layout(name="metrics", ratio=2)
        )
        
        layout["metadata"].split_row(
            Layout(name="metadata_table"),
            Layout(name="language_table")
        )
        
        # Start with empty tables
        layout["translation"].update(Panel("Translating...", title="Translation Result", box=box.ROUNDED))
        layout["metrics"].update(self.enhanced_metrics.get_metrics_table())
        layout["metadata_table"].update(self.metadata_display.get_metadata_table())
        layout["language_table"].update(self.enhanced_metrics.get_language_table())
        
        # Display the layout and perform translation
        with Live(layout, refresh_per_second=4, screen=True):
            # Perform translation
            result = await self.perform_translation(text, "en", target_lang)
            
            # Update result in the layout
            lang_name = LANGUAGES.get(target_lang, target_lang.upper())
            translation_panel = Panel(
                f"[bold]Translated Text ({lang_name}):[/]\n[italic]{result['translated_text']}[/]\n\n"
                f"Model Used: [bold cyan]{result.get('model_used', 'mbart')}[/]",
                title="Translation Result",
                box=box.ROUNDED,
                style="blue"
            )
            layout["translation"].update(translation_panel)
            
            # Update metrics and metadata in the layout
            layout["metrics"].update(self.enhanced_metrics.get_metrics_table())
            layout["metadata_table"].update(self.metadata_display.get_metadata_table())
            layout["language_table"].update(self.enhanced_metrics.get_language_table())
            
            # Pause for readability
            await asyncio.sleep(4)

    async def demonstrate_simplification_with_metrics(self):
        """Demonstrate text simplification with enhanced metrics display"""
        # Select a complex sentence
        text = random.choice(COMPLEX_HOUSING_TEXTS)
        
        # Create a layout for this demonstration
        layout = Layout()
        layout.split_column(
            Layout(Panel(f"[bold]Complex Housing Text:[/]\n[italic]{text}[/]", 
                       box=box.ROUNDED, style="magenta"), name="source"),
            Layout(name="result"),
            Layout(name="metadata", size=12)
        )
        
        layout["result"].split_row(
            Layout(name="simplification", ratio=3),
            Layout(name="metrics", ratio=2)
        )
        
        layout["metadata"].split_row(
            Layout(name="metadata_table"),
            Layout(name="processing_stages", size=15)
        )
        
        # Start with empty content
        layout["simplification"].update(
            Panel("Simplifying...", title="Simplification Result", box=box.ROUNDED))
        layout["metrics"].update(self.enhanced_metrics.get_metrics_table())
        layout["metadata_table"].update(self.metadata_display.get_metadata_table())
        
        # Display the layout and perform simplification
        with Live(layout, refresh_per_second=4, screen=True):
            # Create processing stages panel with progress indicators
            stages = ["Text Analysis", "Complexity Detection", "Sentence Restructuring", 
                     "Vocabulary Simplification", "Final Output"]
            
            # Simulate processing stages - one by one
            progress_panel = ""
            for i, stage in enumerate(stages):
                # Update all previous stages as completed
                for j in range(i):
                    progress_panel += f"{stages[j]}: [bold green]Complete[/]\n"
                
                # Update current stage as in progress
                progress_panel += f"{stage}: [bold yellow]In Progress[/]\n"
                
                # Update remaining stages as pending
                for j in range(i+1, len(stages)):
                    progress_panel += f"{stages[j]}: [dim]Pending[/]\n"
                
                # Update the panel
                layout["processing_stages"].update(
                    Panel(progress_panel.strip(), title="Simplification Process", box=box.ROUNDED)
                )
                
                # Wait a moment to show progress
                await asyncio.sleep(0.5)
            
            # Perform simplification
            result = await self.perform_simplification(text, "4th")
            
            # All stages complete
            progress_panel = ""
            for stage in stages:
                progress_panel += f"{stage}: [bold green]Complete[/]\n"
            
            layout["processing_stages"].update(
                Panel(progress_panel.strip(), title="Simplification Process", box=box.ROUNDED)
            )
            
            # Update result in the layout
            simplification_panel = Panel(
                f"[bold]Simplified Text (4th grade level):[/]\n[italic]{result['simplified_text']}[/]",
                title="Simplification Result",
                box=box.ROUNDED,
                style="green"
            )
            layout["simplification"].update(simplification_panel)
            
            # Update metrics and metadata in the layout
            layout["metrics"].update(self.enhanced_metrics.get_metrics_table())
            layout["metadata_table"].update(self.metadata_display.get_metadata_table())
            
            # Pause for readability
            await asyncio.sleep(4)

    async def demonstrate_cost_estimation(self):
        """Demonstrate cost estimation for Bloom Housing integration"""
        # Create a layout for this demonstration
        layout = Layout()
        layout.split_column(
            Layout(Panel(f"[bold]CasaLingua Cloud Cost Estimation[/]\n"
                      f"Based on Bloom Housing's [bold cyan]1.8 million annual hits[/]", 
                      box=box.ROUNDED, style="yellow"), 
                  name="header"),
            Layout(name="content"),
            Layout(size=2, name="footer")
        )
        
        layout["content"].split_row(
            Layout(name="left_column", ratio=3),
            Layout(name="right_column", ratio=2)
        )
        
        layout["left_column"].split_column(
            Layout(name="scenarios"),
            Layout(name="recommended")
        )
        
        layout["right_column"].split_column(
            Layout(name="costs"),
            Layout(name="roi")
        )
        
        # Start with empty content
        layout["scenarios"].update(
            Panel("Analyzing usage patterns...", title="Usage Scenarios", box=box.ROUNDED))
        layout["recommended"].update(
            Panel("Calculating infrastructure requirements...", title="Deployment Recommendation", box=box.ROUNDED))
        layout["costs"].update(
            Panel("Estimating costs...", title="Cost Breakdown", box=box.ROUNDED))
        layout["roi"].update(
            Panel("Computing ROI metrics...", title="ROI Analysis", box=box.ROUNDED))
        layout["footer"].update(Text("Processing...", style="dim"))
        
        # Display the layout and perform cost estimation
        with Live(layout, refresh_per_second=4, screen=True):
            # Step 1: Display usage scenarios
            await asyncio.sleep(1)
            scenarios_table = self.cost_estimator.display_scenarios_breakdown(BLOOM_SCENARIOS)
            layout["scenarios"].update(scenarios_table)
            
            # Step 2: Get and display recommended deployment
            await asyncio.sleep(1)
            recommended_deployment = self.cost_estimator.get_deployment_recommendation(BLOOM_SCENARIOS)
            deployment_panel = Panel(
                f"[bold green]{recommended_deployment.name}[/]\n"
                f"[dim]{recommended_deployment.description}[/]\n\n"
                f"• Machine Size: [bold]{recommended_deployment.machine_size}[/]\n"
                f"• Replica Count: [bold]{recommended_deployment.replica_count}[/]\n"
                f"• GPU: [bold]{recommended_deployment.gpu_type or 'None'}[/] "
                f"× {recommended_deployment.gpu_count}\n"
                f"• Storage: [bold]{recommended_deployment.storage_gb} GB[/]\n"
                f"• Managed Service: [bold]{'Yes' if recommended_deployment.uses_managed_service else 'No'}[/]",
                title="Recommended Deployment",
                box=box.ROUNDED,
                style="green"
            )
            layout["recommended"].update(deployment_panel)
            
            # Step 3: Calculate costs with the cheapest provider
            await asyncio.sleep(1)
            provider_costs = [
                (provider, self.cost_estimator.estimate_total_monthly_cost(
                    BLOOM_SCENARIOS, recommended_deployment, provider))
                for provider in self.cost_estimator.providers
            ]
            cheapest_provider = min(provider_costs, key=lambda x: x[1]["total"])[0]
            detailed_costs = self.cost_estimator.estimate_total_monthly_cost(
                BLOOM_SCENARIOS, recommended_deployment, cheapest_provider)
            
            cost_table = self.cost_estimator.display_cost_breakdown(detailed_costs)
            layout["costs"].update(cost_table)
            
            # Step 4: Calculate and display ROI metrics
            await asyncio.sleep(1)
            annual_cost = detailed_costs["total"] * 12
            roi_metrics = self.cost_estimator.get_roi_metrics(annual_cost, BLOOM_SCENARIOS)
            roi_table = self.cost_estimator.display_roi_metrics(roi_metrics)
            layout["roi"].update(roi_table)
            
            # Update footer with key insights
            layout["footer"].update(
                Text(f"Monthly Cost: ${detailed_costs['total']:.2f} | "
                    f"Annual ROI: {roi_metrics['roi_percentage']:.1f}% | "
                    f"Payback: {roi_metrics['payback_period_months']:.1f} months", 
                    style="cyan")
            )
            
            # Pause for readability
            await asyncio.sleep(4)

    async def run_demo(self):
        """Run the full demonstration sequence"""
        console.clear()
        console.rule("[bold blue]CasaLingua Comprehensive Demo[/]")
        console.print("[bold cyan]Showcasing functionality, performance, and cost estimation[/]")
        console.print(f"Current time: {datetime.now().isoformat()}")
        console.print("")
        
        try:
            # Initialize components
            await self.initialize()
            
            # Set start and end times
            self.start_time = time.time()
            self.end_time = self.start_time + self.duration
            
            # Main demo loop
            demo_sequence = [
                self.demonstrate_translation_with_metrics,
                self.demonstrate_simplification_with_metrics,
                self.demonstrate_cost_estimation
            ]
            
            sequence_index = 0
            demo_count = 0
            while time.time() < self.end_time:
                # Show section transition
                section_name = demo_sequence[sequence_index].__name__.replace("demonstrate_", "").replace("_", " ").title()
                console.rule(f"[bold green]Section {sequence_index + 1}: {section_name}[/]")
                
                # Run the next demo in sequence
                await demo_sequence[sequence_index]()
                
                # Move to next demo
                sequence_index = (sequence_index + 1) % len(demo_sequence)
                demo_count += 1
                
                # Show remaining time
                remaining = int(self.end_time - time.time())
                if remaining > 0:
                    console.print(f"[dim]Demo will continue for approximately {remaining} more seconds...[/]")
                    console.print("")
                
                # Short delay between demonstrations
                await asyncio.sleep(1)
            
            # Final summary
            console.rule("[bold green]Demo Complete[/]")
            console.print(f"[bold]Demonstrated {demo_count} features across {len(demo_sequence)} sections[/]")
            console.print("[bold cyan]Key Takeaways:[/]")
            console.print("• CasaLingua provides specialized housing-domain translation and simplification")
            console.print("• Performance metrics show real-time processing capabilities")
            console.print("• Integration with Bloom Housing can serve thousands of LEP users")
            
            # Calculate an estimated ROI for display
            provider = self.cost_estimator.providers[0]  # Use the first provider
            deployment = self.cost_estimator.deployments[1]  # Use medium deployment
            costs = self.cost_estimator.estimate_total_monthly_cost(BLOOM_SCENARIOS, deployment, provider)
            yearly_cost = costs["total"] * 12
            roi_metrics = self.cost_estimator.get_roi_metrics(yearly_cost, BLOOM_SCENARIOS)
            
            console.print(f"• Positive ROI of approximately {int(roi_metrics['roi_percentage'])}% with payback in months")
            console.print("")
            console.print("[bold]Thank you for exploring CasaLingua's capabilities![/]")
            console.print("")
            console.print("[dim]For a detailed cost breakdown, visit: [link=https://www.casalingua.ai/cost-analysis]https://www.casalingua.ai/cost-analysis[/link][/dim]")
            console.print("[dim]LEP: Limited English Proficiency - individuals who have limited ability to read, speak, write, or understand English[/dim]")
        
        finally:
            # Clean up API client connection
            if self.api_client:
                await self.api_client.close()

async def main():
    """Main function to run the comprehensive demo"""
    try:
        demo = CasaLinguaDemo(duration=180)  # 3-minute demo
        await demo.run_demo()
    except KeyboardInterrupt:
        console.print("[bold red]Demo interrupted by user[/]")
    except Exception as e:
        console.print(f"[bold red]Error during demo: {str(e)}[/]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())