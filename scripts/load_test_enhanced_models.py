#!/usr/bin/env python3
"""
Load Testing Script for Enhanced Models in CasaLingua

This script performs load testing on the enhanced language models, 
specifically the translation and simplification functionality.
It measures performance metrics including throughput, latency, and
memory usage under various load conditions.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import time
import asyncio
import random
import statistics
import argparse
import psutil
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.logging import get_logger
from app.core.enhanced_integrations import setup_enhanced_components
from app.core.pipeline.translator import TranslationPipeline
from app.core.pipeline.simplifier import SimplificationPipeline
from app.services.models.manager import EnhancedModelManager

logger = get_logger(__name__)

# Sample texts for testing in different languages
SAMPLE_TEXTS = {
    "en": [
        "The implementation of advanced language models has revolutionized natural language processing, enabling more accurate translations and text simplifications across multiple domains and complexity levels.",
        "Climate change presents a significant global challenge that requires immediate and coordinated action from governments, industries, and individuals to mitigate its harmful effects on ecosystems and human populations.",
        "Quantum computing leverages principles of quantum mechanics to process information in ways that classical computers cannot, potentially solving complex problems exponentially faster.",
        "The economic implications of artificial intelligence adoption across various industries include workforce transformation, productivity enhancements, and the creation of entirely new business models and services.",
        "Effective healthcare systems must balance accessibility, quality, and cost-efficiency while addressing preventative care, chronic disease management, and emerging public health challenges."
    ],
    "es": [
        "La implementación de modelos lingüísticos avanzados ha revolucionado el procesamiento del lenguaje natural, permitiendo traducciones y simplificaciones de texto más precisas en múltiples dominios y niveles de complejidad.",
        "El cambio climático presenta un desafío global significativo que requiere acción inmediata y coordinada de gobiernos, industrias e individuos para mitigar sus efectos dañinos en los ecosistemas y las poblaciones humanas.",
        "La computación cuántica aprovecha los principios de la mecánica cuántica para procesar información de maneras que las computadoras clásicas no pueden, potencialmente resolviendo problemas complejos exponencialmente más rápido.",
        "Las implicaciones económicas de la adopción de la inteligencia artificial en varias industrias incluyen la transformación de la fuerza laboral, mejoras en la productividad y la creación de modelos de negocio y servicios completamente nuevos.",
        "Los sistemas de salud efectivos deben equilibrar la accesibilidad, calidad y eficiencia de costos mientras abordan la atención preventiva, el manejo de enfermedades crónicas y los desafíos emergentes de salud pública."
    ],
    "fr": [
        "L'implémentation de modèles linguistiques avancés a révolutionné le traitement du langage naturel, permettant des traductions et des simplifications de texte plus précises à travers de multiples domaines et niveaux de complexité.",
        "Le changement climatique présente un défi mondial important qui nécessite une action immédiate et coordonnée des gouvernements, des industries et des individus pour atténuer ses effets nocifs sur les écosystèmes et les populations humaines.",
        "L'informatique quantique exploite les principes de la mécanique quantique pour traiter l'information d'une manière que les ordinateurs classiques ne peuvent pas, résolvant potentiellement des problèmes complexes exponentiellement plus rapidement.",
        "Les implications économiques de l'adoption de l'intelligence artificielle dans diverses industries comprennent la transformation de la main-d'œuvre, des améliorations de la productivité et la création de modèles commerciaux et de services entièrement nouveaux.",
        "Les systèmes de santé efficaces doivent équilibrer accessibilité, qualité et rentabilité tout en abordant les soins préventifs, la gestion des maladies chroniques et les défis émergents de santé publique."
    ],
    "de": [
        "Die Implementierung fortschrittlicher Sprachmodelle hat die natürliche Sprachverarbeitung revolutioniert und ermöglicht genauere Übersetzungen und Textvereinfachungen über mehrere Domänen und Komplexitätsstufen hinweg.",
        "Der Klimawandel stellt eine erhebliche globale Herausforderung dar, die sofortige und koordinierte Maßnahmen von Regierungen, Industrien und Einzelpersonen erfordert, um seine schädlichen Auswirkungen auf Ökosysteme und menschliche Bevölkerungen zu mildern.",
        "Quantencomputer nutzen Prinzipien der Quantenmechanik, um Informationen auf eine Weise zu verarbeiten, die klassische Computer nicht können, und lösen möglicherweise komplexe Probleme exponentiell schneller.",
        "Die wirtschaftlichen Auswirkungen der Einführung künstlicher Intelligenz in verschiedenen Branchen umfassen die Transformation der Arbeitskräfte, Produktivitätssteigerungen und die Schaffung völlig neuer Geschäftsmodelle und Dienstleistungen.",
        "Effektive Gesundheitssysteme müssen Zugänglichkeit, Qualität und Kosteneffizienz in Einklang bringen und gleichzeitig präventive Pflege, Management chronischer Krankheiten und neu auftretende Herausforderungen im Bereich der öffentlichen Gesundheit angehen."
    ]
}

# Domains for testing simplification
DOMAINS = ["legal", "medical", "technical", "financial", "educational"]

class LoadTester:
    """Load tester for CasaLingua language models."""
    
    def __init__(self):
        """Initialize the load tester."""
        self.translator = None
        self.simplifier = None
        self.model_manager = None
        self.process = psutil.Process(os.getpid())
        self.metrics = {
            "translation": {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "latencies": [],
                "throughput": 0.0,
                "memory_usage": []
            },
            "simplification": {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "latencies": [],
                "throughput": 0.0,
                "memory_usage": []
            }
        }
        
    async def initialize(self):
        """Initialize components and models."""
        logger.info("Initializing load tester and components...")
        
        # Setup enhanced components
        await setup_enhanced_components()
        
        # Initialize model manager
        self.model_manager = EnhancedModelManager()
        
        # Initialize translator
        self.translator = TranslationPipeline(model_manager=self.model_manager)
        
        # Initialize simplifier
        self.simplifier = SimplificationPipeline(model_manager=self.model_manager)
        
        logger.info("Load tester initialized")
        
    async def run_translation_test(self, concurrency: int, num_requests: int):
        """
        Run load test for translation functionality.
        
        Args:
            concurrency: Number of concurrent requests
            num_requests: Total number of requests to make
        """
        logger.info(f"Starting translation load test with concurrency={concurrency}, requests={num_requests}")
        
        start_time = time.time()
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        # Create and gather translation tasks
        tasks = []
        for i in range(num_requests):
            tasks.append(self._run_translation_request(i, semaphore))
            
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        # Calculate metrics
        total_time = time.time() - start_time
        self.metrics["translation"]["throughput"] = self.metrics["translation"]["successes"] / total_time
        
        # Log results
        self._log_test_results("translation", total_time)
        
        return self.metrics["translation"]
        
    async def run_simplification_test(self, concurrency: int, num_requests: int):
        """
        Run load test for simplification functionality.
        
        Args:
            concurrency: Number of concurrent requests
            num_requests: Total number of requests to make
        """
        logger.info(f"Starting simplification load test with concurrency={concurrency}, requests={num_requests}")
        
        start_time = time.time()
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        # Create and gather simplification tasks
        tasks = []
        for i in range(num_requests):
            tasks.append(self._run_simplification_request(i, semaphore))
            
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        # Calculate metrics
        total_time = time.time() - start_time
        self.metrics["simplification"]["throughput"] = self.metrics["simplification"]["successes"] / total_time
        
        # Log results
        self._log_test_results("simplification", total_time)
        
        return self.metrics["simplification"]
    
    async def _run_translation_request(self, request_id: int, semaphore: asyncio.Semaphore):
        """
        Run a single translation request.
        
        Args:
            request_id: Request identifier
            semaphore: Semaphore to limit concurrency
        """
        # Select random language pair
        source_lang = random.choice(list(SAMPLE_TEXTS.keys()))
        target_langs = [lang for lang in SAMPLE_TEXTS.keys() if lang != source_lang]
        target_lang = random.choice(target_langs)
        
        # Select random text
        text = random.choice(SAMPLE_TEXTS[source_lang])
        
        # Record memory usage before request
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        async with semaphore:
            self.metrics["translation"]["requests"] += 1
            
            try:
                # Create translation request
                from app.api.schemas.translation import TranslationRequest
                
                request = TranslationRequest(
                    text=text,
                    source_language=source_lang,
                    target_language=target_lang,
                    preserve_formatting=True
                )
                
                # Perform translation with timing
                start_time = time.time()
                result = await self.translator.translate(request)
                latency = time.time() - start_time
                
                # Record success
                self.metrics["translation"]["successes"] += 1
                self.metrics["translation"]["latencies"].append(latency)
                
                # Record memory usage after request
                memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
                self.metrics["translation"]["memory_usage"].append(memory_after - memory_before)
                
                logger.debug(f"Translation {request_id} completed in {latency:.4f}s: {source_lang} → {target_lang}")
                
            except Exception as e:
                # Record failure
                self.metrics["translation"]["failures"] += 1
                logger.error(f"Translation {request_id} failed: {str(e)}")
    
    async def _run_simplification_request(self, request_id: int, semaphore: asyncio.Semaphore):
        """
        Run a single simplification request.
        
        Args:
            request_id: Request identifier
            semaphore: Semaphore to limit concurrency
        """
        # Select random language
        language = random.choice(list(SAMPLE_TEXTS.keys()))
        
        # Select random text
        text = random.choice(SAMPLE_TEXTS[language])
        
        # Select random simplification level (1-5)
        level = random.randint(1, 5)
        
        # Select random domain (if applicable)
        domain = random.choice(DOMAINS) if random.random() > 0.5 else None
        
        # Record memory usage before request
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        async with semaphore:
            self.metrics["simplification"]["requests"] += 1
            
            try:
                # Prepare options
                options = {
                    "domain": domain,
                    "preserve_formatting": True
                }
                
                # Perform simplification with timing
                start_time = time.time()
                result = await self.simplifier.simplify(
                    text=text,
                    language=language,
                    level=level,
                    options=options
                )
                latency = time.time() - start_time
                
                # Record success
                self.metrics["simplification"]["successes"] += 1
                self.metrics["simplification"]["latencies"].append(latency)
                
                # Record memory usage after request
                memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
                self.metrics["simplification"]["memory_usage"].append(memory_after - memory_before)
                
                domain_str = f", domain={domain}" if domain else ""
                logger.debug(f"Simplification {request_id} completed in {latency:.4f}s: {language}, level={level}{domain_str}")
                
            except Exception as e:
                # Record failure
                self.metrics["simplification"]["failures"] += 1
                logger.error(f"Simplification {request_id} failed: {str(e)}")
    
    def _log_test_results(self, test_type: str, total_time: float):
        """
        Log test results.
        
        Args:
            test_type: Type of test ('translation' or 'simplification')
            total_time: Total test execution time
        """
        metrics = self.metrics[test_type]
        
        # Calculate statistics
        avg_latency = statistics.mean(metrics["latencies"]) if metrics["latencies"] else 0
        p95_latency = self._percentile(metrics["latencies"], 95) if metrics["latencies"] else 0
        max_latency = max(metrics["latencies"]) if metrics["latencies"] else 0
        avg_memory = statistics.mean(metrics["memory_usage"]) if metrics["memory_usage"] else 0
        
        # Log results
        logger.info(f"=== {test_type.upper()} LOAD TEST RESULTS ===")
        logger.info(f"Total requests: {metrics['requests']}")
        logger.info(f"Successful requests: {metrics['successes']}")
        logger.info(f"Failed requests: {metrics['failures']}")
        logger.info(f"Success rate: {metrics['successes'] / max(1, metrics['requests']) * 100:.2f}%")
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Throughput: {metrics['throughput']:.2f} requests/second")
        logger.info(f"Average latency: {avg_latency:.4f} seconds")
        logger.info(f"P95 latency: {p95_latency:.4f} seconds")
        logger.info(f"Max latency: {max_latency:.4f} seconds")
        logger.info(f"Average memory usage per request: {avg_memory:.2f} MB")
        logger.info("=" * 40)
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """
        Calculate percentile value from a list of values.
        
        Args:
            data: List of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        if not data:
            return 0.0
            
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        
        # Interpolate between two values if needed
        floor_index = int(index)
        ceil_index = min(floor_index + 1, len(sorted_data) - 1)
        
        if floor_index == ceil_index:
            return sorted_data[floor_index]
            
        floor_value = sorted_data[floor_index]
        ceil_value = sorted_data[ceil_index]
        
        # Linear interpolation
        fraction = index - floor_index
        return floor_value + (ceil_value - floor_value) * fraction
    
    def save_metrics_to_file(self, filename: str = "load_test_results.json"):
        """
        Save metrics to a JSON file.
        
        Args:
            filename: Output filename
        """
        # Prepare metrics for serialization
        serializable_metrics = {
            "timestamp": datetime.now().isoformat(),
            "translation": {
                "requests": self.metrics["translation"]["requests"],
                "successes": self.metrics["translation"]["successes"],
                "failures": self.metrics["translation"]["failures"],
                "throughput": self.metrics["translation"]["throughput"],
                "avg_latency": statistics.mean(self.metrics["translation"]["latencies"]) if self.metrics["translation"]["latencies"] else 0,
                "p95_latency": self._percentile(self.metrics["translation"]["latencies"], 95) if self.metrics["translation"]["latencies"] else 0,
                "max_latency": max(self.metrics["translation"]["latencies"]) if self.metrics["translation"]["latencies"] else 0,
                "avg_memory_usage": statistics.mean(self.metrics["translation"]["memory_usage"]) if self.metrics["translation"]["memory_usage"] else 0
            },
            "simplification": {
                "requests": self.metrics["simplification"]["requests"],
                "successes": self.metrics["simplification"]["successes"],
                "failures": self.metrics["simplification"]["failures"],
                "throughput": self.metrics["simplification"]["throughput"],
                "avg_latency": statistics.mean(self.metrics["simplification"]["latencies"]) if self.metrics["simplification"]["latencies"] else 0,
                "p95_latency": self._percentile(self.metrics["simplification"]["latencies"], 95) if self.metrics["simplification"]["latencies"] else 0,
                "max_latency": max(self.metrics["simplification"]["latencies"]) if self.metrics["simplification"]["latencies"] else 0,
                "avg_memory_usage": statistics.mean(self.metrics["simplification"]["memory_usage"]) if self.metrics["simplification"]["memory_usage"] else 0
            }
        }
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
            
        logger.info(f"Metrics saved to {filename}")

async def main():
    """Main entry point for load tester."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load tester for CasaLingua language models")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent requests")
    parser.add_argument("--translation-requests", type=int, default=20, help="Number of translation requests")
    parser.add_argument("--simplification-requests", type=int, default=20, help="Number of simplification requests")
    parser.add_argument("--output", type=str, default="load_test_results.json", help="Output file for metrics")
    args = parser.parse_args()
    
    # Initialize load tester
    load_tester = LoadTester()
    await load_tester.initialize()
    
    # Run translation test
    await load_tester.run_translation_test(args.concurrency, args.translation_requests)
    
    # Run simplification test
    await load_tester.run_simplification_test(args.concurrency, args.simplification_requests)
    
    # Save metrics to file
    load_tester.save_metrics_to_file(args.output)

if __name__ == "__main__":
    asyncio.run(main())