#!/usr/bin/env python3
"""
Analyze Load Test Results for Enhanced Language Models

This script analyzes the load test results from load_test_enhanced_models.py,
generates summary statistics, and creates visualizations to help understand
the performance characteristics of the enhanced language models.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

def load_test_results(directory: str) -> List[Dict]:
    """
    Load test results from JSON files in the specified directory.
    
    Args:
        directory: Directory containing JSON result files
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Load results from each file
    for filename in os.listdir(directory):
        if filename.endswith(".json") and "results" in filename:
            filepath = os.path.join(directory, filename)
            
            try:
                with open(filepath, "r") as f:
                    result = json.load(f)
                    result["test_name"] = os.path.splitext(filename)[0].replace("_results", "")
                    results.append(result)
            except Exception as e:
                print(f"Error loading {filepath}: {str(e)}")
                
    return results

def create_performance_dataframe(results: List[Dict]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from the test results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        DataFrame with performance metrics
    """
    # Extract metrics for each test
    rows = []
    
    for result in results:
        # Translation metrics
        rows.append({
            "test_name": result["test_name"],
            "operation": "translation",
            "requests": result["translation"]["requests"],
            "successes": result["translation"]["successes"],
            "failures": result["translation"]["failures"],
            "success_rate": result["translation"]["successes"] / max(1, result["translation"]["requests"]) * 100,
            "throughput": result["translation"]["throughput"],
            "avg_latency": result["translation"]["avg_latency"],
            "p95_latency": result["translation"]["p95_latency"],
            "avg_memory_usage": result["translation"]["avg_memory_usage"]
        })
        
        # Simplification metrics
        rows.append({
            "test_name": result["test_name"],
            "operation": "simplification",
            "requests": result["simplification"]["requests"],
            "successes": result["simplification"]["successes"],
            "failures": result["simplification"]["failures"],
            "success_rate": result["simplification"]["successes"] / max(1, result["simplification"]["requests"]) * 100,
            "throughput": result["simplification"]["throughput"],
            "avg_latency": result["simplification"]["avg_latency"],
            "p95_latency": result["simplification"]["p95_latency"],
            "avg_memory_usage": result["simplification"]["avg_memory_usage"]
        })
        
    return pd.DataFrame(rows)

def plot_throughput_comparison(df: pd.DataFrame, output_dir: str):
    """
    Plot throughput comparison between translation and simplification.
    
    Args:
        df: DataFrame with performance metrics
        output_dir: Directory to save the plot
    """
    # Create throughput comparison plot
    plt.figure(figsize=(12, 6))
    
    # Pivot the data for plotting
    throughput_df = df.pivot(index="test_name", columns="operation", values="throughput")
    throughput_df.plot(kind="bar", width=0.7)
    
    plt.title("Throughput Comparison: Translation vs. Simplification")
    plt.xlabel("Test Scenario")
    plt.ylabel("Throughput (requests/second)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Operation")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "throughput_comparison.png"))
    
def plot_latency_comparison(df: pd.DataFrame, output_dir: str):
    """
    Plot latency comparison between translation and simplification.
    
    Args:
        df: DataFrame with performance metrics
        output_dir: Directory to save the plot
    """
    # Create latency comparison plot
    plt.figure(figsize=(12, 6))
    
    # Pivot the data for plotting
    latency_df = df.pivot(index="test_name", columns="operation", values="avg_latency")
    latency_df.plot(kind="bar", width=0.7)
    
    plt.title("Average Latency Comparison: Translation vs. Simplification")
    plt.xlabel("Test Scenario")
    plt.ylabel("Average Latency (seconds)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Operation")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "latency_comparison.png"))
    
def plot_memory_usage(df: pd.DataFrame, output_dir: str):
    """
    Plot memory usage comparison between translation and simplification.
    
    Args:
        df: DataFrame with performance metrics
        output_dir: Directory to save the plot
    """
    # Create memory usage plot
    plt.figure(figsize=(12, 6))
    
    # Pivot the data for plotting
    memory_df = df.pivot(index="test_name", columns="operation", values="avg_memory_usage")
    memory_df.plot(kind="bar", width=0.7)
    
    plt.title("Memory Usage Comparison: Translation vs. Simplification")
    plt.xlabel("Test Scenario")
    plt.ylabel("Average Memory Usage (MB)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Operation")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "memory_usage_comparison.png"))
    
def generate_analysis_report(df: pd.DataFrame, output_dir: str):
    """
    Generate a detailed performance analysis report.
    
    Args:
        df: DataFrame with performance metrics
        output_dir: Directory to save the report
    """
    # Group data by operation and calculate average metrics
    operation_summary = df.groupby("operation").agg({
        "throughput": ["mean", "min", "max"],
        "avg_latency": ["mean", "min", "max"],
        "p95_latency": ["mean", "min", "max"],
        "avg_memory_usage": ["mean", "min", "max"],
        "success_rate": ["mean", "min"]
    })
    
    # Find best performing test scenario for each operation
    best_throughput = df.loc[df.groupby("operation")["throughput"].idxmax()]
    best_latency = df.loc[df.groupby("operation")["avg_latency"].idxmin()]
    
    # Generate report
    with open(os.path.join(output_dir, "performance_analysis.md"), "w") as f:
        f.write("# Performance Analysis for Enhanced Language Models\n\n")
        f.write(f"Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overall Performance Summary\n\n")
        f.write("### Translation Operation\n")
        f.write(f"- Average throughput: {operation_summary.loc['translation', ('throughput', 'mean')]:.2f} requests/second\n")
        f.write(f"- Average latency: {operation_summary.loc['translation', ('avg_latency', 'mean')]:.4f} seconds\n")
        f.write(f"- Average P95 latency: {operation_summary.loc['translation', ('p95_latency', 'mean')]:.4f} seconds\n")
        f.write(f"- Average memory usage: {operation_summary.loc['translation', ('avg_memory_usage', 'mean')]:.2f} MB\n")
        f.write(f"- Average success rate: {operation_summary.loc['translation', ('success_rate', 'mean')]:.2f}%\n\n")
        
        f.write("### Simplification Operation\n")
        f.write(f"- Average throughput: {operation_summary.loc['simplification', ('throughput', 'mean')]:.2f} requests/second\n")
        f.write(f"- Average latency: {operation_summary.loc['simplification', ('avg_latency', 'mean')]:.4f} seconds\n")
        f.write(f"- Average P95 latency: {operation_summary.loc['simplification', ('p95_latency', 'mean')]:.4f} seconds\n")
        f.write(f"- Average memory usage: {operation_summary.loc['simplification', ('avg_memory_usage', 'mean')]:.2f} MB\n")
        f.write(f"- Average success rate: {operation_summary.loc['simplification', ('success_rate', 'mean')]:.2f}%\n\n")
        
        f.write("## Best Performing Test Scenarios\n\n")
        f.write("### Best Throughput\n")
        f.write(f"- Translation: {best_throughput[best_throughput['operation']=='translation']['test_name'].values[0]} " +
                f"({best_throughput[best_throughput['operation']=='translation']['throughput'].values[0]:.2f} requests/second)\n")
        f.write(f"- Simplification: {best_throughput[best_throughput['operation']=='simplification']['test_name'].values[0]} " +
                f"({best_throughput[best_throughput['operation']=='simplification']['throughput'].values[0]:.2f} requests/second)\n\n")
        
        f.write("### Best Latency\n")
        f.write(f"- Translation: {best_latency[best_latency['operation']=='translation']['test_name'].values[0]} " +
                f"({best_latency[best_latency['operation']=='translation']['avg_latency'].values[0]:.4f} seconds)\n")
        f.write(f"- Simplification: {best_latency[best_latency['operation']=='simplification']['test_name'].values[0]} " +
                f"({best_latency[best_latency['operation']=='simplification']['avg_latency'].values[0]:.4f} seconds)\n\n")
        
        f.write("## Performance Comparison\n\n")
        f.write("### Translation vs. Simplification\n")
        trans_throughput = operation_summary.loc['translation', ('throughput', 'mean')]
        simp_throughput = operation_summary.loc['simplification', ('throughput', 'mean')]
        throughput_ratio = trans_throughput / simp_throughput if simp_throughput > 0 else 0
        
        trans_latency = operation_summary.loc['translation', ('avg_latency', 'mean')]
        simp_latency = operation_summary.loc['simplification', ('avg_latency', 'mean')]
        latency_ratio = trans_latency / simp_latency if simp_latency > 0 else 0
        
        f.write(f"- Throughput ratio (translation/simplification): {throughput_ratio:.2f}\n")
        f.write(f"- Latency ratio (translation/simplification): {latency_ratio:.2f}\n\n")
        
        if throughput_ratio > 1:
            f.write(f"Translation has {throughput_ratio:.2f}x higher throughput than simplification.\n")
        else:
            f.write(f"Simplification has {1/throughput_ratio:.2f}x higher throughput than translation.\n")
            
        if latency_ratio > 1:
            f.write(f"Translation has {latency_ratio:.2f}x higher latency than simplification.\n\n")
        else:
            f.write(f"Simplification has {1/latency_ratio:.2f}x higher latency than translation.\n\n")
            
        f.write("## Performance Visualization\n\n")
        f.write("Graphical visualizations have been generated for:\n")
        f.write("1. Throughput comparison (throughput_comparison.png)\n")
        f.write("2. Latency comparison (latency_comparison.png)\n")
        f.write("3. Memory usage comparison (memory_usage_comparison.png)\n\n")
        
        f.write("## Detailed Test Results\n\n")
        f.write("For a detailed breakdown of all test results, refer to the individual JSON files in the results directory.\n")
        
def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze load test results for enhanced language models")
    parser.add_argument("--results-dir", type=str, default="load_test_results", help="Directory containing the test results")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the analysis output")
    args = parser.parse_args()
    
    # Use the results directory as the output directory if not specified
    output_dir = args.output_dir or args.results_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test results
    results = load_test_results(args.results_dir)
    
    if not results:
        print(f"No test results found in {args.results_dir}")
        return
        
    print(f"Loaded {len(results)} test results")
    
    # Create performance DataFrame
    df = create_performance_dataframe(results)
    
    # Generate plots
    plot_throughput_comparison(df, output_dir)
    plot_latency_comparison(df, output_dir)
    plot_memory_usage(df, output_dir)
    
    # Generate analysis report
    generate_analysis_report(df, output_dir)
    
    # Save the DataFrame to CSV for further analysis
    df.to_csv(os.path.join(output_dir, "performance_metrics.csv"), index=False)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    
if __name__ == "__main__":
    main()