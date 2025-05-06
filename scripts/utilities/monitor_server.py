#!/usr/bin/env python3
"""
CasaLingua Server Monitor

This script monitors a running CasaLingua server by watching logs
and checking memory usage. It runs alongside the server to identify
any issues during operation.
"""

import os
import sys
import time
import subprocess
import psutil
import datetime

def find_server_process():
    """Find the Python process running the CasaLingua server"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' and any('main.py' in cmd for cmd in proc.info['cmdline'] if cmd):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

def monitor_process(pid, duration=60):
    """Monitor a process for a specific duration"""
    print(f"Monitoring CasaLingua server (PID: {pid}) for {duration} seconds")
    
    start_time = time.time()
    end_time = start_time + duration
    
    # Initial stats
    process = psutil.Process(pid)
    
    # Header
    print(f"\n{'-' * 80}")
    print(f"{'Time':12} | {'CPU %':7} | {'Memory':10} | {'Threads':8} | {'Status':8}")
    print(f"{'-' * 80}")
    
    try:
        while time.time() < end_time:
            # Get current time
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            
            try:
                # Check if process is still running
                if not process.is_running():
                    print(f"{current_time:12} | {'N/A':7} | {'N/A':10} | {'N/A':8} | STOPPED")
                    print("\nProcess has stopped!")
                    return
                
                # Get process information
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)
                
                # Get CPU usage
                cpu_percent = process.cpu_percent(interval=1)
                
                # Get thread count
                thread_count = process.num_threads()
                
                # Print status
                print(f"{current_time:12} | {cpu_percent:7.1f} | {mem_mb:7.1f} MB | {thread_count:8} | RUNNING")
                
                # Sleep for a bit
                time.sleep(1)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                print(f"{current_time:12} | {'N/A':7} | {'N/A':10} | {'N/A':8} | ERROR")
                print(f"\nError monitoring process: {str(e)}")
                return
            
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    
    print(f"\nMonitoring completed after {int(time.time() - start_time)} seconds")

def main():
    """Main function"""
    print("CasaLingua Server Monitor")
    
    # Check for server process
    server_process = find_server_process()
    
    if not server_process:
        print("Error: CasaLingua server process not found")
        print("Please make sure the server is running (python -m app.main)")
        return 1
    
    print(f"Found CasaLingua server process (PID: {server_process.pid})")
    
    # Monitor the server
    monitor_process(server_process.pid, duration=120)  # Monitor for 2 minutes
    
    return 0

if __name__ == "__main__":
    sys.exit(main())