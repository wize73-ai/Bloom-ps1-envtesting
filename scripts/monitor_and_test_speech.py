#!/usr/bin/env python3
"""
Monitor server logs and test speech endpoints for CasaLingua.
This script helps identify and fix issues with TTS and STT functionality.
"""

import os
import sys
import time
import json
import subprocess
import argparse
import signal
import threading
from pathlib import Path

def run_command(command, background=False):
    """Run a command and optionally capture its output."""
    if background:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        return process
    else:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        return result.stdout

def monitor_logs(log_file, stop_event):
    """Monitor a log file and print new content."""
    if not log_file or not os.path.exists(os.path.dirname(log_file)):
        print(f"Log directory for {log_file} doesn't exist. Creating empty log file.")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w') as f:
            pass  # Create empty file

    # Get the current size of the log file
    try:
        current_size = os.path.getsize(log_file)
    except FileNotFoundError:
        # Create the file if it doesn't exist
        with open(log_file, 'w') as f:
            pass
        current_size = 0

    print(f"Monitoring log file: {log_file}")
    
    while not stop_event.is_set():
        try:
            # Check if the file exists and has new content
            if os.path.exists(log_file):
                new_size = os.path.getsize(log_file)
                if new_size > current_size:
                    # Read and print only the new content
                    with open(log_file, 'r') as f:
                        f.seek(current_size)
                        new_content = f.read()
                        if new_content:
                            print(f"\n--- NEW LOG CONTENT ---\n{new_content}", end='')
                    current_size = new_size
        except Exception as e:
            print(f"Error monitoring log: {str(e)}")
        
        # Sleep briefly before checking again
        time.sleep(0.1)

def restart_server():
    """Restart the server."""
    print("Stopping server if running...")
    run_command("pkill -f 'python -m uvicorn app.main:app'")
    time.sleep(2)
    
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Start the server in the background
    print("Starting server...")
    server_process = run_command(
        "python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > logs/server.log 2>&1",
        background=True
    )
    
    # Sleep to allow server to start
    print("Waiting for server to start...")
    time.sleep(5)
    
    return server_process

def run_tests():
    """Run the speech endpoint tests."""
    print("\n--- RUNNING SPEECH ENDPOINT TESTS ---\n")
    test_output = run_command("python scripts/test_speech_endpoints_fixed.py")
    print(test_output)

def main():
    parser = argparse.ArgumentParser(description='Monitor and test CasaLingua speech endpoints')
    parser.add_argument('--no-restart', action='store_true', help='Do not restart the server')
    parser.add_argument('--no-monitor', action='store_true', help='Do not monitor logs')
    parser.add_argument('--log-file', default='logs/server.log', help='Log file to monitor')
    args = parser.parse_args()
    
    server_process = None
    
    # Set up signal handling for graceful exit
    stop_event = threading.Event()
    
    def signal_handler(sig, frame):
        print("\nStopping monitoring and shutting down...")
        stop_event.set()
        if server_process:
            server_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Restart server if needed
        if not args.no_restart:
            server_process = restart_server()
        
        # Start log monitoring in a separate thread
        if not args.no_monitor:
            monitor_thread = threading.Thread(
                target=monitor_logs,
                args=(args.log_file, stop_event)
            )
            monitor_thread.daemon = True
            monitor_thread.start()
        
        # Run the tests
        run_tests()
        
        if not args.no_monitor:
            print("\nLog monitoring is active. Press Ctrl+C to exit.")
            while not stop_event.is_set():
                time.sleep(1)
        
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        stop_event.set()
        if server_process:
            server_process.terminate()

if __name__ == "__main__":
    main()