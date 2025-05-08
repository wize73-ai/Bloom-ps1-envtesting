#!/usr/bin/env python3
"""
Script to comment out problematic endpoints in the CasaLingua API.
"""
import os
import re
import shutil

# Base directory
BASE_DIR = "/Users/jameswilson/Desktop/PRODUCTION/test/casMay4"
API_ROUTES_DIR = os.path.join(BASE_DIR, "app/api/routes")

# Endpoints to comment out with their files
# Format: (file_path, method, endpoint_path)
ENDPOINTS_TO_COMMENT = [
    # RAG endpoints in rag.py
    (os.path.join(API_ROUTES_DIR, "rag.py"), "post", "/translate"),
    (os.path.join(API_ROUTES_DIR, "rag.py"), "post", "/query"),
    (os.path.join(API_ROUTES_DIR, "rag.py"), "post", "/chat"),
    
    # Admin endpoints in admin.py
    (os.path.join(API_ROUTES_DIR, "admin.py"), "get", "/system/info"),
    (os.path.join(API_ROUTES_DIR, "admin.py"), "get", "/models"),
    (os.path.join(API_ROUTES_DIR, "admin.py"), "get", "/languages"),
    
    # Metrics endpoint in metrics.py
    (os.path.join(API_ROUTES_DIR, "metrics.py"), "get", "/metrics"),
]

def comment_out_endpoint(file_path, method, endpoint_path):
    """Comment out a specific endpoint in a file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    # Create a backup of the file
    backup_path = file_path + ".bak"
    shutil.copy2(file_path, backup_path)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Clean the endpoint path for regex matching
    endpoint_regex = endpoint_path.replace('/', '\\/')
    
    # Find the endpoint decorator
    decorator_pattern = rf'@router\.{method}\s*\(\s*[\'"]({endpoint_regex}|{endpoint_regex[1:]})[\'"]'
    
    # Track whether we found the endpoint
    found = False
    
    # Track which lines to comment
    commenting = False
    comment_start_line = -1
    comment_end_line = -1
    indent = ""
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line contains the decorator
        if re.search(decorator_pattern, line):
            found = True
            commenting = True
            comment_start_line = i
            
            # Get the indentation
            indent_match = re.match(r'^(\s*)', line)
            indent = indent_match.group(1) if indent_match else ''
            
            # Comment out the decorator line
            lines[i] = f"{indent}# {line[len(indent):]}"
        
        # If we're in commenting mode, check if this is the function definition
        elif commenting and i == comment_start_line + 1 and 'async def' in line:
            # Comment out the function definition
            lines[i] = f"{indent}# {line[len(indent):]}"
        
        # If we're in commenting mode, check if this is part of the function body
        elif commenting and i > comment_start_line + 1:
            # Check the indentation level to see if we're still in the function
            current_indent = re.match(r'^(\s*)', line).group(1)
            
            if len(current_indent) > len(indent) or (len(current_indent) == len(indent) and line.strip() == ''):
                # Still in the function body, comment it out
                lines[i] = f"{indent}# {line[len(indent):]}"
            else:
                # We've reached the end of the function
                commenting = False
                comment_end_line = i - 1
        
        i += 1
    
    if found:
        # Write the modified file
        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        print(f"Commented out {method.upper()} {endpoint_path} in {os.path.basename(file_path)}")
        return True
    else:
        # Remove the backup if we didn't find the endpoint
        os.remove(backup_path)
        print(f"Endpoint {method.upper()} {endpoint_path} not found in {os.path.basename(file_path)}")
        return False

def main():
    """Main function to comment out problematic endpoints."""
    print("Starting to comment out problematic endpoints...")
    
    # Count of endpoints commented out
    commented_count = 0
    
    for file_path, method, endpoint_path in ENDPOINTS_TO_COMMENT:
        if comment_out_endpoint(file_path, method, endpoint_path):
            commented_count += 1
    
    print(f"\nFinished: commented out {commented_count} endpoints")

if __name__ == "__main__":
    main()