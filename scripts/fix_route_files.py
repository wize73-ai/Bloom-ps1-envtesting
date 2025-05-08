#!/usr/bin/env python3
"""
Script to identify and comment out non-working endpoints in route files.
"""
import os
import glob
import re
from typing import List, Dict, Tuple

# Base directory for the API routes
ROUTES_DIR = "/Users/jameswilson/Desktop/PRODUCTION/test/casMay4/app/api/routes"

# List of endpoints that we want to comment out
# Format: (method, path)
ENDPOINTS_TO_COMMENT = [
    # RAG endpoints
    ("POST", "/translate"),
    ("POST", "/query"),
    ("POST", "/chat"),
    
    # Admin endpoints
    ("GET", "/system/info"),
    ("GET", "/models"),
    ("GET", "/languages"),
    ("GET", "/metrics"),
]

def find_endpoint_in_file(file_path: str, method: str, endpoint: str) -> List[Tuple[int, str, str]]:
    """
    Find an endpoint definition in a file.
    
    Args:
        file_path: Path to the file to search
        method: HTTP method (GET, POST, etc.)
        endpoint: The endpoint path
    
    Returns:
        List of tuples (line_number, line, indentation)
    """
    matches = []
    method_lower = method.lower()
    
    # Normalize endpoint for matching
    if not endpoint.startswith('/'):
        endpoint = '/' + endpoint
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        # Match the router.method decorator line
        pattern = rf'@router\.{method_lower}\s*\(\s*[\'"]({endpoint}|{endpoint[1:]})[\'"]'
        if re.search(pattern, line):
            # Get indentation
            indent_match = re.match(r'^(\s*)', line)
            indentation = indent_match.group(1) if indent_match else ''
            matches.append((i, line, indentation))
    
    return matches

def comment_out_endpoint(file_path: str, line_number: int, indentation: str) -> bool:
    """
    Comment out an endpoint in a file.
    
    Args:
        file_path: Path to the file
        line_number: Line number of the router decorator
        indentation: Indentation to use for the commented sections
    
    Returns:
        True if successful, False otherwise
    """
    with open(file_path, 'r') as f:
        lines = list(f.readlines())
    
    # Comment out the decorator line
    lines[line_number] = indentation + '# ' + lines[line_number].lstrip()
    
    # Find the function definition line (should be the next line)
    if line_number + 1 < len(lines) and 'async def' in lines[line_number + 1]:
        func_line = line_number + 1
        lines[func_line] = indentation + '# ' + lines[func_line].lstrip()
        
        # Comment out the function body
        current_line = func_line + 1
        while current_line < len(lines):
            line = lines[current_line]
            # Check if this line has more indentation than the function definition
            current_indent = re.match(r'^(\s*)', line).group(1)
            if len(current_indent) <= len(indentation):
                # End of function body
                break
            
            # Comment out this line of the function body
            lines[current_line] = indentation + '# ' + line.lstrip()
            current_line += 1
    
    # Write the modified file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    return True

def find_and_comment_endpoints():
    """
    Find and comment out specified endpoints in route files.
    """
    py_files = glob.glob(os.path.join(ROUTES_DIR, "*.py"))
    
    commented_count = 0
    for endpoint_method, endpoint_path in ENDPOINTS_TO_COMMENT:
        found = False
        for py_file in py_files:
            matches = find_endpoint_in_file(py_file, endpoint_method, endpoint_path)
            if matches:
                found = True
                for line_num, line, indent in matches:
                    file_name = os.path.basename(py_file)
                    print(f"Found {endpoint_method} {endpoint_path} in {file_name}:{line_num+1}")
                    success = comment_out_endpoint(py_file, line_num, indent)
                    if success:
                        commented_count += 1
                        print(f"Commented out {endpoint_method} {endpoint_path} in {file_name}")
                    else:
                        print(f"Failed to comment out {endpoint_method} {endpoint_path} in {file_name}")
        
        if not found:
            print(f"WARNING: Could not find {endpoint_method} {endpoint_path} in any route file")
    
    print(f"\nCommented out {commented_count} endpoints")

if __name__ == "__main__":
    find_and_comment_endpoints()