#!/usr/bin/env python3
"""
Fix corrupted JSON files by removing extra characters at the end
Specifically for dist_X.X_result.json files that may have been corrupted by concurrent writes
"""
import os
import json
import re
from pathlib import Path
from tqdm import tqdm

def find_valid_json_end(content):
    """
    Find the position where a valid JSON object ends
    Returns the position of the last '}' that completes a valid JSON object
    """
    # Try to find the matching closing brace
    depth = 0
    last_valid_pos = -1
    
    for i, char in enumerate(content):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                last_valid_pos = i
            elif depth < 0:
                # Unmatched closing brace
                break
    
    return last_valid_pos

def fix_json_file(file_path):
    """
    Try to fix a corrupted JSON file by removing extra characters at the end
    Returns: (fixed, error_message)
    """
    try:
        # First, try to parse as-is
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            json.loads(content)
            return (False, None)  # File is already valid
        except json.JSONDecodeError as e:
            # File is corrupted, try to fix it
            error_msg = str(e)
            
            # Try to find where the valid JSON ends
            valid_end = find_valid_json_end(content)
            
            if valid_end > 0:
                # Extract valid JSON part
                valid_content = content[:valid_end + 1]
                
                # Try to parse the valid part
                try:
                    json.loads(valid_content)
                    # Valid JSON found, write it back
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(valid_content)
                    return (True, f"Removed {len(content) - len(valid_content)} characters from end")
                except json.JSONDecodeError:
                    pass
            
            # If that didn't work, try removing common patterns from the end
            # Common corruption patterns: "false} false}", "alse}", "se}", etc.
            patterns = [
                r'\s+false\s*\}\s*$',  # " false}"
                r'\s+alse\s*\}\s*$',   # " alse}"
                r'\s+se\s*\}\s*$',     # " se}"
                r'\s+e\s*\}\s*$',      # " e}"
                r'\s+\}\s*$',          # Extra "}"
            ]
            
            for pattern in patterns:
                cleaned = re.sub(pattern, '', content)
                if cleaned != content:
                    try:
                        json.loads(cleaned)
                        # Valid JSON found, write it back
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(cleaned)
                        removed = len(content) - len(cleaned)
                        return (True, f"Removed {removed} characters using pattern '{pattern}'")
                    except json.JSONDecodeError:
                        continue
            
            return (False, f"Could not fix: {error_msg}")
            
    except Exception as e:
        return (False, f"Error reading file: {str(e)}")

def main():
    import sys
    
    # Allow directory to be specified as command line argument
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = '/data1/xzl/ManipLLM/test/test_results/distance_eval_results_test_v1'
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        print(f"Usage: python {sys.argv[0]} [directory_path]")
        return
    
    # Find all dist_X.X_result.json files
    pattern = 'dist_*_result.json'
    json_files = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith('dist_') and file.endswith('_result.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} dist_X.X_result.json files")
    print(f"Checking and fixing corrupted files...\n")
    
    fixed_count = 0
    already_valid_count = 0
    failed_count = 0
    
    for file_path in tqdm(json_files, desc="Processing files"):
        fixed, error = fix_json_file(file_path)
        
        if fixed:
            fixed_count += 1
            if error:
                print(f"Fixed: {file_path}")
                print(f"  {error}")
        elif error:
            failed_count += 1
            print(f"Failed: {file_path}")
            print(f"  {error}")
        else:
            already_valid_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files: {len(json_files)}")
    print(f"  Already valid: {already_valid_count}")
    print(f"  Fixed: {fixed_count}")
    print(f"  Failed to fix: {failed_count}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

