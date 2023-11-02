import sys
import re

# Define the patterns to search for
PATTERNS = [r"\d{2}-\d{8}A\d", r"\d{2}-\d{6}-\d{2}-\d"]

def search_patterns_in_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            content = content.replace(" ", "")

        for pattern in PATTERNS:
            matches = re.findall(pattern, content)
            if matches:
                print(f"Pattern: {pattern} - Matches: {matches}")
            else:
                print(f"Pattern: {pattern} - No matches found")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        file_path = sys.argv[1]
        search_patterns_in_file(file_path)
