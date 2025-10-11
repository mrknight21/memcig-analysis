import json
import string
import random

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def parse_json_markdown(llm_returned_string):
    # 1. Remove the markdown code block delimiters
    # This checks if the string starts and ends with the markdown delimiters
    if llm_returned_string.startswith("```json\n") and llm_returned_string.endswith("```"):
        json_string = llm_returned_string[len("```json\n"):-len("```")]
    else:
        # If it's not wrapped, assume it's pure JSON.
        # You might want more robust error handling here for real-world scenarios.
        json_string = llm_returned_string

    # 2. Parse the JSON string into a Python list of dictionaries
    try:
        parsed_data = json.loads(json_string)
        return parsed_data

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Problematic string portion: {json_string[e.pos - 20:e.pos + 20]}")  # Show context around the error
        return None
    except KeyError as e:
        print(f"Missing expected key in JSON data: {e}")
        return None

def generate_session_token(length=16):
    """Generate a random session token with only alphanumeric characters."""
    allowed_chars = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
    return ''.join(random.choices(allowed_chars, k=length))



