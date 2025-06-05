import json


def extract_json_from_text(content):
    # Simple approach: find the first { and last } in the content
    start_idx = content.find("{")
    if start_idx == -1:
        return {}  # No JSON found

    end_idx = content.rfind("}")
    if end_idx == -1 or end_idx < start_idx:
        return {}  # No closing brace or invalid structure

    # Extract the potential JSON string
    json_str = content[start_idx : end_idx + 1]

    try:
        # Attempt to parse it
        json_obj = json.loads(json_str)
        return json_obj
    except json.JSONDecodeError:
        # If direct extraction fails, try a more robust approach
        try:
            # Replace newlines with actual newlines (in case they're escaped)
            cleaned = json_str.replace("\\n", "\n")
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Last resort: try to find any JSON-like string with balanced braces
        brace_level = 0
        json_start = None

        for i, char in enumerate(content):
            if char == "{":
                if brace_level == 0:
                    json_start = i
                brace_level += 1
            elif char == "}":
                brace_level -= 1
                if brace_level == 0 and json_start is not None:
                    # Found a balanced JSON-like string
                    try:
                        potential_json = content[json_start : i + 1]
                        return json.loads(potential_json)
                    except json.JSONDecodeError:
                        # Continue looking for other balanced braces
                        pass

        # Nothing worked
        return {}
