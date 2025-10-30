"""Utility functions for Paper2Code multi-agent system.

This module provides helper functions for:
- Extracting and parsing planning information from LLM trajectories
- Converting content to structured JSON format
- Cost calculation and tracking
- File reading and processing
- Code extraction from LLM responses
"""

import json
import re
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

def extract_planning(trajectories_json_file_path: str) -> List[str]:
    """Extract planning context from trajectories JSON file.
    
    Reads the LLM conversation trajectories and extracts assistant responses,
    removing thinking tags and limiting to the first 3 planning steps.
    
    Args:
        trajectories_json_file_path: Path to the JSON file containing conversation trajectories
        
    Returns:
        List of extracted context strings from assistant responses (max 3 items)
        
    Raises:
        FileNotFoundError: If the trajectories file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(trajectories_json_file_path, 'r', encoding='utf-8') as f:
        traj = json.load(f)

    context_lst = []
    for turn in traj:
        if turn['role'] == 'assistant':
            content = turn['content']
            # Remove thinking tags if present
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            context_lst.append(content)

    # Limit to first 3 planning steps
    context_lst = context_lst[:3] 

    return context_lst



def content_to_json(data: str) -> Dict[str, Any]:
    """Convert content string to JSON with fallback parsing strategies.
    
    Attempts multiple parsing strategies to extract JSON from LLM-generated content,
    handling common formatting issues like comments and trailing commas.
    
    Args:
        data: Raw content string containing JSON data
        
    Returns:
        Parsed JSON data as a dictionary
        
    Note:
        Uses a cascade of parsing strategies (content_to_json, content_to_json2, 
        content_to_json3, content_to_json4) with increasing robustness.
    """
    # Remove [CONTENT] tags
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()

    # Remove inline comments
    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)

    # Remove trailing commas
    clean_data = re.sub(r',\s*\]', ']', clean_data)

    # Remove whitespace
    clean_data = re.sub(r'\n\s*', '', clean_data)

    # JSON parsing
    try:
        json_data = json.loads(clean_data)
        return json_data
    except json.JSONDecodeError:
        return content_to_json2(data)
        
    
def content_to_json2(data: str) -> Dict[str, Any]:
    """Second-level JSON parsing strategy with more aggressive cleaning.
    
    Args:
        data: Raw content string containing JSON data
        
    Returns:
        Parsed JSON data as a dictionary, or falls back to content_to_json3
    """
    # Remove [CONTENT][/CONTENT]
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()

    # "~~~~", #comment -> "~~~~",
    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)

    # "~~~~" #comment → "~~~~"
    clean_data = re.sub(r'(".*?")\s*#.*', r'\1', clean_data)

    # ("~~~~",] -> "~~~~"])
    clean_data = re.sub(r',\s*\]', ']', clean_data)

    clean_data = re.sub(r'\n\s*', '', clean_data)

    # JSON parsing
    try:
        json_data = json.loads(clean_data)
        return json_data
    
    except json.JSONDecodeError:
        return content_to_json3(data)

def content_to_json3(data: str) -> Dict[str, Any]:
    """Third-level JSON parsing strategy handling quote variations.
    
    Args:
        data: Raw content string containing JSON data
        
    Returns:
        Parsed JSON data as a dictionary, or falls back to content_to_json4
    """
    # remove [CONTENT] [/CONTENT]
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()

    # "~~~~", #comment -> "~~~~",
    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)

    # "~~~~" #comment → "~~~~"
    clean_data = re.sub(r'(".*?")\s*#.*', r'\1', clean_data)

    # remove ("~~~~",] -> "~~~~"])
    clean_data = re.sub(r',\s*\]', ']', clean_data)

    clean_data = re.sub(r'\n\s*', '', clean_data) 
    clean_data = re.sub(r'"""', '"', clean_data)  # Replace triple double quotes
    clean_data = re.sub(r"'''", "'", clean_data)  # Replace triple single quotes
    clean_data = re.sub(r"\\", "'", clean_data)  # Replace \ 

    # JSON parsing
    try:
        json_data = json.loads(f"""{clean_data}""")
        return json_data
    
    except json.JSONDecodeError:
        return content_to_json4(data)
    
def content_to_json4(data: str) -> Dict[str, Any]:
    """Final fallback JSON parsing strategy using regex extraction.
    
    Attempts to extract specific fields (Logic Analysis, Task list) using regex
    when standard JSON parsing fails.
    
    Args:
        data: Raw content string containing JSON data
        
    Returns:
        Dictionary with extracted fields, or empty dict if extraction fails
    """
    # Extract Logic Analysis and Task list using regex
    pattern = r'"Logic Analysis":\s*(\[[\s\S]*?\])\s*,\s*"Task list":\s*(\[[\s\S]*?\])'
    match = re.search(pattern, data)

    if match:
        try:
            logic_analysis = json.loads(match.group(1))
            task_list = json.loads(match.group(2))

            result = {
                "Logic Analysis": logic_analysis,
                "Task list": task_list
            }
        except json.JSONDecodeError:
            result = {}
    else:
        result = {}

    return result

def extract_code_from_content(content: str) -> str:
    """Extract code from markdown-formatted content.
    
    Searches for code blocks with language specifiers (e.g., ```python).
    
    Args:
        content: Text content containing code blocks
        
    Returns:
        Extracted code as a string, or empty string if no code found
    """
    pattern = r'```(?:\w+)?\s*\n(.*?)\n\s*```'
    code = re.findall(pattern, content, re.DOTALL)
    if len(code) == 0:
        return ""
    else:
        return code[0]
    
def extract_code_from_content2(content: str) -> str:
    """Alternative method to extract Python code from content.
    
    Specifically looks for ```python code blocks.
    
    Args:
        content: Text content containing Python code blocks
        
    Returns:
        Extracted Python code as a string, or empty string with warning if not found
    """
    pattern = r'```python\s*(.*?)```'
    result = re.search(pattern, content, re.DOTALL)

    if result:
        extracted_code = result.group(1).strip()
    else:
        extracted_code = ""
        print("[WARNING] No Python code found.")
    return extracted_code

def format_json_data(data: Dict[str, Any]) -> str:
    """Format JSON data as human-readable text with separators.
    
    Args:
        data: Dictionary containing structured data
        
    Returns:
        Formatted string with section headers and bullet points
    """
    formatted_text = ""
    for key, value in data.items():
        formatted_text += "-" * 40 + "\n"
        formatted_text += "[" + key + "]\n"
        if isinstance(value, list):
            for item in value:
                formatted_text += f"- {item}\n"
        else:
            formatted_text += str(value) + "\n"
        formatted_text += "\n"
    return formatted_text


def cal_cost(response_json: Dict[str, Any], model_name: str) -> Dict[str, Union[str, int, float]]:
    """Calculate API cost for a given LLM response.
    
    Computes input, cached input, and output token costs based on the model's
    pricing structure. Supports OpenAI models including GPT-4, o1, o3, etc.
    
    Args:
        response_json: Response object from the API containing usage information
        model_name: Name of the LLM model used (e.g., 'gpt-4o', 'o3-mini')
        
    Returns:
        Dictionary containing:
            - model_name: Name of the model
            - actual_input_tokens: Non-cached input tokens
            - input_cost: Cost for input tokens
            - cached_tokens: Number of cached tokens
            - cached_input_cost: Cost for cached tokens
            - output_tokens: Number of output tokens
            - output_cost: Cost for output tokens
            - total_cost: Total cost in USD
            
    Note:
        Prices are in USD per 1 million tokens. Model pricing is kept up-to-date
        with OpenAI's pricing at the time of implementation.
    """
    model_cost = {
        # gpt-4.1
        "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "gpt-4.1-2025-04-14": {"input": 2.00, "cached_input": 0.50, "output": 8.00},

        # gpt-4.1-mini
        "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
        "gpt-4.1-mini-2025-04-14": {"input": 0.40, "cached_input": 0.10, "output": 1.60},

        # gpt-4.1-nano
        "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
        "gpt-4.1-nano-2025-04-14": {"input": 0.10, "cached_input": 0.025, "output": 0.40},

        # gpt-4.5-preview
        "gpt-4.5-preview": {"input": 75.00, "cached_input": 37.50, "output": 150.00},
        "gpt-4.5-preview-2025-02-27": {"input": 75.00, "cached_input": 37.50, "output": 150.00},

        # gpt-4o
        "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
        "gpt-4o-2024-08-06": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
        "gpt-4o-2024-11-20": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
        "gpt-4o-2024-05-13": {"input": 5.00, "cached_input": None, "output": 15.00},

        # gpt-4o-audio-preview
        "gpt-4o-audio-preview": {"input": 2.50, "cached_input": None, "output": 10.00},
        "gpt-4o-audio-preview-2024-12-17": {"input": 2.50, "cached_input": None, "output": 10.00},
        "gpt-4o-audio-preview-2024-10-01": {"input": 2.50, "cached_input": None, "output": 10.00},

        # gpt-4o-realtime-preview
        "gpt-4o-realtime-preview": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
        "gpt-4o-realtime-preview-2024-12-17": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
        "gpt-4o-realtime-preview-2024-10-01": {"input": 5.00, "cached_input": 2.50, "output": 20.00},

        # gpt-4o-mini
        "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
        "gpt-4o-mini-2024-07-18": {"input": 0.15, "cached_input": 0.075, "output": 0.60},

        # gpt-4o-mini-audio-preview
        "gpt-4o-mini-audio-preview": {"input": 0.15, "cached_input": None, "output": 0.60},
        "gpt-4o-mini-audio-preview-2024-12-17": {"input": 0.15, "cached_input": None, "output": 0.60},

        # gpt-4o-mini-realtime-preview
        "gpt-4o-mini-realtime-preview": {"input": 0.60, "cached_input": 0.30, "output": 2.40},
        "gpt-4o-mini-realtime-preview-2024-12-17": {"input": 0.60, "cached_input": 0.30, "output": 2.40},

        # o1
        "o1": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
        "o1-2024-12-17": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
        "o1-preview-2024-09-12": {"input": 15.00, "cached_input": 7.50, "output": 60.00},

        # o1-pro
        "o1-pro": {"input": 150.00, "cached_input": None, "output": 600.00},
        "o1-pro-2025-03-19": {"input": 150.00, "cached_input": None, "output": 600.00},

        # o3
        "o3": {"input": 10.00, "cached_input": 2.50, "output": 40.00},
        "o3-2025-04-16": {"input": 10.00, "cached_input": 2.50, "output": 40.00},

        # o4-mini
        "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
        "o4-mini-2025-04-16": {"input": 1.10, "cached_input": 0.275, "output": 4.40},

        # o3-mini
        "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
        "o3-mini-2025-01-31": {"input": 1.10, "cached_input": 0.55, "output": 4.40},

        # o1-mini
        "o1-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
        "o1-mini-2024-09-12": {"input": 1.10, "cached_input": 0.55, "output": 4.40},

        # gpt-4o-mini-search-preview
        "gpt-4o-mini-search-preview": {"input": 0.15, "cached_input": None, "output": 0.60},
        "gpt-4o-mini-search-preview-2025-03-11": {"input": 0.15, "cached_input": None, "output": 0.60},

        # gpt-4o-search-preview
        "gpt-4o-search-preview": {"input": 2.50, "cached_input": None, "output": 10.00},
        "gpt-4o-search-preview-2025-03-11": {"input": 2.50, "cached_input": None, "output": 10.00},

        # computer-use-preview
        "computer-use-preview": {"input": 3.00, "cached_input": None, "output": 12.00},
        "computer-use-preview-2025-03-11": {"input": 3.00, "cached_input": None, "output": 12.00},

        # gpt-image-1
        "gpt-image-1": {"input": 5.00, "cached_input": None, "output": None},
    }

    
    prompt_tokens = response_json["usage"]["prompt_tokens"]
    completion_tokens = response_json["usage"]["completion_tokens"]
    cached_tokens = response_json["usage"]["prompt_tokens_details"].get("cached_tokens", 0)

    # input token = (prompt_tokens - cached_tokens)
    actual_input_tokens = prompt_tokens - cached_tokens
    output_tokens = completion_tokens

    cost_info = model_cost[model_name]

    input_cost = (actual_input_tokens / 1_000_000) * cost_info['input']
    cached_input_cost = (cached_tokens / 1_000_000) * cost_info['cached_input']
    output_cost = (output_tokens / 1_000_000) * cost_info['output']

    total_cost = input_cost + cached_input_cost + output_cost

    return {
        'model_name': model_name,
        'actual_input_tokens': actual_input_tokens,
        'input_cost': input_cost,
        'cached_tokens': cached_tokens,
        'cached_input_cost': cached_input_cost,
        'output_tokens': output_tokens,
        'output_cost': output_cost,
        'total_cost': total_cost,
    }

def load_accumulated_cost(accumulated_cost_file: str) -> float:
    """Load accumulated cost from a JSON file.
    
    Args:
        accumulated_cost_file: Path to the cost tracking JSON file
        
    Returns:
        Total accumulated cost as a float, or 0.0 if file doesn't exist
    """
    if os.path.exists(accumulated_cost_file):
        with open(accumulated_cost_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("total_cost", 0.0)
    else:
        return 0.0

def save_accumulated_cost(accumulated_cost_file: str, cost: float) -> None:
    """Save accumulated cost to a JSON file.
    
    Args:
        accumulated_cost_file: Path to the cost tracking JSON file
        cost: Total cost to save
    """
    with open(accumulated_cost_file, "w", encoding="utf-8") as f:
        json.dump({"total_cost": cost}, f)

def print_response(completion_json: Dict[str, Any], is_llm: bool = False) -> None:
    """Print the response content from LLM completion.
    
    Args:
        completion_json: Completion response JSON from the API
        is_llm: If True, expects vLLM format; if False, expects OpenAI format
    """
    print("============================================")
    if is_llm:
        print(completion_json['text'])
    else:
        print(completion_json['choices'][0]['message']['content'])
    print("============================================\n")

def print_log_cost(completion_json: Dict[str, Any], gpt_version: str, 
                   current_stage: str, output_dir: str, 
                   total_accumulated_cost: float) -> float:
    """Print and log cost information for an API call.
    
    Calculates the cost for the current API call, updates the total accumulated cost,
    and logs the information to both console and file.
    
    Args:
        completion_json: Completion response JSON from the API
        gpt_version: Model version used
        current_stage: Description of the current stage (e.g., "[Planning] Overall plan")
        output_dir: Directory to save cost logs
        total_accumulated_cost: Running total of costs so far
        
    Returns:
        Updated total accumulated cost including this call
    """
    usage_info = cal_cost(completion_json, gpt_version)

    current_cost = usage_info['total_cost']
    total_accumulated_cost += current_cost

    output_lines = []
    output_lines.append("🌟 Usage Summary 🌟")
    output_lines.append(f"{current_stage}")
    output_lines.append(f"🛠️ Model: {usage_info['model_name']}")
    output_lines.append(f"📥 Input tokens: {usage_info['actual_input_tokens']} (Cost: ${usage_info['input_cost']:.8f})")
    output_lines.append(f"📦 Cached input tokens: {usage_info['cached_tokens']} (Cost: ${usage_info['cached_input_cost']:.8f})")
    output_lines.append(f"📤 Output tokens: {usage_info['output_tokens']} (Cost: ${usage_info['output_cost']:.8f})")
    output_lines.append(f"💵 Current total cost: ${current_cost:.8f}")
    output_lines.append(f"🪙 Accumulated total cost so far: ${total_accumulated_cost:.8f}")
    output_lines.append("============================================\n")

    output_text = "\n".join(output_lines)
    
    print(output_text)

    with open(f"{output_dir}/cost_info.log", "a", encoding="utf-8") as f:
        f.write(output_text + "\n")
    
    return total_accumulated_cost


def num_tokens_from_messages(messages: List[Dict[str, str]], model: str = "gpt-4o-2024-08-06") -> int:
    """Calculate the number of tokens used by a list of messages.
    
    Uses tiktoken to accurately count tokens for cost estimation and context limits.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model name for selecting the appropriate tokenizer
        
    Returns:
        Total number of tokens in the messages
        
    Raises:
        NotImplementedError: If the model is not supported
    """
    import tiktoken
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")

    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value, allowed_special={"<|endoftext|>"},disallowed_special=()))
            
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens



def read_all_files(directory: str, allowed_ext: List[str], is_print: bool = True) -> Dict[str, str]:
    """Recursively read all files with specified extensions from a directory.
    
    Skips hidden files, directories starting with '.', and files larger than 200KB.
    
    Args:
        directory: Root directory to search
        allowed_ext: List of allowed file extensions (e.g., ['.py', '.yaml'])
        is_print: Whether to print skip messages
        
    Returns:
        Dictionary mapping relative file paths to their contents
        
    Note:
        Files larger than 200KB are logged but still included in results.
    """
    all_files_content = {}
    
    for root, _, files in os.walk(directory):  # Recursively traverse directories
        for filename in files:
            relative_path = os.path.relpath(os.path.join(root, filename), directory)  # Preserve directory structure

            _file_name, ext = os.path.splitext(filename)
            
            # Skip hidden directories
            is_skip = False
            if len(directory) < len(root):
                root2 = root[len(directory)+1:]
                for dirname in root2.split("/"):
                    if dirname.startswith("."):
                        is_skip = True
                        break
            
            # Skip hidden files, requirements.txt, and files in hidden directories
            if filename.startswith(".") or "requirements.txt" in filename or ext == "" or is_skip:
                if is_print and ext == "":
                    print(f"[SKIP] {os.path.join(root, filename)}")
                continue
                
            # Check if extension is allowed (special case for README files)
            if ext not in allowed_ext:
                if _file_name.lower() != "readme": 
                    if is_print:
                        print(f"[SKIP] {os.path.join(root, filename)}")
                    continue

            try:
                filepath = os.path.join(root, filename)
                file_size = os.path.getsize(filepath) # bytes
                
                # Log large files (> 200KB)
                if file_size > 204800:
                    print(f"[BIG] {filepath} {file_size}")

                with open(filepath, "r", encoding="utf-8") as file:
                    all_files_content[relative_path] = file.read()
            except Exception as e:
                print(f"[ERROR] {e}")
                print(f"[SKIP] {os.path.join(root, filename)}")
    
    return all_files_content

def read_python_files(directory: str) -> Dict[str, str]:
    """Recursively read all Python files from a directory.
    
    Args:
        directory: Root directory to search
        
    Returns:
        Dictionary mapping relative file paths to their contents
    """
    python_files_content = {}
    
    for root, _, files in os.walk(directory):  # Recursively traverse directories
        for filename in files:
            if filename.endswith(".py"):  # Check if file has .py extension
                relative_path = os.path.relpath(os.path.join(root, filename), directory)  # Preserve directory structure
                with open(os.path.join(root, filename), "r", encoding="utf-8") as file:
                    python_files_content[relative_path] = file.read()
    
    return python_files_content
  

def extract_json_from_string(text: str) -> str:
    """Extract JSON content from markdown code blocks.
    
    Args:
        text: Text containing ```json...``` code blocks
        
    Returns:
        Extracted JSON string, or empty string if not found
    """
    # Extract content inside ```json\n...\n```
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)

    if match:
        json_content = match.group(1)
        return json_content
    else:
        print("No JSON content found.")
        return ""


def get_now_str() -> str:
    """Get current timestamp as a formatted string.
    
    Returns:
        Timestamp in format: YYYYMMdd_HHmmss (e.g., "20250427_205124")
    """
    now = datetime.now()
    now = str(now)
    now = now.split(".")[0]
    now = now.replace("-","").replace(" ","_").replace(":","")
    return now
