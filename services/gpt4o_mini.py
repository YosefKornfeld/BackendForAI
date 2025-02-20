# services/gpt4o_mini.py
import json
import re
from openai import OpenAI

from config import settings
from services.serp_search import get_serp_results

def remove_invisible_chars(text: str) -> str:
    """
    Removes many invisible / zero-width / direction characters
    that may appear in Hebrew or other RTL text, causing JSON parsing issues.
    """
    # This range covers most zero-width / direction chars (U+200B to U+200F, U+202A to U+202E, etc.)
    # You can add more if needed.
    return re.sub(r'[\u200B-\u200F\u202A-\u202E]', '', text)

def normalize_quotes(text: str) -> str:
    """
    Replace curly/smart quotes or apostrophes with standard ASCII quotes/apostrophes.
    """
    replacements = {
        '“': '"',  # left double quote
        '”': '"',  # right double quote
        '‘': "'",  # left single quote
        '’': "'",  # right single quote
        '״': '"',  # Hebrew double quote (gereshayim)
        '׳': "'",  # Hebrew single quote (geresh)
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def extract_json_block(text: str) -> str:
    """
    Attempt to extract the first valid JSON object enclosed by { } from the text.
    This version is more robust:
      - Removes triple-backticks.
      - Searches multiple possible code fences.
      - Finds the first substring that is valid JSON, if possible.
    """
    # Remove code fences in case the response is wrapped in ```json ... ``` or similar
    cleaned_text = re.sub(r"```(\w+)?", "", text).strip("`")
    
    # This regex-like approach looks for a balanced JSON structure from '{' to '}'
    brace_count = 0
    json_start = -1
    
    for i, char in enumerate(cleaned_text):
        if (char == '{'):
            if (brace_count == 0):
                json_start = i
            brace_count += 1
        elif (char == '}'):
            brace_count -= 1
            if (brace_count == 0 and json_start != -1):
                # We found a balanced JSON string from json_start to i
                return cleaned_text[json_start:i+1]
    
    # If we couldn't find a well-formed JSON block, fallback to the original text
    return text

def validate_schema(json_data: dict) -> bool:
    """
    Optional step: Validate that the returned JSON structure matches our expected schema:
      {
        "answer": <string>,
        "qa_list": [
          {"question": <string>, "answer": <string>},
          ...
        ]
      }
    Returns True if it matches, False otherwise.
    """
    if not isinstance(json_data, dict):
        return False
    
    if "answer" not in json_data or "qa_list" not in json_data:
        return False
    
    if not isinstance(json_data["answer"], str):
        return False
    
    if not isinstance(json_data["qa_list"], list):
        return False
    
    # Optionally check for exact keys in each QA pair
    for item in json_data["qa_list"]:
        if not isinstance(item, dict):
            return False
        if "question" not in item or "answer" not in item:
            return False
    
    return True

def build_prompt(question: str, qa_pairs: str) -> str:
    """
    Build the user prompt to be sent to the language model.
    """
    return f"""

Previous Q&A pairs (in JSON format):
{qa_pairs}

Current question:
"{question}"
""".strip()

# Instantiate the OpenAI client (ensure openai==1.55.3 and httpx==0.27.2 are used)
client = OpenAI(api_key=settings.openai_api_key)

def get_gpt4mini_answer(question: str, qa_pairs: str) -> dict:
    """
    Calls OpenAI's GPT-4o to generate an answer based on actual search results and Q&A pairs.
    The model is instructed to incorporate provided search results and Q&A pairs, and output the main answer.
    """
    # Construct the prompt
    prompt = build_prompt(question, qa_pairs)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Replace or adjust the model name as appropriate
            messages=[
                {
                    "role": "user",
                    "content": """You are an expert in answering halachic questions. You have been provided with a 
                        list of previous Q&A pairs that may be relevant to the current question. Your task is 
                        to carefully analyze these Q&A pairs and determine which information applies to answering 
                        the current question. If a direct answer is present, use it; if not, infer the answer from 
                        the context provided by the Q&A pairs and your expertise. 
                        In your response, include a detailed explanation of your reasoning by referring directly 
                        to the relevant parts of the provided Q&A pairs (for example, 'In Q&A pair 1, ...' or 
                        'Based on Q&A pair 2, ...'). Ensure that your answer is coherent and comprehensive. 
                        Output your answer as plain text with no markdown formatting, code blocks, or additional commentary."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        content = response.choices[0].message.content.strip()
        
        # 1) Clean the content from invisible characters and normalize quotes
        content = remove_invisible_chars(content)
        content = normalize_quotes(content)

        # Construct the JSON response
        answer_data = {
            "answer": content,
            "qa_list": json.loads(qa_pairs)
        }
        
        return answer_data

    except Exception as e:
        print("Error calling GPT-4o-mini API:", e)
        print("Raw GPT output (for debugging):", content if 'content' in locals() else "<No content>")
        # Return a fallback
        return {
            "answer": "Sorry, an error occurred while generating the answer.",
            "qa_list": json.loads(qa_pairs)
        }
