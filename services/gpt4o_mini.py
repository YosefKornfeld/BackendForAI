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
        if char == '{':
            if brace_count == 0:
                json_start = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and json_start != -1:
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

def build_prompt(question: str, search_results: str) -> str:
    """
    Build the user prompt to be sent to the language model.
    """
    return f"""

Search Results:
{search_results}

Do not include any extra text before or after the JSON.
Question: "{question}"
""".strip()

# Instantiate the OpenAI client (ensure openai==1.55.3 and httpx==0.27.2 are used)
client = OpenAI(api_key=settings.openai_api_key)

def get_gpt4mini_answer(question: str) -> dict:
    """
    Calls OpenAI's GPT-4o to generate an answer based on actual search results.
    The model is instructed to incorporate provided search results and output a JSON response.
    Expected JSON format:
      {
        "answer": "<main answer to the question>",
        "qa_list": [
           { "question": "<related question 1>", "answer": "<answer for related question 1>" },
           ...
        ]
      }
    """
    # Retrieve search results for the question
    search_results = get_serp_results(question, num_results=5)
    # Construct the prompt
    prompt = build_prompt(question, search_results)

    try:
        response = client.chat.completions.create(
            model="o1-mini",  # Replace or adjust the model name as appropriate
            messages=[
                {
                    "role": "user",
                    "content": """You are an expert in answering halachic questions. Based on your knowledge and if the search results provided below are correct use them also, answer the following question. Your output must be valid JSON that exactly matches the format described. Do not include any additional text, markdown formatting, or explanation. Output ONLY the JSON object.YOU MUST RETURN A CLEAN JSON, WITHOUT ANY EXTRA TEXT! make sure the json you provide is correctly formatted and is an actual json, the json must be super correct
                    The qa_list should provide very, very detailed example questions.

Generate a JSON response in the following exact format:
{{
  "answer": "<main answer to the question based solely on the search results>",
  "qa_list": [
    {{
      "question": "<related question 1 from the search>",
      "answer": "<answer for related question 1 from the search>"
    }},
    {{
      "question": "<related question 2 from the search>",
      "answer": "<answer for related question 2>"
    }},
    {{
      "question": "<related question 3 from the search>",
      "answer": "<answer for related question 3 from the search>"
    }},
    {{
      "question": "<related question 4 from the search>",
      "answer": "<answer for related question 4>"
    }},
    {{
      "question": "<related question 5 from the search>",
      "answer": "<answer for related question 5 from the search>"
    }}
  ]
}}"""
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

        # 2) Extract JSON from the content
        json_block = extract_json_block(content)
        
        # 3) Parse the extracted JSON
        answer_data = json.loads(json_block)
        
        # (Optional) Validate the structure of the JSON
        if not validate_schema(answer_data):
            raise ValueError("Returned JSON does not match expected schema.")
        
        return answer_data

    except Exception as e:
        print("Error calling GPT-4o-mini API:", e)
        print("Raw GPT output (for debugging):", content if 'content' in locals() else "<No content>")
        # Return a fallback
        return {
            "answer": "Sorry, an error occurred while generating the answer.",
            "qa_list": []
        }
