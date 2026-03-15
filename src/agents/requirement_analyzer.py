# src/agents/requirement_analyzer.py

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Initialize the Google GenAI client.
# This is what connects to Gemini.
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


async def analyze_requirements(raw_text: str) -> dict:
    """
    Takes raw user input like:
    "users should be able to login, also need password reset"

    Returns structured requirements list like:
    ["User must be able to login with email and password",
     "User must be able to reset their password via email"]
    """

    # This is the "system prompt" — instructions we give the AI
    # about HOW to behave, before it sees the actual user input.
    system_prompt = """
    You are a requirements analyst. Your job is to take raw, 
    informal text and convert it into clear, structured software 
    requirements.

    Rules:
    - Each requirement must start with "The system shall" or "The user must"
    - Be specific and testable
    - One requirement per line
    - Return ONLY the requirements list, no extra text
    - Return as a JSON array of strings

    Example output:
    ["The user must be able to login with email and password",
     "The system shall lock the account after 5 failed attempts"]
    """

    # The actual user message — what we want analyzed
    user_message = f"Please analyze and structure these requirements:\n\n{raw_text}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",  # fast and cheap model, good for this task
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                # temperature controls randomness
                # 0.0 = very deterministic (same output every time)
                # 1.0 = very creative/random
                # 0.2 = mostly consistent, slight variation
                # We want LOW temperature for structured output
            )
        )

        # Extract the text from the response
        response_text = response.text.strip()

        # Clean up the response — AI sometimes wraps JSON in ```json ... ```
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        # Parse the JSON array
        import json
        requirements = json.loads(response_text)

        return {
            "status": "success",
            "requirements": requirements,
            "count": len(requirements)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "requirements": []
        }