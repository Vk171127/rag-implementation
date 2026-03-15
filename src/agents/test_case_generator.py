# src/agents/test_case_generator.py

import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


async def generate_test_cases(
    requirements: list[str],
    rag_context: list[str] = None
) -> dict:
    """
    Takes a list of requirements + optional RAG context,
    returns structured test cases.

    requirements = ["The user must login with email and password", ...]
    rag_context  = relevant docs from ChromaDB (may be empty)
    """

    # Format requirements as numbered list for the prompt
    requirements_text = "\n".join(
        f"{i+1}. {req}" for i, req in enumerate(requirements)
    )

    # Format RAG context if we have any
    # This is the "augmented" part of RAG — we're giving the AI
    # extra knowledge from our own documents
    context_text = ""
    if rag_context:
        context_text = "\n\nRelevant context from our knowledge base:\n"
        context_text += "\n".join(f"- {ctx}" for ctx in rag_context)

    system_prompt = """
    You are an expert QA engineer. Generate comprehensive test cases
    for the given requirements.

    For each test case provide:
    - test_name: short descriptive name
    - test_type: functional / security / edge_case / negative
    - priority: high / medium / low
    - preconditions: what must be true before running
    - test_steps: list of steps to execute
    - expected_result: what should happen

    Return ONLY a JSON array of test case objects. No extra text.
    """

    user_message = f"""
    Generate test cases for these requirements:

    {requirements_text}
    {context_text}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3
                # slightly higher than analyzer —
                # we want some creativity in test case generation
            )
        )

        response_text = response.text.strip()

        # Clean markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        test_cases = json.loads(response_text)

        return {
            "status": "success",
            "test_cases": test_cases,
            "count": len(test_cases),
            "used_rag_context": rag_context is not None and len(rag_context) > 0
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "test_cases": []
        }