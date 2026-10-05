import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str):
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code, {})
        return {
            "success": True,
            "output": sys.stdout.getvalue()
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc()
        }

    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, error_output: str):
    client = OpenAI(
        api_key=os.environ["AIPIPE_TOKEN"],
        base_url="https://aipipe.org/openai/v1"
    )

    prompt = f"""
Analyze the Python code and traceback below.

Return ONLY JSON in this exact format:
{{"error_lines":[3]}}

CODE:
{code}

TRACEBACK:
{error_output}

Identify the exact source-code line number where the error occurred.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )

    result = ErrorAnalysis.model_validate_json(
        response.choices[0].message.content
    )

    return result.error_lines


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }
