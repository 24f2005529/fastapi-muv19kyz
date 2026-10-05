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


# CORS required for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact output.

    Returns:
        {
            "success": bool,
            "output": str
        }
    """
    old_stdout = sys.stdout
    stdout_capture = StringIO()
    sys.stdout = stdout_capture

    try:
        exec(code, {})

        output = stdout_capture.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout


def get_exact_error_line(error_output: str) -> List[int]:
    """
    Extract the exact source-code line number from Python's traceback.

    The traceback contains entries such as:
        File "<string>", line 3, in <module>

    The final <string> frame is the actual source location
    where the exception occurred.
    """
    extracted = traceback.extract_tb(sys.exc_info()[2]) if sys.exc_info()[2] else []

    lines = [
        frame.lineno
        for frame in extracted
        if frame.filename == "<string>"
    ]

    if lines:
        return [lines[-1]]

    return []


def analyze_error_with_ai(code: str, error_output: str) -> List[int]:
    """
    Ask the LLM to analyze the traceback and identify error lines.

    The AI result is validated with Pydantic.
    Python's traceback remains authoritative for the final line number.
    """
    client = OpenAI(
        api_key=os.environ["AIPIPE_TOKEN"],
        base_url="https://aipipe.org/openai/v1"
    )

    prompt = f"""
Analyze the following Python code and traceback.

The required output is JSON in exactly this format:
{{"error_lines":[3]}}

Important rules:
- Line numbers refer only to the ORIGINAL Python code.
- Python line numbering starts at 1.
- Do not count traceback lines.
- Identify the line where the exception actually occurred.
- Use the traceback as the primary evidence.

CODE:
{code}

TRACEBACK:
{error_output}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise Python traceback analyzer. "
                    "Return only valid JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    result = ErrorAnalysis.model_validate_json(content)

    return result.error_lines


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    """
    Execute submitted Python code.

    Successful execution:
        {
            "error": [],
            "result": "..."
        }

    Failed execution:
        {
            "error": [line_number],
            "result": "Traceback..."
        }
    """
    execution = execute_python_code(request.code)

    # Successful code: AI is NOT called
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # Error case: AI analysis is performed
    try:
        ai_error_lines = analyze_error_with_ai(
            request.code,
            execution["output"]
        )
    except Exception:
        ai_error_lines = []

    # Extract exact line directly from Python traceback.
    # This prevents LLM off-by-one mistakes.
    traceback_lines = []

    for line in execution["output"].splitlines():
        if 'File "<string>", line ' in line:
            try:
                number_part = line.split('File "<string>", line ')[1]
                line_number = int(number_part.split(",")[0])
                traceback_lines.append(line_number)
            except (ValueError, IndexError):
                pass

    if traceback_lines:
        error_lines = [traceback_lines[-1]]
    else:
        error_lines = ai_error_lines

    return {
        "error": error_lines,
        "result": execution["output"]
    }
