import os, re, sys, json, traceback
from io import StringIO
from typing import List

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS: lets the grader's website call your API (required!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- The shapes of our data ----------
class CodeRequest(BaseModel):
    code: str

class ErrorAnalysis(BaseModel):
    error_lines: List[int]

# ---------- PART 1: The tool that runs code ----------
def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {})                       # run the user's code
        output = sys.stdout.getvalue()       # what it printed
        return {"success": True, "output": output}
    except BaseException:
        output = traceback.format_exc()      # the full crash message
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout              # always restore the screen

# ---------- PART 2: The AI that finds the bad line ----------
def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(code.split("\n"), 1))
    prompt = f"""Analyze this Python code and its error traceback.
Identify the line number(s) in the CODE where the error occurred.
Line numbers start at 1.

CODE (with line numbers):
{numbered}

TRACEBACK:
{tb}

Return the line number(s) where the error is located."""

    resp = requests.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['AIPIPE_TOKEN']}"},
        json={
            "model": "google/gemini-2.0-flash-lite-001",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "error_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error_lines": {"type": "array", "items": {"type": "integer"}}
                        },
                        "required": ["error_lines"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        timeout=30,
    )
    text = resp.json()["choices"][0]["message"]["content"]
    return ErrorAnalysis.model_validate_json(text).error_lines

# Safety net: if the AI call fails, read the line from the traceback
def fallback_lines(tb: str) -> List[int]:
    nums = re.findall(r'File "<string>", line (\d+)', tb)
    return [int(nums[-1])] if nums else []

# ---------- The endpoint ----------
@app.post("/code-interpreter")
def code_interpreter(req: CodeRequest):
    result = execute_python_code(req.code)

    if result["success"]:
        return {"error": [], "result": result["output"]}

    try:
        lines = analyze_error_with_ai(req.code, result["output"])
    except Exception:
        lines = fallback_lines(result["output"])

    return {"error": lines, "result": result["output"]}

# ---------- Q10: Students API ----------
import csv
from typing import Optional
from fastapi import Query

def load_students():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q-fastapi.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [
            {"studentId": int(row["studentId"]), "class": row["class"].strip()}
            for row in csv.DictReader(f)
        ]

STUDENTS = load_students()

@app.get("/api")
def get_students(class_: Optional[List[str]] = Query(None, alias="class")):
    if not class_:
        return {"students": STUDENTS}
    return {"students": [s for s in STUDENTS if s["class"] in class_]}