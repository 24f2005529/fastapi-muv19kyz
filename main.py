import csv
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

CSV_FILE = Path(__file__).parent / "students.csv"


def load_students():
    students = []

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            students.append({
                "studentId": int(row["studentId"]),
                "class": row["class"]
            })

    return students


@app.get("/api")
def get_students(
    class_: list[str] | None = Query(default=None, alias="class")
):
    students = load_students()

    if class_:
        wanted = set(class_)
        students = [
            student
            for student in students
            if student["class"] in wanted
        ]

    return {"students": students}
