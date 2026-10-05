import csv
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# =========================================================
# Q10 - Student API
# =========================================================

CSV_FILE = Path(__file__).parent / "q-fastapi.csv"


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


# =========================================================
# Q11 - Batch Sentiment Analysis
# =========================================================

class SentimentRequest(BaseModel):
    sentences: list[str]


positive_words = {
    "love", "loved", "like", "liked", "great", "excellent",
    "amazing", "wonderful", "fantastic", "awesome", "happy",
    "joy", "joyful", "good", "best", "perfect", "beautiful",
    "enjoy", "enjoyed", "excited", "pleased", "delighted",
    "success", "successful", "brilliant", "superb", "win",
    "won", "glad", "thankful", "fun", "nice", "helpful",
    "impressive", "positive", "recommend", "recommended"
}

negative_words = {
    "hate", "hated", "dislike", "disliked", "bad", "terrible",
    "awful", "horrible", "sad", "angry", "upset", "worst",
    "poor", "disappointed", "disappointing", "failure",
    "failed", "problem", "problems", "pain", "painful",
    "annoying", "annoyed", "frustrated", "frustrating",
    "wrong", "suffer", "suffering", "cry", "crying",
    "boring", "bored", "negative", "regret", "regretted",
    "disaster", "useless", "badly", "difficult"
}


def classify_sentiment(sentence: str) -> str:
    text = sentence.lower()

    positive_score = 0
    negative_score = 0

    # Count positive and negative words
    for word in positive_words:
        if word in text:
            positive_score += 1

    for word in negative_words:
        if word in text:
            negative_score += 1

    # Common negation handling
    negation_phrases = [
        "not ",
        "never ",
        "no ",
        "don't ",
        "didn't ",
        "isn't ",
        "wasn't ",
        "can't ",
        "cannot ",
        "couldn't ",
        "won't ",
        "wouldn't "
    ]

    has_negation = any(phrase in text for phrase in negation_phrases)

    if has_negation:
        positive_score, negative_score = negative_score, positive_score

    if positive_score > negative_score:
        return "happy"

    if negative_score > positive_score:
        return "sad"

    return "neutral"


@app.post("/sentiment")
async def sentiment_analysis(request: SentimentRequest):
    results = []

    for sentence in request.sentences:
        results.append({
            "sentence": sentence,
            "sentiment": classify_sentiment(sentence)
        })

    return {"results": results}
