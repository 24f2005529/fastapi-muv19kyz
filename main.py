import csv
import re
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

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


POSITIVE_WORDS = {
    "love", "loved", "lovely", "like", "liked", "likes",
    "great", "excellent", "amazing", "amazingly", "awesome",
    "wonderful", "fantastic", "fabulous", "brilliant", "superb",
    "perfect", "best", "good", "better", "nice", "enjoy",
    "enjoyed", "enjoyable", "happy", "happier", "happiness",
    "joy", "joyful", "excited", "exciting", "thrilled",
    "delighted", "pleased", "glad", "grateful", "thankful",
    "hopeful", "optimistic", "fun", "funny", "impressive",
    "success", "successful", "win", "wins", "won", "victory",
    "positive", "recommend", "recommended", "satisfied",
    "satisfaction", "beautiful", "brilliant", "cheerful",
    "smile", "smiling", "laugh", "laughing", "relaxed",
    "peaceful", "pleasant", "pleasure", "favorite", "favourite",
    "adorable", "incredible", "outstanding", "magnificent",
    "terrific", "delightful", "remarkable", "proud", "pride"
}

NEGATIVE_WORDS = {
    "hate", "hated", "hates", "dislike", "disliked", "dislikes",
    "bad", "worse", "worst", "terrible", "horrible", "awful",
    "atrocious", "disgusting", "disgusted", "sad", "sadder",
    "sadness", "unhappy", "anger", "angry", "mad", "furious",
    "upset", "annoyed", "annoying", "irritated", "irritating",
    "frustrated", "frustrating", "disappointed", "disappointing",
    "disappointment", "poor", "failure", "failed", "fail",
    "problem", "problems", "issue", "issues", "pain", "painful",
    "suffer", "suffering", "cry", "crying", "tears", "boring",
    "bored", "boring", "negative", "regret", "regretted",
    "disaster", "useless", "worthless", "ridiculous", "stupid",
    "horrendous", "dreadful", "miserable", "misery", "lonely",
    "loneliness", "fear", "afraid", "scared", "terrified",
    "worried", "worry", "worrying", "stress", "stressed",
    "horrific", "evil", "badly", "broken", "damage", "damaged",
    "loss", "lost", "losing", "complaint", "complain",
    "complained", "dislike", "weak", "annoyance", "disaster"
}

# Strong sentiment phrases
POSITIVE_PHRASES = {
    "feel great",
    "feeling great",
    "feel good",
    "feeling good",
    "feel happy",
    "feeling happy",
    "very happy",
    "so happy",
    "really happy",
    "absolutely love",
    "really love",
    "truly love",
    "love it",
    "love this",
    "love that",
    "highly recommend",
    "very pleased",
    "very satisfied",
    "great experience",
    "wonderful experience",
    "best ever",
    "made me happy",
    "made my day",
    "looking forward",
    "can't wait",
    "cannot wait",
    "so excited",
    "really excited",
    "very excited",
    "extremely happy"
}

NEGATIVE_PHRASES = {
    "feel terrible",
    "feeling terrible",
    "feel horrible",
    "feeling horrible",
    "feel awful",
    "feeling awful",
    "feel sad",
    "feeling sad",
    "very sad",
    "so sad",
    "really sad",
    "absolutely hate",
    "really hate",
    "hate it",
    "hate this",
    "hate that",
    "very disappointed",
    "really disappointed",
    "extremely disappointed",
    "very angry",
    "really angry",
    "extremely angry",
    "very upset",
    "really upset",
    "very frustrated",
    "really frustrated",
    "terrible experience",
    "horrible experience",
    "worst ever",
    "not happy",
    "not good",
    "not great",
    "not satisfied",
    "very unhappy",
    "extremely unhappy",
    "can't stand",
    "cannot stand",
    "fed up",
    "not worth"
}


def classify_sentiment(sentence: str) -> str:
    text = sentence.lower().strip()

    # Normalize punctuation
    cleaned = re.sub(r"[^a-z0-9\s']", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    positive_score = 0
    negative_score = 0

    # Phrase matching gets higher weight
    for phrase in POSITIVE_PHRASES:
        if phrase in cleaned:
            positive_score += 3

    for phrase in NEGATIVE_PHRASES:
        if phrase in cleaned:
            negative_score += 3

    # Token-based matching
    words = set(re.findall(r"\b[a-z]+\b", cleaned))

    positive_score += sum(
        1 for word in POSITIVE_WORDS if word in words
    )

    negative_score += sum(
        1 for word in NEGATIVE_WORDS if word in words
    )

    # Handle explicit negation of individual sentiment words
    negation_patterns = [
        r"\bnot\s+(good|great|happy|nice|excellent|amazing|wonderful|love|like|perfect)\b",
        r"\bnever\s+(good|great|happy|love|like)\b",
        r"\bdon't\s+(like|love|enjoy)\b",
        r"\bdo\s+not\s+(like|love|enjoy)\b",
        r"\bdoesn't\s+(like|love|enjoy)\b",
        r"\bdidn't\s+(like|love|enjoy)\b",
        r"\bcan't\s+(enjoy|like|love)\b",
        r"\bcannot\s+(enjoy|like|love)\b",
    ]

    for pattern in negation_patterns:
        if re.search(pattern, cleaned):
            negative_score += 2

    # "not bad" / "not terrible" is generally positive
    if re.search(r"\bnot\s+(bad|terrible|awful|horrible|worst)\b", cleaned):
        positive_score += 2

    # Emoji clues
    if any(x in text for x in ["😊", "😄", "😀", "😍", "🥰", "❤️", "❤", "👍", "🎉"]):
        positive_score += 2

    if any(x in text for x in ["😢", "😭", "😞", "😔", "😡", "🤬", "💔", "👎"]):
        negative_score += 2

    if positive_score > negative_score:
        return "happy"

    if negative_score > positive_score:
        return "sad"

    return "neutral"


@app.post("/sentiment")
async def sentiment_analysis(request: SentimentRequest):
    return {
        "results": [
            {
                "sentence": sentence,
                "sentiment": classify_sentiment(sentence)
            }
            for sentence in request.sentences
        ]
    }
