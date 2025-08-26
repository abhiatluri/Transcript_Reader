import re
import pandas as pd
from typing import List, Tuple, Dict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

COMPLETION_PHRASES = [
    r"\bappointment (?:is )?(?:booked|scheduled|confirmed)\b",
    r"\b(?:book|schedule|confirm)ed (?:you|your|the) appointment\b",
    r"\badded you to the schedule\b",
    r"\b(?:rescheduled|moved) (?:your|the) appointment\b",
    r"\breservation (?:is )?(?:booked|confirmed)\b",
    r"\b(?:sent|emailed|text(?:ed)?) (?:you )?(?:the )?(?:details|information|confirmation)\b",
    r"\b(?:i|we) (?:have|got) (?:your )?(?:name|number|email|details)\b",
    r"\b (?:provided|given) (?:you )?(?:the )?information\b",
    r"\bpayment (?:processed|completed|received)\b",
    r"\bintake (?:form|forms) (?:completed|received|submitted)\b",
    r"\b(?:transferred|connected) you to (?:the )?(?:clinic|staff|front desk|representative)\b",
    r"\b(?:set|arranged|scheduled) (?:a )?callback\b",
    r"\border (?:placed|confirmed|complete)\b",
]

PARTIAL_SUCCESS_PHRASES = [
    r"\bleft (?:a )?voicemail\b",
    r"\bno answer\b",
    r"\boutside (?:of )?business hours\b",
    r"\bmessage (?:has been )?sent to (?:the )?(?:clinic|team|staff)\b",
    r"\bshared our hours\b",
    r"\bprovided (?:our )?address\b",
]

FAILURE_PHRASES = [
    r"\b(?:could not|couldn't|unable to|can't|won't be able to)\b",
    r"\bno availability\b",
    r"\bcall (?:dropped|disconnected)\b",
    r"\b(?:not|no longer) (?:accepting|taking) new patients\b",
    r"\bplease try again later\b",
    r"\b(?:i )?did not understand\b",
    r"\bthat (?:didn't|did not) go through\b",
]

MISUNDERSTANDINGS = [
    r"\b(i )?(didn't|did not) (catch|hear|understand) (that|you)\b",
    r"\bcould you repeat\b",
    r"\bsay that again\b",
    r"\bplease rephrase\b",
]

GOODBYE_PHRASES = [
    r"\bthank you\b",
    r"\bhave a (?:great|good|nice) day\b",
    r"\bbye\b",
    r"\bgoodbye\b"
]

def sentiment_score(text: str) -> Tuple[int, float, Dict[str, float]]:
    if not text:
        return 0, 0.0, {}
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound > 0.05:
        sentiment = 1
    elif compound < -0.05:
        sentiment = -1
    else:
        sentiment = 0
    return sentiment, compound, scores

def count_matches(text: str, patterns: List[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text, flags=re.I))

def classify_transcript(text: str, duration_sec: int | None = None) -> Tuple[str, Dict[str, int | float | dict | str]]:
    t = str(text).strip() if text else ""
    comp = count_matches(t, COMPLETION_PHRASES)
    partial = count_matches(t, PARTIAL_SUCCESS_PHRASES)
    fail = count_matches(t, FAILURE_PHRASES)
    misunderstand = sum(len(re.findall(p, t, flags=re.I)) for p in MISUNDERSTANDINGS)
    senti, compound, senti_details = sentiment_score(t)
    goodbye = count_matches(t, GOODBYE_PHRASES)
    score = 0
    score += 3 * comp
    score += 1 * partial
    score += -3 * fail
    if misunderstand >= 3:
        score -= 2
    score += senti
    if goodbye:
        score += 1
    if duration_sec is not None and duration_sec < 15 and comp == 0 and partial == 0:
        score -= 2
    if comp > 0:
        label = "Success"
        reason = "completion"
    elif partial > 0:
        label = "Success"
        reason = "partial"
    elif score >= 1:
        label = "Success"
        reason = "score"
    else:
        label = "Failure"
        reason = "failure"
    debug = {
        "completion_hits": comp,
        "partial_hits": partial,
        "failure_hits": fail,
        "misunderstand_count": misunderstand,
        "sentiment": senti,
        "sentiment_compound": compound,
        "sentiment_breakdown": senti_details,
        "goodbye_hits": goodbye,
        "duration_penalty_applied": int(duration_sec is not None and duration_sec < 15 and comp == 0 and partial == 0),
        "final_score": score,
        "label_reason": reason
    }
    return label, debug

if __name__ == "__main__":
    df = pd.read_csv("Inbound_Calls_Data.csv")
    print("Columns:", df.columns.tolist())  # 👈 sanity check
    
    results = []
    for i, row in df.head(10).iterrows():
        # auto-detect transcript column
        text_col = next((c for c in df.columns if "transcript" in c.lower()), None)
        text = row.get(text_col, "")
        
        label, debug = classify_transcript(text, duration_sec=row.get("duration", None))
        results.append({
            "row": i,
            "label": label,
            "reason": debug["label_reason"],
            "final_score": debug["final_score"],
            "sentiment": debug["sentiment"],
            "sentiment_compound": debug["sentiment_compound"],
            "sentiment_breakdown": debug["sentiment_breakdown"],
            "sample_text": text[:100]  # 👈 preview first 100 chars
        })
    results_df = pd.DataFrame(results)
    print(results_df)
