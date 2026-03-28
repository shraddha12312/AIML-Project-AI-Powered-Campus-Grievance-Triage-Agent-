"""
CO5: CASE STUDIES - NLP and its applications, Sentiment Analyzer.
Uses NLTK VADER to determine how frustrated the student is.
"""
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def initialize_analyzer():
    # Downloads the lexicon quietly. If already downloaded, NLTK ignores it.
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

def analyze_frustration(text: str) -> dict:
    """
    Takes raw NLP text and returns a frustration score and priority multiplier.
    A negative sentiment score indicates anger/frustration.
    """
    analyzer = initialize_analyzer()
    scores = analyzer.polarity_scores(text)
    
    # We focus on the 'compound' score which is normalized between -1 (most negative) and +1 (most positive)
    compound_score = scores['compound']
    
    # Logic: The more negative the student, the higher the priority we assign.
    # We invert the score so negative becomes a positive multiplier.
    if compound_score <= -0.5:
        level = "High Frustration"
        priority_boost = 3
    elif compound_score < 0:
        level = "Slightly Frustrated"
        priority_boost = 1.5
    else:
        level = "Neutral / Positive"
        priority_boost = 1
        
    return {
        "compound_score": compound_score,
        "frustration_level": level,
        "priority_boost": priority_boost
    }
