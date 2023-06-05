import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(review):
    # Perform sentiment analysis on the review text
    sentiment_scores = sia.polarity_scores(review)
    
    # Interpret the sentiment scores
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment, sentiment_scores

# Example usage
review_text = "I absolutely love this product! It works great and exceeded my expectations."
sentiment, scores = analyze_sentiment(review_text)
print("Review: ", review_text)
print("Sentiment: ", sentiment)
print("Sentiment Scores: ", scores)
