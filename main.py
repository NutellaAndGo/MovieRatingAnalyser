from textblob import TextBlob
import csv
import time
import nltk
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

def get_sentiment_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def assess_reviews(csv_file):
    review_scores_map = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            sentiment_score = 0;
            for column in row:
                current_sentiment_score = get_sentiment_score(column)
                sentiment_score += current_sentiment_score
            print(f"Sentiment score: {sentiment_score}")
            review_scores_map[row[0]] = sentiment_score
    return review_scores_map


def extract_collocations(reviews, sentiment):
    for key, value in reviews.items():
        if value != sentiment:
            del reviews[key]
    

    # Preprocess the reviews
    preprocessed_reviews = [TextBlob(review).words.lower() for review in reviews]

    # Create collocation finders
    bigram_finder = BigramCollocationFinder.from_documents(preprocessed_reviews)
    trigram_finder = TrigramCollocationFinder.from_documents(preprocessed_reviews)

    # Apply part-of-speech filtering
    bigram_finder.apply_freq_filter(3)  # Adjust the frequency threshold as needed
    trigram_finder.apply_freq_filter(3)  # Adjust the frequency threshold as needed

    # Get the top 40 collocations
    top_40_bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 40)
    top_40_trigrams = trigram_finder.nbest(TrigramAssocMeasures.likelihood_ratio, 40)

    return top_40_bigrams, top_40_trigrams

# Usage
csv_file = 'data.csv'
scores = assess_reviews(csv_file)

positive_bigrams, positive_trigrams = extract_collocations(scores, sentiment=1)
negative_bigrams, negative_trigrams = extract_collocations(scores, sentiment=-1)

print("Top 40 positive bigrams:")
for bigram in positive_bigrams:
    print(bigram)

print("Top 40 positive trigrams:")
for trigram in positive_trigrams:
    print(trigram)

print("Top 40 negative bigrams:")
for bigram in negative_bigrams:
    print(bigram)

print("Top 40 negative trigrams:")
for trigram in negative_trigrams:
    print(trigram)