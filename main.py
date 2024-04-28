from textblob import TextBlob
import csv
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk import pos_tag
from nltk.corpus import stopwords
import pandas as pd


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def get_sentiment_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
#
import csv

def assess_reviews(csv_file):
    review_scores_map = {}  # Create an empty dictionary to store the review scores
    with open(csv_file, 'r') as file:  # Open the CSV file in read mode
        reader = csv.reader(file)  # Create a CSV reader object
        next(reader)  # Skip the header row
        for row in reader:  # Iterate over each row in the CSV file
            sentence = ""  # Create an empty string to store the concatenated columns
            for column in row:  # Iterate over each column in the row
                sentence += column + " "  # Concatenate the column values with a space
            sentiment_score = get_sentiment_score(sentence)  # Get the sentiment score for the sentence
            #print(f"Sentiment score: {sentiment_score}")  # Uncomment this line to print the sentiment score
            review_scores_map[row[0]] = sentiment_score  # Add the sentiment score to the dictionary with the review ID as the key
    return review_scores_map  # Return the dictionary of review scores

def assess_reviews(csv_file):
    review_scores_map = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            sentence = ""
            for column in row:
                sentence += column + " "
            sentiment_score = get_sentiment_score(sentence)
            #print(f"Sentiment score: {sentiment_score}")
            review_scores_map[row[0]] = sentiment_score
    return review_scores_map


def global_extract_collocations(tagged_reviews):
    bigram_measures = BigramAssocMeasures()
    all_words = [word for review in tagged_reviews for word, tag in review if tag.startswith(('JJ', 'NN'))]
    finder = BigramCollocationFinder.from_words(all_words)
    finder.apply_freq_filter(3)  # Collocations appearing at least 3 times
    return finder.nbest(bigram_measures.pmi, 40)


# Usage
csv_file = 'data.csv'
scores = assess_reviews(csv_file)

# Tokenize, clean, and POS tag the reviews
def tokenize_and_tag(review):
    tokens = word_tokenize(review)
    tokens = [w.lower() for w in tokens if w.isalpha()]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return pos_tag(filtered_tokens)

tagged_reviews = [tokenize_and_tag(review) for review in scores.keys()]

# Extracting and displaying top collocations for each sentiment category
positive_reviews = [tagged_reviews[i] for i, score in enumerate(scores.values()) if score > 0]
negative_reviews = [tagged_reviews[i] for i, score in enumerate(scores.values()) if score < 0]

top_positive_collocations = global_extract_collocations(positive_reviews)
top_negative_collocations = global_extract_collocations(negative_reviews)

print("Top Positive Collocations:", top_positive_collocations)
print("\n")
print("Top Negative Collocations:", top_negative_collocations)

# Get the top 40 positive and negative reviews
top_positive_reviews = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:40]
top_negative_reviews = sorted(scores.items(), key=lambda x: x[1])[:40]

# Print the top positive reviews
print("Top Positive Reviews:")
for review_id, sentiment_score in top_positive_reviews:
    print(f"Review ID: {review_id}, Sentiment Score: {sentiment_score}")

print ("\n")

# Print the top negative reviews
print("Top Negative Reviews:")
for review_id, sentiment_score in top_negative_reviews:
    print(f"Review ID: {review_id}, Sentiment Score: {sentiment_score}")


