from textblob import TextBlob
import csv
import time
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords
from nltk import pos_tag

def get_sentiment_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

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



# Usage
csv_file = 'data.csv'
scores = assess_reviews(csv_file)

# Sort the scores
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Print the top 40 positive and negative reviews
print("Top 40 positive reviews:")
for i in range(40):
    print(sorted_scores[i])

print("Top 40 negative reviews:")
for i in range(40):
    print(sorted_scores[-i-1])


