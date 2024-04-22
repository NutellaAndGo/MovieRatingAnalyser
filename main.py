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


def extract_collocations(reviews, sentiment):
      # Tokenize the reviews and filter out stopwords
    tokens = [word for review in reviews for word in nltk.word_tokenize(review) if word not in stopwords.words('english')]
    
    # If part-of-speech filtering is enabled, only keep tokens that are nouns or adjectives
    tokens = [token for token, pos in pos_tag(tokens) if not pos.startswith('N') and not pos.startswith('J') and not pos.startswith('NNP')]

    # Use BigramCollocationFinder to find bigrams
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    print(bigram_finder)
    
    # Filter bigrams to only those that appear at least 3 times
    bigram_finder.apply_freq_filter(3)
    
    # If sentiment is positive, return the 40 bigrams with the highest PMI (Pointwise Mutual Information)
    if sentiment == 1:
        return bigram_finder.nbest(BigramAssocMeasures.pmi, 40)
    # If sentiment is negative, return the 40 bigrams with the highest Chi-Square
    elif sentiment == -1:
        return bigram_finder.nbest(BigramAssocMeasures.chi_sq, 40)

# Usage
csv_file = 'data.csv'
scores = assess_reviews(csv_file)

positive_bigrams = extract_collocations(scores, sentiment=1)
negative_bigrams = extract_collocations(scores, sentiment=-1)

print("Top 40 positive bigrams:")
print(positive_bigrams)

print("Top 40 negative bigrams:")
print(negative_bigrams)

