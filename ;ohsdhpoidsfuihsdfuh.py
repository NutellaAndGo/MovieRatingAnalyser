import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk import pos_tag
from nltk.corpus import stopwords
from textblob import TextBlob

# Ensure necessary NLTK resources are available
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

file_path = 'data.csv'  
reviews_with_id = []

# Read and preprocess the reviews
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')
        review_id = parts[0].strip('"')  # Correct ID extraction
        review_text = ' '.join(parts[2:]).replace(',', ' ').strip('"')
        reviews_with_id.append((review_id, review_text))

reviews_df = pd.DataFrame(reviews_with_id, columns=['ID', 'Review'])

# Function to determine sentiment using TextBlob
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply sentiment analysis
reviews_df['Sentiment'] = reviews_df['Review'].apply(get_sentiment)
reviews_df['SentimentCategory'] = reviews_df['Sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative')

# Tokenize, clean, and POS tag the reviews
def tokenize_and_tag(review):
    tokens = word_tokenize(review)
    tokens = [w.lower() for w in tokens if w.isalpha()]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return pos_tag(filtered_tokens)

reviews_df['TaggedTokens'] = reviews_df['Review'].apply(tokenize_and_tag)

# Function to globally extract top collocations based on sentiment
def global_extract_collocations(tagged_reviews):
    bigram_measures = BigramAssocMeasures()
    all_words = [word for review in tagged_reviews for word, tag in review if tag.startswith(('JJ', 'NN'))]
    finder = BigramCollocationFinder.from_words(all_words)
    finder.apply_freq_filter(3)  # Collocations appearing at least 3 times
    return finder.nbest(bigram_measures.pmi, 40)

# Extracting and displaying top collocations for each sentiment category
positive_reviews = reviews_df[reviews_df['SentimentCategory'] == 'positive']['TaggedTokens']
negative_reviews = reviews_df[reviews_df['SentimentCategory'] == 'negative']['TaggedTokens']
top_positive_collocations = global_extract_collocations(positive_reviews.tolist())
top_negative_collocations = global_extract_collocations(negative_reviews.tolist())

print("Top Positive Collocations:", top_positive_collocations)
print("Top Negative Collocations:", top_negative_collocations)

# Print a sample of 40 reviews neatly with IDs, sentiment score, and sentiment category
test_sample_df = reviews_df.sample(n=40, random_state=42)
for index, row in test_sample_df.iterrows():
    sentiment_score = row['Sentiment']
    sentiment_category = row['SentimentCategory']
    print(f"ID: {row['ID']}\nReview: {row['Review']}\nSentiment Score: {sentiment_score}\nSentiment Category: {sentiment_category}\n")
