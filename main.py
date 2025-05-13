import pandas as pd
import re
from flask import Flask, request, jsonify
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from functools import lru_cache

# Load lexicon-based sentiment analysis
positive_df = pd.read_csv('positive.tsv', sep='\t')
negative_df = pd.read_csv('negative.tsv', sep='\t')

# Create the lexicon from datasets
indonesian_lexicon = {row['word']: row['weight'] for _, row in positive_df.iterrows()}
indonesian_lexicon.update({row['word']: row['weight'] for _, row in negative_df.iterrows()})

# Initialize the app
app = Flask(__name__)

# Global variables for preprocessing
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load slang words from file
slang_words = {}
with open('kamuskatabaku.txt') as f:
    for line in f:
        if ',' in line:
            key, value = line.strip().split(',')
            slang_words[key] = value

# Load formal words from file
formal_words = {}
with open('kamuskatabaku.txt') as f:
    for line in f:
        if ',' in line:
            key, value = line.strip().split(',')
            formal_words[key] = value

# Load stopwords
with open('combined_stop_words.txt') as f:
    stop_words = set(line.strip() for line in f)

# Text cleaning
@lru_cache(maxsize=1024)
def clean_text(text):
    # Remove URLs, mentions, hashtags, digits, non-alphabetic characters, and extra spaces
    text = re.sub(r'http[s]?://\S+|@\w+|#\w+|\d+|[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()

    # Tokenize the text
    words = text.split()

    # Normalize using slang words and formal words
    words = [slang_words.get(word, word) for word in words]
    words = [formal_words.get(word, word) for word in words]

    # Remove stopwords but retain meaningful words for sentiment analysis
    meaningful_words = [word for word in words if word not in stop_words or word in indonesian_lexicon]

    # Stem the words
    stemmed_words = [stemmer.stem(word) for word in meaningful_words]

    return stemmed_words

def analyze_sentiment(text):
    # Split the text into words
    words = clean_text(text)
    # Compute the sentiment score
    sentiment_score = sum(indonesian_lexicon.get(word, 0) for word in words)
    return sentiment_score

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    sentiment_score = analyze_sentiment(text)

    if sentiment_score > 0:
        sentiment = 'positive'
    else:
        sentiment = 'negative'

    return jsonify({
        'cleaned_text': ' '.join(clean_text(text)),
        'sentiment_score': sentiment_score,
        'sentiment': sentiment
    })
# Tambahkan endpoint untuk membersihkan cache
@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    # Membersihkan cache dari fungsi clean_text
    clean_text.cache_clear()
    return jsonify({"message": "Cache berhasil dibersihkan"})

if __name__ == '__main__':
    app.run(debug=True)
