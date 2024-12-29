import pandas as pd
import re
from flask import Flask, request, jsonify
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load lexicon-based sentiment analysis
positive_df = pd.read_csv('positive.tsv', sep='\t')
negative_df = pd.read_csv('negative.tsv', sep='\t')

# Verify the column names
print(positive_df.columns)
print(negative_df.columns)

# Create the lexicon from datasets
indonesian_lexicon = {}

# Add positive words to lexicon with their weight
for _, row in positive_df.iterrows():
    indonesian_lexicon[row['word']] = row['weight']

# Add negative words to lexicon with their weight
for _, row in negative_df.iterrows():
    indonesian_lexicon[row['word']] = row['weight']

# Initialize the app
app = Flask(__name__)

# Text cleaning
def clean_text(text):
    # Remove emoticons, numbers, and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Transform text to lowercase
    text = text.lower()

    # Tokenize the text
    words = text.split()

    # Normalize the words with dictionary 'slangwords.txt' the file format is 'singkatan:panjang'
    slang_words = {}
    with open('slangwords.txt') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) == 2:
                slang, formal = parts
                slang_words[slang] = formal

    words = [slang_words.get(word, word) for word in words]

    # Normalize it too with another dictionary 'kata.txt' the file format is 'singkatan:panjang'
    formal_words = {}
    with open('kata.txt') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) == 2:
                slang, formal = parts
                formal_words[slang] = formal

    words = [formal_words.get(word, word) for word in words]

    # Remove stopwords with stopwords dictionary 'combined_stop_words.txt' the file format is 'stopword'
    stop_words = set()
    with open('combined_stop_words.txt') as f:
        for line in f:
            stop_words.add(line.strip())

    words = [word for word in words if word not in stop_words]

    # Stemming the words with Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    words = [stemmer.stem(word) for word in words]

    return words

def analyze_sentiment(text):
    # Split the text into words
    words = clean_text(text)
    # Initialize the sentiment score
    sentiment_score = 0
    # Loop through all the words
    for word in words:
        # If the word is in the lexicon, add the weight to the sentiment score
        if word in indonesian_lexicon:
            sentiment_score += indonesian_lexicon[word]
    # Return the sentiment score
    return sentiment_score

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    sentiment_score = analyze_sentiment(text)

    if sentiment_score > 0:
        sentiment = 'positive'
    elif sentiment_score < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return jsonify({
        'cleaned_text': ' '.join(clean_text(text)),
        'sentiment_score': sentiment_score,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)