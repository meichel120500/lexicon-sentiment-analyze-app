from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
    )
from werkzeug.exceptions import abort
from flaskr.auth import login_required
from flaskr.db import get_db
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import pandas as pd

bp = Blueprint('dashboard', __name__)

def load_sentiment_dictionary():
        """Memuat file kamus sentimen."""
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Mendapatkan direktori file dashboard.py
        positive_path = os.path.join(base_dir, '../positive.tsv')  # Path ke file positive.tsv
        negative_path = os.path.join(base_dir, '../negative.tsv')  # Path ke file negative.tsv

        # Membaca file dengan konversi kolom weight ke float
        positive_words = pd.read_csv(positive_path, sep='\t', names=['word', 'weight'])
        positive_words['weight'] = pd.to_numeric(positive_words['weight'], errors='coerce')  # Konversi ke float
        
        negative_words = pd.read_csv(negative_path, sep='\t', names=['word', 'weight'])
        negative_words['weight'] = pd.to_numeric(negative_words['weight'], errors='coerce')  # Konversi ke float
        
        return (
            positive_words.set_index('word')['weight'].to_dict(), 
            negative_words.set_index('word')['weight'].to_dict()
        )


def classify_words(comments, positive_dict, negative_dict):
        """Mengklasifikasikan kata-kata dalam komentar menjadi positif, negatif, atau netral."""
        positive_words = []
        negative_words = []
       

        for comment in comments:
            words = comment['preprocessed_comment'].split()
            for word in words:
                if word in positive_dict:
                    positive_words.append((word, positive_dict[word]))
                elif word in negative_dict:
                    negative_words.append((word, negative_dict[word]))
                

        return positive_words, negative_words

def generate_wordcloud(word_list, filename):
        """Menghasilkan WordCloud dari daftar kata dan menyimpannya ke file."""
        if not word_list:
            print(f"No words to generate wordcloud for {filename}")
            return False

        word_freq = {word: weight for word, weight in word_list}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

        wordcloud_path = os.path.join('flaskr', 'static', filename)
        wordcloud.to_file(wordcloud_path)

        return True  # Mengembalikan True jika gambar berhasil dibuat
@bp.route('/')
@login_required
def index():
        db = get_db()
        sentiments = db.execute(
            'SELECT id, comment, preprocessed_comment, sentiment_score, sentiment_label'
            ' FROM comments'
        ).fetchall()
        
        total_comments = db.execute(
            'SELECT COUNT(*) FROM comments'
        ).fetchone()[0]
        
        total_positive = db.execute(
            "SELECT COUNT(*) FROM comments WHERE sentiment_label = 'positive'"
        ).fetchone()[0]
        
        total_negative = db.execute(
            "SELECT COUNT(*) FROM comments WHERE sentiment_label = 'negative'"
        ).fetchone()[0]
        

        # Memuat kamus sentimen
        positive_dict, negative_dict = load_sentiment_dictionary()

        # Mengklasifikasikan kata-kata
        positive_words, negative_words = classify_words(sentiments, positive_dict, negative_dict)

        # Membuat WordCloud hanya jika ada kata-kata
        positive_wordcloud_exists = generate_wordcloud(positive_words, 'positive_wordcloud.png')
        negative_wordcloud_exists = generate_wordcloud(negative_words, 'negative_wordcloud.png')

        return render_template(
            'dashboard/index.html', 
            sentiments=sentiments,
            total_comments=total_comments,
            total_positive=total_positive,
            total_negative=total_negative,
            positive_wordcloud_exists=positive_wordcloud_exists,  # Kirim status ke template
            negative_wordcloud_exists=negative_wordcloud_exists
        )