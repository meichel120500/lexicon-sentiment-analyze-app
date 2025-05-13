from flask import (Blueprint, flash, g, redirect, render_template, request, url_for)
from werkzeug.exceptions import abort
from flaskr.auth import login_required
from flaskr.db import get_db
import pandas as pd
from main import clean_text, analyze_sentiment
import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


nltk.download('stopwords')

# Ambil stopwords bahasa Indonesia dari nltk
indonesian_stopwords = stopwords.words('indonesian')

# Inisialisasi vectorizer
vectorizer = CountVectorizer(stop_words=indonesian_stopwords)

# Fungsi untuk menghitung metrik evaluasi secara manual
def calculate_metrics(true_labels, predicted_labels, classes):
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    
    for true, pred in zip(true_labels, predicted_labels):
        true_idx = classes.index(true)
        pred_idx = classes.index(pred)
        confusion_matrix[true_idx, pred_idx] += 1
    
    # Akurasi
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    # Precision, Recall, F1 Score
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)

    return confusion_matrix, accuracy, precision, recall, f1

# Blueprint untuk sentiment analysis
bp = Blueprint('sentiment', __name__)

@bp.route('/sentiment', methods=('GET', 'POST'))
def index():
    db = get_db()
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join('storage/temp/csv/', file.filename)
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                file.save(filepath)
                flash(f'File saved to {filepath}')
            except Exception as e:
                flash(f'Failed to save file: {e}')
                return redirect(request.url)

            try:
                data = pd.read_csv(filepath)
                for index, row in data.iterrows():
                    comment = row['comment']
                    sentiment_score = analyze_sentiment(comment)
                    cleaned_text = ' '.join(clean_text(comment))

                    sentiment_label = 'positive' if sentiment_score > 0 else 'negative'

                    db.execute(
                        'INSERT INTO comments (comment, preprocessed_comment, sentiment_score, sentiment_label) VALUES (?, ?, ?, ?)',
                        (comment, cleaned_text, sentiment_score, sentiment_label)
                    )
                db.commit()
                flash('File successfully uploaded and processed')
            except Exception as e:
                flash(f'Failed to process file: {e}')
                return redirect(request.url)
        else:
            flash('Invalid file format. Please upload a CSV file.')
            return redirect(request.url)

    page = request.args.get('page', default=1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    sentiments = db.execute(
        'SELECT id, comment, preprocessed_comment, sentiment_score, sentiment_label'
        ' FROM comments'
        ' LIMIT ? OFFSET ?', (per_page, offset)
    ).fetchall()

    total_comments = db.execute(
        'SELECT COUNT(*) FROM comments'
    ).fetchone()[0]

    total_pages = (total_comments + per_page - 1) // per_page

    return render_template('sentiment/index.html', sentiments=sentiments, page=page, total_comments=total_comments, total_pages=total_pages)

@bp.route('/naive_bayes', methods=('GET', 'POST'))
@login_required
def naive_bayes():
    db = get_db()
    sentiments = db.execute(
        'SELECT id, comment, preprocessed_comment, sentiment_label FROM comments'
    ).fetchall()

    # If no data is available in the database
    if not sentiments:
        flash('Tidak ada data yang tersedia di database.')

        # Set confusion matrix kosong dan metrik 0 jika tidak ada data
        confusion_matrix = np.zeros((2, 2), dtype=int)  # 2 kelas (positive, negative)
        cm_dict = {
            'positive': {'positive': confusion_matrix[0, 0], 'negative': confusion_matrix[0, 1]},
            'negative': {'positive': confusion_matrix[1, 0], 'negative': confusion_matrix[1, 1]}
        }

        return render_template(
            'sentiment/naive_bayes.html',
            processed_results=[],
            page=1,
            total_comments=0,
            total_pages=1,
            accuracy=0,
            confusion_matrix=cm_dict,
            precision=0,
            recall=0,
            f1_score=0,
            img_path=None
        )

    sentiments_sorted = sorted(sentiments, key=lambda x: x['id']) 
    # Proceed with Naive Bayes only if there is data
    X_data = [sentiment['preprocessed_comment'] for sentiment in sentiments]
    y_data = [sentiment['sentiment_label'] for sentiment in sentiments]

    if len(X_data) < 2:
        flash('Tidak ada cukup data untuk membagi menjadi data pelatihan dan pengujian.')

        confusion_matrix = np.zeros((2, 2), dtype=int)
        cm_dict = {
            'positive': {'positive': confusion_matrix[0, 0], 'negative': confusion_matrix[0, 1]},
            'negative': {'positive': confusion_matrix[1, 0], 'negative': confusion_matrix[1, 1]}
        }

        return render_template(
            'sentiment/naive_bayes.html',
            processed_results=[],
            page=1,
            total_comments=len(sentiments),
            total_pages=1,
            accuracy=0,
            confusion_matrix=cm_dict,
            precision=0,
            recall=0,
            f1_score=0,
            img_path=None
        )

    # Proceed with model training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    classes = ['positive', 'negative']
    class_probs = {cls: np.mean(np.array(y_train) == cls) for cls in classes}

    word_probs = {cls: np.zeros(X_train_vec.shape[1]) for cls in classes}
    for cls in classes:
        class_data = X_train_vec[np.array(y_train) == cls] # Ambil data pelatihan untuk kelas ini
        word_counts = np.sum(class_data, axis=0)  # Hitung frekuensi kata untuk kelas ini
        # Laplace smoothing (penambahan 1 pada frekuensi kata dan pembagi dengan jumlah kata + jumlah fitur)
        word_probs[cls] = (word_counts + 1) / (np.sum(word_counts) + X_train_vec.shape[1]) 

    true_labels = []
    predicted_labels = []
    for preprocessed_comment, true_label in zip(X_test, y_test):
        comment_vec = vectorizer.transform([preprocessed_comment]).toarray()

        posteriors = []
        for cls in classes:
            posterior = np.log(class_probs[cls])
            posterior += np.sum(comment_vec * np.log(word_probs[cls]), axis=1)
            posteriors.append(posterior)

        predicted_label = classes[np.argmax(posteriors)]
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # Calculate metrics
    confusion_matrix, accuracy, precision, recall, f1 = calculate_metrics(true_labels, predicted_labels, classes)

    cm_dict = {
        classes[i]: {classes[j]: confusion_matrix[i, j] for j in range(len(classes))}
        for i in range(len(classes))
    }

    # Save confusion matrix as image
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix - Naive Bayes")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    img_path = 'flaskr/static/src/confusion_matrix.png'
    plt.savefig(img_path, format="png")
    plt.close()

    accuracy = np.round(accuracy * 100, 2)
    precision = np.round(np.mean(precision) * 100, 2)
    recall = np.round(np.mean(recall) * 100, 2)
    f1_score = np.round(np.mean(f1) * 100, 2)

    # Pagination setup
    page = request.args.get('page', default=1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    display_results = []
    for sentiment, pred, true in zip(sentiments, predicted_labels, true_labels):
        display_results.append({
            'preprocessed_comment': sentiment['preprocessed_comment'],
            'predicted_label': pred,
            'sentiment_label': sentiment['sentiment_label']
        })

    display_results = display_results[offset:offset + per_page]

    total_comments = len(sentiments)
    total_pages = (total_comments + per_page - 1) // per_page

    return render_template(
        'sentiment/naive_bayes.html',
        processed_results=display_results,
        page=page,
        total_comments=total_comments,
        total_pages=total_pages,
        accuracy=accuracy,
        confusion_matrix=cm_dict,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        img_path=url_for('static', filename='src/confusion_matrix.png')
    )



                        # Pass confusion matrix as a dict

@bp.route('/sentiment/create', methods=('GET', 'POST'))
@login_required
def create():
    result = None
    if request.method == 'POST':
        comment = request.form['comment']
        error = None

        if not comment:
            error = 'Comment is required.'

        if error is not None:
            flash(error)
        else:
            sentiment_score = analyze_sentiment(comment)
            cleaned_text = ' '.join(clean_text(comment))

            sentiment_label = 'positive' if sentiment_score > 0 else 'negative'

            db = get_db()
            db.execute(
                'INSERT INTO comments (comment, preprocessed_comment, sentiment_score, sentiment_label) VALUES (?, ?, ?, ?)',
                (comment, cleaned_text, sentiment_score, sentiment_label)
            )
            db.commit()
            result = {
                'comment': comment,
                'cleaned_text': cleaned_text,
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label
            }

    return render_template('sentiment/create.html', result=result)

@bp.route('/sentiment/delete', methods=('POST',))
@login_required
def delete():
    db = get_db()
    db.execute('DELETE FROM comments')
    db.commit()

    return redirect(url_for('sentiment.index'))