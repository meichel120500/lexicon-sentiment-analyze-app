from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort
from flaskr.auth import login_required
from flaskr.db import get_db
from main import clean_text, analyze_sentiment
import pandas as pd
import os

bp = Blueprint('sentiment', __name__)

@bp.route('/sentiment', methods=('GET', 'POST'))
@login_required
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
            # Use relative path
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

                    if sentiment_score > 0:
                        sentiment_label = 'positive'
                    elif sentiment_score < 0:
                        sentiment_label = 'negative'
                    else:
                        sentiment_label = 'neutral'

                    db.execute(
                        'INSERT INTO comments (comment, preprocessed_comment, sentiment_score, sentiment_label)'
                        ' VALUES (?, ?, ?, ?)',
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
            # Analyze the sentiment of the comment
            sentiment_score = analyze_sentiment(comment)
            cleaned_text = ' '.join(clean_text(comment))

            if sentiment_score > 0:
                sentiment_label = 'positive'
            elif sentiment_score < 0:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'

            db = get_db()
            db.execute(
                'INSERT INTO comments (comment, preprocessed_comment, sentiment_score, sentiment_label)'
                ' VALUES (?, ?, ?, ?)',
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