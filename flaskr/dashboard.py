from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

from flaskr.auth import login_required
from flaskr.db import get_db

bp = Blueprint('dashboard', __name__)

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
    
    total_neutral = db.execute(
        "SELECT COUNT(*) FROM comments WHERE sentiment_label = 'neutral'"
    ).fetchone()[0]
    
    return render_template(
        'dashboard/index.html', 
        sentiments=sentiments,
        total_comments=total_comments,
        total_positive=total_positive,
        total_negative=total_negative,
        total_neutral=total_neutral
    )