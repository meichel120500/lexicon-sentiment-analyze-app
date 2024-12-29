import os

from flask import Flask, g
from flask_login import LoginManager
from flaskr.auth import load_logged_in_user
from flaskr.db import get_db


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from . import db
    db.init_app(app)

    from . import auth
    app.register_blueprint(auth.bp)

    from . import dashboard
    app.register_blueprint(dashboard.bp)
    app.add_url_rule('/', endpoint='dashboard')

    from .import sentiment
    app.register_blueprint(sentiment.bp)
    app.add_url_rule('/sentiment', endpoint='sentiment_index')

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return get_db().execute(
            'SELECT * FROM users WHERE id = ?', (user_id,)
        ).fetchone()

    @app.context_processor
    def inject_user():
        return dict(current_user=g.user, max=max, min=min)

    return app