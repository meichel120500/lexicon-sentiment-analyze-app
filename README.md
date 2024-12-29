# Lexicon Sentiment Analyze App

This project is a Flask-based web application that performs sentiment analysis using a lexicon-based approach.

## Features

- Analyze sentiment of text input
- Display sentiment score and classification
- Simple and intuitive web interface

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/lexicon-sentiment-analyze-app.git
    ```
2. Navigate to the project directory:
    ```bash
    cd lexicon-sentiment-analyze-app
    ```
3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
5. Install the required dependencies:
    - Python dependencies
        ```bash
        pip install -r requirements.txt
        ```
    - Node dependencies
        ```bash
        cd flaskr
        ```
        ```bash
        npm install
        ```


## Usage

1. Run the Flask application:
    - Python app
        ```bash
        flask --app flaskr run --debug
        ```
    - Styles
        ```bash
        npx tailwindcss -i ./static/src/input.css -o ./static/dist/css/output.css --watch
        ```
2. Open your web browser and go to `http://127.0.0.1:5000`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
