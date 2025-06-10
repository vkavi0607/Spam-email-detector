# Spam Email Detector

## Description
Spam Email Detector is a web application that allows users to input a message and detect whether it is spam or not. The application uses a machine learning model trained on SMS spam data to classify messages as spam or ham (not spam). The user-friendly interface enables quick and easy spam detection.

## Features
- Input any message via a web interface.
- Detects spam messages using a pre-trained machine learning model.
- Displays the prediction result instantly.
- Simple and clean user interface.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Spam-email-detector
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Open your web browser and go to:
   ```
   http://localhost:5000
   ```

3. Enter your message in the text area and click the "Detect Spam" button.

4. The prediction ("Spam" or "Ham") will be displayed on the page.

## Project Structure

```
Spam-email-detector/
│
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── data/                   # Dataset files (SMS spam data)
│   ├── sms.tsv
│   └── SMSSpamCollection.tsv
├── models/                 # Trained model and vectorizer files
│   ├── model.pkl
│   └── vectorizer.pkl
├── src/                    # Source code for training, preprocessing, prediction, and config
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
└── templates/              # HTML templates for the web interface
    └── index.html
```

## Dataset

The model is trained on SMS spam datasets located in the `data/` folder. These datasets contain labeled messages classified as spam or ham.

## Acknowledgments

- The project uses Flask for the web application.
- Machine learning model built using scikit-learn.
