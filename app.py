from flask import Flask, request, jsonify, render_template
from src.predict import predict_spam

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Invalid input. Please provide a "message" key.'}), 400
    message = data['message']
    prediction = predict_spam(message)
    return jsonify({'message': message, 'prediction': prediction})

@app.route('/predict-web', methods=['POST'])
def predict_web():
    message = request.form['message']
    prediction = predict_spam(message)
    return f"<h2>Prediction: {prediction}</h2>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
