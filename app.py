from flask import Flask, render_template, request
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
with open('model/spam_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the form
    email_text = request.form['email_text']

    # Transform the text using the vectorizer
    email_vec = vectorizer.transform([email_text])

    # Predict if the text is spam or not
    prediction = model.predict(email_vec)[0]

    # Return the result
    result = 'Spam' if prediction == 1 else 'Not Spam'
    return render_template('result.html', prediction=result)



if __name__ == '__main__':
    app.run(debug=True)
