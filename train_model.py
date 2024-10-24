import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Step 1: Load your datasets
# These files must exist in the 'data/' directory (replace with your paths if different)
original_data = pd.read_csv('data/spam_detect.csv')
additional_data = pd.read_csv('data/additional_data.csv')

# Step 2: Combine both datasets
data = pd.concat([original_data, additional_data])

# Step 3: Preprocess the data (X: text, y: label)
X = data['text']  # Assuming 'text' column has email content
y = data['label']  # Assuming 'label' column has 0 (not spam) and 1 (spam)

# Step 4: Vectorize the text data (convert text to numeric features)
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)  # Fit and transform the data

# Step 5: Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_vec, y)

# Step 6: Save the trained model and vectorizer using pickle
# Create a 'model/' directory if it doesn't exist
import os
if not os.path.exists('model'):
    os.makedirs('model')

# Save the trained model
with open('model/spam_classifier.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the fitted vectorizer
with open('model/vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")
