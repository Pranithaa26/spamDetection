import pickle
import numpy as np  # import necessary modules

# Load the original model
with open('model\spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Save it as a new compatible model
with open('spam_classifier_compatible.pkl', 'wb') as f:
    pickle.dump(model, f)
