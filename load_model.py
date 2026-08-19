from tensorflow.keras.models import load_model

# Load the model
model = load_model('models/brain_model.keras')

# Print model summary
model.summary()

# Example: If you have input data, you can predict
# predictions = model.predict(your_input_data)