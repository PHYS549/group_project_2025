import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from fermi_data_exploration import load_npy_to_dataframe
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.linalg import norm
from tensorflow import reduce_sum
from fermi_poshist_data import detector_orientation, plot_all_detector_positions

# Load data
fermi_data = load_npy_to_dataframe(data_type='fermi')

# List of detectors
detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]

# Define input columns
input_columns = ['QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4'] + [f"{detector}_PH_CNT" for detector in detectors]

# Extract inputs (X)
X = fermi_data[input_columns].values.astype(np.float64)

# Convert RA and DEC to radians and then to Cartesian unit vectors
ra_rad = np.radians(fermi_data['RA'].values.astype(np.float64))
dec_rad = np.radians(fermi_data['DEC'].values.astype(np.float64))
x = np.cos(dec_rad) * np.cos(ra_rad)
y = np.cos(dec_rad) * np.sin(ra_rad)
z = np.sin(dec_rad)
y = np.stack((x, y, z), axis=-1)

# Clean and split the data
X = np.nan_to_num(X)
y = np.nan_to_num(y)

split_ratio = 0.8
train_size = int(len(X) * split_ratio)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Normalize inputs
mean_X = np.mean(X_train, axis=0)
std_X = np.std(X_train, axis=0)
std_X = np.where(std_X == 0, 1, std_X)

X_train_scaled = (X_train - mean_X) / std_X
X_test_scaled = (X_test - mean_X) / std_X

# Custom cosine similarity loss
def cosine_similarity_loss(y_true, y_pred):
    import tensorflow as tf
    dot_product = reduce_sum(y_true * y_pred, axis=-1)
    norm_true = norm(y_true, axis=-1)
    norm_pred = norm(y_pred, axis=-1)
    cosine_sim = dot_product / (norm_true * norm_pred)
    return -cosine_sim

# Check if the model file exists
model_path = "fermi_direction_model.h5"
if os.path.exists(model_path):
    # Load the existing model with custom objects
    model = load_model(model_path, custom_objects={'cosine_similarity_loss': cosine_similarity_loss, 'CosineSimilarity': CosineSimilarity})
    print("Model loaded from fermi_direction_model.h5")
    
else:
    # Define and train the model as before
    print("No existing model found. Training a new model...")
    
    # Define model with Dropout
    model = Sequential([
        Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(3, activation=None),  # Output layer, no activation
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=cosine_similarity_loss,
        metrics=[CosineSimilarity(name='cosine_similarity')]
    )

    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=300,
        batch_size=16,
        validation_data=(X_test_scaled, y_test)
    )

    # Save the trained model
    model.save(model_path)
    print(f"Model trained and saved as {model_path}")

    # Evaluate the model (if loaded or newly trained)
    loss, cosine_sim = model.evaluate(X_test_scaled, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Cosine Similarity: {cosine_sim:.4f}")

    # Predictions
    predictions = model.predict(X_test_scaled)
    norms = np.linalg.norm(predictions, axis=1)
    print("Mean norm of predicted vectors:", np.mean(norms))
    print("Standard deviation of norms:", np.std(norms))

    # Plot loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Evolution During Training')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot cosine similarity
    plt.plot(history.history['cosine_similarity'], label='Train Cosine Similarity')
    plt.plot(history.history['val_cosine_similarity'], label='Validation Cosine Similarity')
    plt.xlabel('Epochs')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity Evolution During Training')
    plt.legend()
    plt.grid(True)
    plt.show()

def event_detector_orientation(df):
    orientation = detector_orientation(df)
    def average_vector(vecs):
        vecs = np.array(vecs)
        vec = np.array([np.sum(vecs[:,0]), np.sum(vecs[:,1]), np.sum(vecs[:,2])])
        vec = vec/(np.sum(vec**2))**0.5
        return vec
    def convert_to_cartesian(df):
        ra_rad = np.radians(df['RA'].values.astype(np.float64))
        dec_rad = np.radians(df['DEC'].values.astype(np.float64))
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        return np.array([x,y,z])
    print(convert_to_cartesian(df))
    print(average_vector([orientation[1], orientation[2], orientation[5]]))
    plot_all_detector_positions(df)
    

# Extract the event data for ID 'bn170817529'
event_id = 'bn170817529'
event_data = fermi_data[fermi_data['ID'] == event_id]

# Prepare the input for prediction (similar to how you prepared the training data)
event_input = event_data[input_columns].values.astype(np.float64)

# Normalize the input
event_input_scaled = (event_input - mean_X) / std_X

# Make the prediction using the trained model
predicted_direction = model.predict(event_input_scaled)
event_detector_orientation(event_data)


# Normalize the prediction manually to ensure unit vector
predicted_direction_normalized = predicted_direction / np.linalg.norm(predicted_direction, axis=1, keepdims=True)

# Output the predicted direction
print(f"Predicted direction for event {event_id}: {predicted_direction_normalized}")

# Check the norm of the predicted vector (should be close to 1)
norm_predicted_direction = np.linalg.norm(predicted_direction_normalized, axis=1)
print(f"Norm of predicted direction: {norm_predicted_direction}")
