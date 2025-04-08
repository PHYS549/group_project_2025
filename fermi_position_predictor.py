import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fermi_data_exploration import load_npy_to_dataframe

# Load data
fermi_data = load_npy_to_dataframe(data_type='fermi')

# List of detectors
detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]

# Define input columns
input_columns = ['QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4'] + [f"{detector}_PH_CNT" for detector in detectors]

# Extract inputs (X) and outputs (y) (RA and DEC to 3D unit vector)
X = fermi_data[input_columns].values

# Convert RA and DEC columns to NumPy arrays
ra = np.array(fermi_data['RA'].values, dtype=np.float64)
dec = np.array(fermi_data['DEC'].values, dtype=np.float64)

# Convert degrees to radians
ra_rad = np.radians(ra)  # Convert RA to radians
dec_rad = np.radians(dec)  # Convert DEC to radians

# Compute the Cartesian coordinates (unit vector)
x = np.cos(dec_rad) * np.cos(ra_rad)
y = np.cos(dec_rad) * np.sin(ra_rad)
z = np.sin(dec_rad)

# The output will be the unit vector [x, y, z]
y = np.stack((x, y, z), axis=-1)

# Ensure X and y are numpy arrays with the correct dtype
X = np.array(X, dtype=np.float64)
y = np.array(y, dtype=np.float64)

# Manually split the data into training and testing sets (80% train, 20% test)
split_ratio = 0.8
train_size = int(len(X) * split_ratio)

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# Check for NaN or infinite values in X_train and X_test, and handle them
if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
    print("Warning: Found NaN or infinite values in X_train. Replacing with 0.")
    X_train = np.nan_to_num(X_train)

if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
    print("Warning: Found NaN or infinite values in X_test. Replacing with 0.")
    X_test = np.nan_to_num(X_test)

# Check for NaN or infinite values in y_train and y_test, and handle them
if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
    print("Warning: Found NaN or infinite values in y_train. Replacing with 0.")
    y_train = np.nan_to_num(y_train)

if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)):
    print("Warning: Found NaN or infinite values in y_test. Replacing with 0.")
    y_test = np.nan_to_num(y_test)

# Normalize the data (standardization: subtract mean and divide by std)
mean_X = np.mean(X_train, axis=0)
std_X = np.std(X_train, axis=0)

# Avoid division by zero if std_X contains zeros
std_X = np.where(std_X == 0, 1, std_X)

X_train_scaled = (X_train - mean_X) / std_X
X_test_scaled = (X_test - mean_X) / std_X

# Custom Cosine Similarity Metric
def cosine_similarity(y_true, y_pred):
    dot_product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=-1)
    norm_true = tf.linalg.norm(y_true, axis=-1)
    norm_pred = tf.linalg.norm(y_pred, axis=-1)
    cosine_sim = dot_product / (norm_true * norm_pred)
    return -cosine_sim

# Define the neural network model using TensorFlow (Keras API)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_dim=X_train_scaled.shape[1], activation='relu'),  # First hidden layer
    tf.keras.layers.Dense(5, activation='relu'),  # Second hidden layer
    tf.keras.layers.Dense(3)  # Output layer (x, y, z)
])

# Define custom learning rate
learning_rate = 0.0001  # Adjust this value as needed

# Create the Adam optimizer with the specified learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the model with the custom optimizer and cosine similarity as a metric
model.compile(optimizer=optimizer, loss=[cosine_similarity])

# Train the model and capture history
history = model.fit(X_train_scaled, y_train, epochs=1000, batch_size=32, validation_data=(X_test_scaled, y_test))

# Evaluate the model
loss, cosine_sim = model.evaluate(X_test_scaled, y_test)
print(f"Test loss: {loss}")
print(f"Cosine Similarity: {cosine_sim}")

# Make predictions
predictions = model.predict(X_test_scaled)
print(predictions)

# Plot the loss evolution
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Evolution During Training')
plt.legend()
plt.grid(True)
plt.show()

# Plot the cosine similarity evolution
plt.plot(history.history['cosine_similarity'], label='Train Cosine Similarity')
plt.plot(history.history['val_cosine_similarity'], label='Validation Cosine Similarity')
plt.xlabel('Epochs')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity Evolution During Training')
plt.legend()
plt.grid(True)
plt.show()
