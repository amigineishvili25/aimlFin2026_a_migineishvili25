import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Data Generation (Simulation)
# Simulating 2000 network traffic records with 50 features each
num_samples = 2000
num_features = 50

# Synthetic data: Normal (0) and Malicious/DDoS traffic (1)
X = np.random.rand(num_samples, num_features)
y = np.random.randint(0, 2, num_samples)

# CNN requires data in the format: (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Splitting data into training (80%) and testing (20%) sets
split = int(0.8 * num_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 2. Creating the 1D CNN Model Architecture
model = models.Sequential([
    # Explicit Input layer to avoid Keras warnings
    layers.Input(shape=(num_features, 1)),
    
    # First Convolutional Layer
    layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    
    # Second Convolutional Layer
    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    
    # Flattening and Classification
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Binary (0 or 1) output
])

# 3. Model Compilation
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Training the Model
history = model.fit(X_train, y_train, epochs=15, batch_size=32, 
                    validation_data=(X_test, y_test), verbose=0)

# 5. Model Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Test Accuracy for detecting network anomalies: {test_acc:.4f}")

# 6. Visualizing the training process
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Model Training Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# Saving the plot to the current directory
plt.savefig('cnn_training_plot.png')
