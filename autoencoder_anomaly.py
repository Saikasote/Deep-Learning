# a. Import required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

# b. Access the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))

# Keep only digit '0' for training (normal data)
x_train_normal = x_train[y_train.flatten() == 0]

print("Training data shape (normal only):", x_train_normal.shape)
print("Test data shape:", x_test.shape)

# c. Define Encoder
input_dim = 28*28
latent_dim = 64

input_layer = layers.Input(shape=(input_dim,))
# Encoder
encoded = layers.Dense(128, activation='relu')(input_layer)
encoded = layers.Dense(latent_dim, activation='relu')(encoded)

# d. Define Decoder
decoded = layers.Dense(128, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

# Autoencoder Model
autoencoder = models.Model(inputs=input_layer, outputs=decoded)
autoencoder.summary()

# e. Compile the model
autoencoder.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['accuracy'])

# Train the autoencoder
history = autoencoder.fit(x_train_normal, x_train_normal,
                          epochs=50,
                          batch_size=256,
                          validation_split=0.1,
                          verbose=2)

# Evaluate anomaly detection
# Reconstruct test images
x_test_pred = autoencoder.predict(x_test)

# Compute reconstruction error
reconstruction_error = np.mean(np.square(x_test - x_test_pred), axis=1)

# Simple threshold (mean + 2*std of reconstruction error on normal train data)
x_train_pred = autoencoder.predict(x_train_normal)
train_error = np.mean(np.square(x_train_normal - x_train_pred), axis=1)
threshold = np.mean(train_error) + 2*np.std(train_error)

# Detect anomalies
anomalies = reconstruction_error > threshold
print("Detected anomalies:", np.sum(anomalies))
print("Total test samples:", x_test.shape[0])

# Plot some examples
n = 5
plt.figure(figsize=(10,4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    label = y_test[i]
    if isinstance(label, np.ndarray):
        label = label[0]
    plt.title(f"Label:{label}")
    plt.axis('off')

    # Reconstructed
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(x_test_pred[i].reshape(28,28), cmap='gray')
    plt.title(f"Recon Error:{reconstruction_error[i]:.4f}")
    plt.axis('off')
plt.show()
