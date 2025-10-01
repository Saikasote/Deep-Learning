# transfer_learning_cnn.py

# ------------------------------
# a. Import required libraries
# ------------------------------
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ------------------------------
# b. Load training data (CIFAR-10)
# ------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
num_classes = 10
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# ------------------------------
# c. Load pre-trained CNN (MobileNetV2) without top layers
# ------------------------------
base_model = MobileNetV2(input_shape=(32,32,3),
                         include_top=False,
                         weights='imagenet', 
                         pooling='avg')  # Global average pooling

# d. Freeze base model layers (lower convolutional layers)
base_model.trainable = False

# ------------------------------
# e. Add custom classifier layers
# ------------------------------
inputs = layers.Input(shape=(32,32,3))
x = base_model(inputs, training=False)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.summary()

# ------------------------------
# f. Compile model
# ------------------------------
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ------------------------------
# g. Train classifier layers
# ------------------------------
history = model.fit(x_train, y_train_cat,
                    validation_split=0.1,
                    epochs=15,
                    batch_size=64,
                    verbose=2)

# ------------------------------
# h. Evaluate model
# ------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

# ------------------------------
# i. Fine-tuning: Unfreeze some layers
# ------------------------------
base_model.trainable = True

# Fine-tune last 50 layers only
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training (fine-tuning)
history_finetune = model.fit(x_train, y_train_cat,
                             validation_split=0.1,
                             epochs=10,
                             batch_size=64,
                             verbose=2)

# Evaluate again
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)
print(f"Fine-tuned Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

# ------------------------------
# j. Plot training accuracy and loss
# ------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Initial Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Initial Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
