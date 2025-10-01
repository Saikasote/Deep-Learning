# a. Data preparation
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Lambda, Dense
import tensorflow.keras.backend as K

# Toy corpus
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the mouse",
    "the dog chased the cat"
]

# Tokenize words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
word2idx = tokenizer.word_index
idx2word = {v:k for k,v in word2idx.items()}
vocab_size = len(word2idx) + 1  # +1 for padding index 0

# Create sequences
sequences = []
window_size = 2
for sentence in corpus:
    tokens = tokenizer.texts_to_sequences([sentence])[0]
    for i, target_word in enumerate(tokens):
        context = []
        for j in range(i - window_size, i + window_size + 1):
            if j != i and j >= 0 and j < len(tokens):
                context.append(tokens[j])
        for context_word in context:
            sequences.append((context_word, target_word))  # (context, target)

# Separate inputs and outputs
X = []
Y = []
for context_word, target_word in sequences:
    X.append(context_word)
    Y.append(target_word)

X = np.array(X)
Y = np.array(Y)

print("Vocabulary:", word2idx)
print("Sample X, Y:", X[:10], Y[:10])

# b. Generate training data
# One-hot encode target
Y = tf.keras.utils.to_categorical(Y, num_classes=vocab_size)

# c. Train CBOW model
embedding_dim = 10

# Input
input_target = Input(shape=(1,))
# Embedding layer
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1)(input_target)
embedding = Lambda(lambda x: K.mean(x, axis=1))(embedding)  # CBOW: average embeddings
# Output layer (predict target word)
output = Dense(vocab_size, activation='softmax')(embedding)

# Model
model = Model(inputs=input_target, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X, Y, epochs=100, verbose=2)

# d. Output
# Get word embeddings
embeddings = model.get_layer('embedding').get_weights()[0]

print("\nWord embeddings:")
for word, i in word2idx.items():
    print(word, embeddings[i])
