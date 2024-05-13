import os
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Attention , MultiHeadAttention
from tensorflow.keras.models import Model

# Set the environment variable for oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set the standard output encoding to UTF-8
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Example corpus of text
corpus = [
    'deep learning algorithms improve',
    'deep learning algorithms analyze',
    'algorithms improve data processing',
    'data processing with deep learning'
]

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Preparing input sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Padding sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Creating predictors and label
predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
label = np.array(label)

# One-hot encoding the labels
label = np.eye(total_words)[label]

# Transformer architecture
embedding_dim = 32
num_heads = 4
feed_forward_dim = 64

inputs = Input(shape=(max_sequence_len - 1,))
embedding_layer = Embedding(total_words, embedding_dim)(inputs)
# Use MultiHeadAttention instead of Attention
attention_layer = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(embedding_layer, embedding_layer)
flatten_layer = tf.keras.layers.Flatten()(attention_layer)
dense_layer = Dense(feed_forward_dim, activation='relu')(flatten_layer)
output_layer = Dense(total_words, activation='softmax')(dense_layer)

model = Model(inputs=inputs, outputs=output_layer)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(predictors, label, epochs=100, verbose=1)

# Predict the next word
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(sequence, verbose=0)
    predicted = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return ""

# Test the prediction
print(predict_next_word('improve'))
