import os
import numpy as np
import sys
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense # type: ignore

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
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Preparing input sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)


#A vector is like an arrow. It has two main features: how long it is and the direction 
# it points in. Imagine you’re drawing a line from one point to another on a 
# piece of paper. The line itself, including its length and the way it’s pointing, is like a vector.

# Padding sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Creating predictors and label
predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
label = np.array(label)

# One-hot encoding the labels
label = np.eye(total_words)[label]

# Model architecture
model = Sequential()
model.add(Embedding(total_words, 10))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(predictors, label, epochs=300, verbose=1)
# an epoch is a complete pass through all the training data in one cycle 
# to train a machine learning model



# 1)Input layer (Feeding the Sentence)
# 2)Hidden layers (thinking process,probability of each word)
# 3)Weights and Biases(Learning from Experience)
# 4)Backpropagation Layer(Improving the probability,this is the process wher neural network
# learn from its errors by going back through the layers and adjusting the weights
# )
# 5)Output Layer
#Optimizer(Getting better over time)
#Epochs(practice makes perfect)

# Predict the next word
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(sequence, verbose=0)
    predicted = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return ""

#Purpose of Softmax:
#The softmax function transforms a vector of real numbers into a probability distribution.
#It ensures that the output values are between 0 and 1 and sum up to 1, 

# Test the prediction
print(predict_next_word('improve'))

# data
# deep
# algorithm
# analyze
# learning
