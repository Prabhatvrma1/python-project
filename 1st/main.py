# for i in range(0, len(text) - seq_length, step_size):
#     sentences.append(text[i: i + seq_length])
#     next_char.append(text[i + seq_length])

# # Create input and output data
# x = np.zeros((len(sentences), seq_length, len(characters)), dtype=bool)
# y = np.zeros((len(sentences), len(characters)), dtype=bool)

# for i, sentence in enumerate(sentences):
#     for t, char in enumerate(sentence):
#         x[i, t, char_to_index[char]] = 1
#     y[i, char_to_index[next_char[i]]] = 1

# # Define the model
# model = Sequential()
# model.add(LSTM(128, input_shape=(seq_length, len(characters))))
# model.add(Dense(len(characters)))
# model.add(Activation('softmax'))

# # Compile the model with corrected optimizer parameter
# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))


# # Train the model
# model.fit(x, y, batch_size=256, epochs=4)

# # Save the model
# model.save('textgen.keras')





import pandas as pd
import numpy as np
import tensorflow as tf
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
filepath = tf.keras.utils.get_file('shakes.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]
characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

seq_length = 40
step_size = 3

sentences = []
next_char = []

model = tf.keras.models.load_model('textgen.keras')
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')  
    preds = np.log(preds + 1e-8) / temperature 
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_txt(length, temperature):
    start_index = random.randint(0, len(text) - seq_length - 1)
    generated = ''
    sentence = text[start_index: start_index + seq_length]
    generated += sentence

    for i in range(length):
        x = np.zeros((1, seq_length, len(characters)))
        for t, char in enumerate(sentence):  
            x[0, t, char_to_index[char]] = 1  

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char 

    return generated  


print('------------------0.2----------')
print(generate_txt(300, 0.2))
print('------------------0.4----------')
print(generate_txt(300, 0.4))
print('------------------0.6----------')
print(generate_txt(300, 0.6))
print('------------------0.8----------')
print(generate_txt(300, 0.8))
print('------------------1.0----------')
print(generate_txt(300, 1.0))
