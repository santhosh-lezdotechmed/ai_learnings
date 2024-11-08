import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Sample text data
texts = [
    'I love this product',
    'This is the worst purchase I made',
    'Absolutely fantastic! Will buy again',
    'Horrible, do not waste your money',
    'This is great, highly recommend',
    'Not good at all, very disappointed',
    'I am very satisfied with my order',
    'Terrible experience, never again'
]

# Sample labels (1 = positive, 0 = negative)
labels = [1, 0, 1, 0, 1, 0, 1, 0]

# Step 1: Tokenize the text (convert words into integers)
tokenizer = Tokenizer(num_words=5000)  # Use top 5000 most frequent words
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Step 2: Pad sequences to ensure uniform input size
max_length = 10  # Define the maximum length of sequences
X = pad_sequences(sequences, maxlen=max_length)

# Step 3: Convert labels to numpy array
y = np.array(labels)

# Step 4: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build the RNN model
model = Sequential()

# Embedding layer converts integer sequences into dense vectors
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_length))

# RNN layer (SimpleRNN)
model.add(SimpleRNN(units=64, activation='tanh'))

# Dropout layer to prevent overfitting
model.add(Dropout(0.5))

# Output layer for classification (binary classification)
model.add(Dense(1, activation='sigmoid'))

# Step 6: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Train the model
model.fit(X_train, y_train, epochs=5, batch_size=2, validation_data=(X_test, y_test))

# Step 8: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
