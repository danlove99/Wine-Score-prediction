import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
import tensorflow as tf  
from tensorflow import keras 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint

# Variables for word embedding
vocab_size = 10000
embedding_dim = 16
max_length = 50
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 35000

# Load data and create target array
df = pd.read_csv('wine.csv')
y = df['points'].values

# One hot encode categorical columns
encoded_countries = pd.get_dummies(df['country'])
encoded_provinces = pd.get_dummies(df['province'])

# Embedding for titles
titles = []
for index, row in df.iterrows():
	titles.append(row['title'])
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(titles)
print("Fit tokenizer on wine titles")
word_index = tokenizer.word_index
titles_sequences = tokenizer.texts_to_sequences(titles)
titles_padded = pad_sequences(titles_sequences, maxlen=max_length, padding=padding_type,
								truncating=trunc_type)

# Embedding for descriptions
descriptions = []
for index, row in df.iterrows():
	descriptions.append(row['description'])
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(descriptions)
print("Fit tokenizer on wine descriptions")
word_index = tokenizer.word_index
descriptions_sequences = tokenizer.texts_to_sequences(descriptions)
descriptions_padded = pad_sequences(descriptions_sequences, maxlen=max_length, padding=padding_type,
									truncating=trunc_type)

# Drop preprocessed columns and unwanted columns
df = df.drop(['taster_twitter_handle','taster_name','region_2',
	'region_1','winery','variety','province','country','designation',
	'points', 'title', 'description'], axis=1)

# concatenate for final array
X = df.values
arrays = (X, np.array(descriptions_padded), np.array(titles_padded),
		encoded_countries.values, encoded_provinces.values)
X = np.concatenate(arrays, axis=1)

# Retreive case for testing
test = X[-1]
test = np.expand_dims(test, axis=0)
test_target = y[-1]
y = y[:-1]
X = X[:-1]

# Remove NAN values
X = np.nan_to_num(X)
y = np.nan_to_num(y)

# Create and compile model
model = tf.keras.Sequential([
  layers.Dense(570, input_dim=570),
  layers.Dense((570 * 2), activation='relu'),
  layers.Dense((570 / 2), activation='relu'),
  layers.Dense(1)
])

# LR Decay
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)

# Compile and train
model.compile(optimizer=optimizer,
              loss='MSE',
              metrics=['MSE'])

checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, callbacks=[checkpointer])

# Predict with comparison to correct value
prediction = model.predict(test)
print("Predicted score: {}".format(prediction[0][0]))
print("Actual score: {}".format(test_target))
