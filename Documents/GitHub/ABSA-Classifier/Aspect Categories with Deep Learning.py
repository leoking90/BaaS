from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense
from keras.models import Sequential, Input, Model
import keras
from keras.layers import Conv1D, Conv2D,GlobalAveragePooling1D, MaxPooling1D, Flatten
import xml.etree.ElementTree as ET
restaurantAspectTree = ET.parse('Restaurants_Train.xml')
restaurantAspectRoot = restaurantAspectTree.getroot()
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

aspCatList = ['service','food','price','ambience','anecdotes/miscellaneous']
mlb = MultiLabelBinarizer(classes=(aspCatList))
#print(mlb.classes_)
        
def getCategory(sentences):
    categoryList = []
    for aspCategories in sentence.iter('aspectCategories'):
        #for aspectTerm in aspectTerms.iter('aspectTerm'): 
        aspCategory = aspCategories.findall('aspectCategory')#.get('term')
        for categoryElem in aspCategory:
            categories = categoryElem.get('category')
            categories = categories.lower()
            #categories = categories.split('#')
            print (categories)
            categoryList.append(categories)
            #print(aspects)
    return categoryList
        
aspCategoryList = []
reviewList = []
# Get Categories from corpus
for sentences in restaurantAspectRoot.iter('sentence'):
    for sentence in sentences.iter('sentence'):
        aspCategoryList.append(getCategory(sentence))
        reviewList.append(sentence.find('text').text)

labels = mlb.fit_transform(aspCategoryList)

MAX_NB_WORDS = 4504
MAX_SEQUENCE_LENGTH = 10
        
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(reviewList)
sequences = tokenizer.texts_to_sequences(reviewList)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

#labels = keras.utils.to_categorical(aspCategoryList)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
VALIDATION_SPLIT=0.2
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


embeddings_index = {}
with open('glove.6B.100d.txt', "r",  encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = 100
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
                        
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
print(embedding_layer)


model = Sequential()
"""model.add(LSTM(
        output_dim=50,
        return_sequences=False,
        input_length=10,
        input_dim=10))"""
model.add(Dense(86, activation='sigmoid', input_dim=10))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

# fit the model
model.fit(x_train, y_train, epochs=100, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
print('Accuracy: %f' % (accuracy*100))

"""model = Sequential()
e = Embedding(4505, 100, weights=[embedding_matrix], input_length=10, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(x_train, labels, epochs=100, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))"""
"""
labels_index = {}  # dictionary mapping label name to numeric id

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=2, batch_size=128)
"""

"""
sentence  = (sentence_length, word_size,word_vector)

batch_size =3044 

sentence_length = 10
word_vector_size = 4504
model = Sequential()
model.add(LSTM(
        output_dim=50,
        return_sequences=False,
        input_length=sentence_length,
        input_dim=word_vector_size))

 
model.add(Dense(output_dim=160, activation='tanh'))

model.compile(
      loss='binary_crossentropy',
      metrics=['accuracy'],
      optimizer='adam')

model.fit(x_train, y_train, nb_epoch=100, batch_size=batch_size,validation_data=(x_val, y_val))

## Evaluation the model

y_val_pred = model.predict_classes(x_val, verbose=0)


cm = confusion_matrix(y_val,y_val_pred)

print("Confusion Matrix \n\n")

print(cm)
print("\n")

accuracy_number = accuracy_score(y_val,y_val_pred)
precision_number = precision_score(y_val,y_val_pred,average="macro")
recall_number = recall_score(y_val,y_val_pred,average="macro")
f1_score_number = f1_score(y_val,y_val_pred,average="macro")

"""