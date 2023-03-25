import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random
import json
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('wordnet')

with open('intents.json') as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore = ['?', '.', ',', '!']

lemmatizer = WordNetLemmatizer()

for intent in data['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents with their corresponding class
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize, convert to lower case, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    # initialize bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create the bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle the features and make numpy array
random.shuffle(training)
training = np.array(training)

# create training and testing lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sentences = np.array([
    "Hello, how are you?",
    "What's up?",
    "I need some help.",
    "Can you help me please?",
    "Thank you very much!"
])

trainLabels = [
    "Hello, how are you?",
    "What's up?",
    "I need some help.",
    "Can you help me please?",
    "Thank you very much!"
]

testLabels = [
    "Hello, how are you?",
    "What's up?",
    "I need some help.",
    "Can you help me please?",
    "Thank you very much!"
]


text_list = sentences.tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_list)
sequences = tokenizer.texts_to_sequences(text_list)

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

# compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

def preprocess_text(text):
    # Tokenize the input text using the tokenizer
    tokens = tokenizer.texts_to_sequences([text])[0]
    
    # Pad the token sequence to a fixed length
    max_length = 20
    padded_tokens = pad_sequences([tokens], maxlen=max_length, padding='post', truncating='post')
    
    return padded_tokens

def chatbot_response(model, tokenizer, input_text):
    response = ""
    label_encoder = LabelEncoder()
    all_labels = set(trainLabels + testLabels) # combine train and test labels
    label_encoder.fit(list(all_labels))
    # Encode the categorical labels into numerical labels
    # transform the labels
    train_labels_encoded = label_encoder.transform(trainLabels)
    test_labels_encoded = label_encoder.transform(testLabels)

    with open('intents.json') as file:
        intents = json.load(file)

    # Preprocess the user input text
    input_text = preprocess_text(input_text)

    # Convert the preprocessed input text to a sequence of integers
    input_seq = tokenizer.texts_to_sequences([str(input_text)])[0]

    # Pad the input sequence so that it has the same length as the model's input shape
    input_seq = pad_sequences([input_seq], maxlen=model.input_shape[1], padding='post')

    # Use the model to get a prediction for the input sequence
    predicted_class = np.argmax(model.predict(input_seq))

    # Get the corresponding intent tag for the predicted class
    for intent in intents['intents']:
        if intent['tag'] == label_encoder.inverse_transform([predicted_class])[0]:
            response = random.choice(intent['responses'])
            break
    return response

# Load the trained model and tokenizer
model = load_model('chatbot_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Get user input and get chatbot response
while True:
    user_input = input('You: ')
    if user_input.lower() == 'exit':
        break
    response = chatbot_response(model, tokenizer, user_input)
    print('Chatbot:', response)
