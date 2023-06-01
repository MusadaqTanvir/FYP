import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer 
import numpy as np
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def steming(word):
    return stemmer.stem(word.lower()) 
def bag_of_words(tokenized_data, array):
    tokenized_data = [steming(word) for word in tokenized_data]
    bag = np.zeros(len(array),dtype=np.float32)
    for idx, word in enumerate(array):
        if word in tokenized_data:
            bag[idx] = 1.0
    return bag
