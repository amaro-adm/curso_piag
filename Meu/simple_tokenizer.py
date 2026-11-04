# Simple tokenizer using manual input
import string

dataInput = input('Write the text input: ')

attentionWindow = []

def tokenize(data):
        clean_words = []
        for word in data.split():
                clean_words.append(word.lower().strip(string.punctuation))
        return clean_words

dataTokens = tokenize(dataInput)
attentionWindow.append(dataTokens)
print(dataTokens)