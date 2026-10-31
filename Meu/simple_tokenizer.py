# Simple tokenizer using manual input
import string

dataInput = input('Write the text input: ')

attentionWindow = []

def tokenize(data):
        for word in data.split():
                clean_words = word.lower().strip(string.punctuation)
                return clean_words

dataTokens = tokenize(dataInput)
attentionWindow.append(dataTokens)
print(attentionWindow)