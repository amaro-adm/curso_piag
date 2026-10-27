# Simple tokenizer using manual input
import re
import string

dataInput = input('Write the text input: ')
dataLenght = len(dataInput)

def tokenize(data):
        tokens = re.split(string.punctuation, data)

        tokens = [token.strip() for token in tokens if token.strip()]
        return tokens

dataTokens = tokenize(dataInput)