import nltk
from nltk.tokenize import word_tokenize

# Download the necessary NLTK data files

nltk.download('punkt')

# Sample sentence
sentence = "Natural Language Processing with NLTK is fun!"

# Tokenize the sentence
tokens = word_tokenize(sentence)

# Print the tokens
print(tokens)