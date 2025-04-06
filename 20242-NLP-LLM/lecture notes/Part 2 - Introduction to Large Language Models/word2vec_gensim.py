# pip install gensim
# pip install nltk

import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize


nltk.download('punkt')  # Ensure punkt tokenizer is downloaded

# Sample corpus
sentences = [
"Large language models are transforming business applications",
"Natural language processing helps computers understand human language",
"Word embeddings capture semantic relationships between words",
"Neural networks learn distributed representations of words",
"Businesses use language models for various applications",
"Customer service can be improved with language technology",
"Modern language models require significant computing resources",
"Language models can generate human-like text for businesses"]

# Tokenize the sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
# Train Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, 
 vector_size=100,  # Embedding dimension
 window=5,         # Context window size
 min_count=1,      # Minimum word frequency
 workers=4)        # Number of threads
# Save the model
model.save("word2vec.model")

# Find the most similar words to "language"
similar_words = model.wv.most_similar("language", topn=5)
print("Words most similar to 'language':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")

# Vector for a specific word
word_vector = model.wv["business"]
print(f"\nVector for 'business' (first 10 dimensions):\n{word_vector[:10]}")

# Word analogies
analogy_result = model.wv.most_similar(positive=["business", "language"], 
                                       negative=["models"], 
                                       topn=3)
print("\nAnalogy results:")
for word, similarity in analogy_result:
    print(f"{word}: {similarity:.4f}")

