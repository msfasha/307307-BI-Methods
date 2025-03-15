import nltk
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample movie reviews dataset
reviews = [
    {"text": "This movie was absolutely fantastic! Great acting and storyline.", "sentiment": 1},
    {"text": "I loved this film. The characters were so well developed.", "sentiment": 1},
    {"text": "Amazing cinematography and directing. One of the best films I've seen.", "sentiment": 1},
    {"text": "The acting was good but the story was too predictable.", "sentiment": 0},
    {"text": "Terrible movie. I wasted two hours of my life.", "sentiment": 0},
    {"text": "The special effects were amazing but everything else was boring.", "sentiment": 0},
    {"text": "I enjoyed the action sequences but the dialogue was poorly written.", "sentiment": 0},
    {"text": "Brilliant performance by the lead actor! Highly recommended.", "sentiment": 1},
    {"text": "So disappointing. The trailer was better than the actual movie.", "sentiment": 0},
    {"text": "A masterpiece of modern cinema. I was captivated throughout.", "sentiment": 1}
]

# Create DataFrame
df = pd.DataFrame(reviews)
print("Original Data:")
print(df.head())
print("\n")

# Step 1: Tokenization
def tokenize_text(text):
    tokens = word_tokenize(text)
    print(f"After tokenization: {tokens}")
    return tokens

# Step 2: Normalization
def normalize_tokens(tokens):
    # Convert to lowercase and remove punctuation
    normalized = [re.sub(r'[^\w\s]', '', token.lower()) for token in tokens]
    normalized = [token for token in normalized if token]  # Remove empty strings
    print(f"After normalization: {normalized}")
    return normalized

# Step 3: Remove stop words
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered = [token for token in tokens if token not in stop_words]
    print(f"After stopword removal: {filtered}")
    return filtered

# Step 4a: Stemming
def stem_tokens(tokens):
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(token) for token in tokens]
    print(f"After stemming: {stemmed}")
    return stemmed

# Step 4b: Lemmatization
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    print(f"After lemmatization: {lemmatized}")
    return lemmatized

# Complete preprocessing pipeline
def preprocess_text(text, use_stemming=True):
    tokens = tokenize_text(text)
    normalized_tokens = normalize_tokens(tokens)
    no_stopwords = remove_stopwords(normalized_tokens)
    
    if use_stemming:
        return stem_tokens(no_stopwords)
    else:
        return lemmatize_tokens(no_stopwords)

# Demonstrate the preprocessing pipeline on one example
print("PREPROCESSING PIPELINE DEMONSTRATION:")
sample_text = df['text'][0]
print(f"Original text: '{sample_text}'")
processed_tokens = preprocess_text(sample_text, use_stemming=True)
print("\n")

# Process all reviews
df['processed_text'] = df['text'].apply(lambda x: ' '.join(preprocess_text(x, use_stemming=True)))
print("Processed Reviews:")
print(df[['text', 'processed_text']].head())
print("\n")

# Step 5: Bag of Words representation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed_text'])

# Show vocabulary
print("Bag of Words Vocabulary:")
print(vectorizer.get_feature_names_out())
print("\n")

# Show BOW representation for first few documents
print("BOW Matrix (First 3 Documents):")
bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(bow_df.head(3))
print("\n")

# Step 6: Simple classifier
X_train, X_test, y_train, y_test = train_test_split(
    X, df['sentiment'], test_size=0.3, random_state=42
)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate
print("SENTIMENT CLASSIFICATION RESULTS:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Demonstrate classification on new examples
def predict_sentiment(text):
    # Preprocess
    processed = ' '.join(preprocess_text(text, use_stemming=True))
    # Transform to BOW
    bow = vectorizer.transform([processed])
    # Predict
    prediction = clf.predict(bow)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment

print("\nTRY WITH NEW EXAMPLES:")
new_reviews = [
    "This movie exceeded all my expectations. The plot was incredible!",
    "What a waste of time. The worst movie I've seen this year."
]

for review in new_reviews:
    sentiment = predict_sentiment(review)
    print(f"Review: '{review}'")
    print(f"Predicted sentiment: {sentiment}\n")