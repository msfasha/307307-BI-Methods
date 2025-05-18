Using BERT (Bidirectional Encoder Representations from Transformers) for **sentiment analysis** involves fine-tuning a pre-trained BERT model on a labeled dataset of text samples with sentiment labels (e.g., positive, negative, neutral). Here's a step-by-step guide:

---

### **1. Choose a Pre-trained BERT Model**

Use a pre-trained BERT model from libraries like **Hugging Face's Transformers**:

```bash
pip install transformers
```

Popular options:

* `bert-base-uncased` – Lowercased English text.
* `distilbert-base-uncased` – Lighter and faster.

---

### **2. Prepare the Dataset**

Your dataset should have:

* **Text samples**: sentences or documents.
* **Sentiment labels**: numerical or categorical (e.g., 0 = negative, 1 = positive).

Example format:

```csv
text,label
"I love this movie!",1
"This was a terrible experience.",0
```

---

### **3. Tokenize the Text**

Use BERT's tokenizer to convert text into tokens that BERT understands:

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

---

### **4. Load BERT with a Classification Head**

You can use `BertForSequenceClassification`, which adds a linear layer on top of BERT for classification tasks:

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```

---

### **5. Train the Model**

Use a PyTorch or TensorFlow training loop. With Hugging Face's `Trainer` API, it's easier:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

---

### **6. Make Predictions**

Use the trained model to make predictions on new text:

```python
inputs = tokenizer("This is amazing!", return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=1)
```

---

### **7. Evaluate the Model**

Use standard metrics like accuracy, precision, recall, or F1-score:

```python
from sklearn.metrics import accuracy_score

accuracy_score(true_labels, predicted_labels)
```

---

### **Optional: Use Pipelines**

For quick experimentation:

```python
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")
result = sentiment_pipeline("I love using BERT for NLP tasks!")
print(result)
```

---