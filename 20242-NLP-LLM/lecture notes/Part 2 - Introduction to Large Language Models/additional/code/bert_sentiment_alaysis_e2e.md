**end-to-end BERT sentiment analysis example** using the Hugging Face `Transformers` library, focused on ease of understanding and reproducibility. 

We'll use the IMDb dataset (binary sentiment: positive or negative).

---

### ✅ Prerequisites

Install the necessary packages:

```bash
pip install transformers datasets torch scikit-learn
```

---

### 🧠 Step-by-Step Code (End-to-End)

```python
# 1. Import Libraries
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import torch

# 2. Load IMDb Dataset
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

# 3. Load Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 4. Tokenize the Data
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_data = train_data.map(tokenize, batched=True, batch_size=512)
test_data = test_data.map(tokenize, batched=True, batch_size=512)

# 5. Set Format for PyTorch
train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 6. Load Pretrained Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 7. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./bert-imdb-results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
)

# 8. Define Compute Metrics Function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# 9. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
)

# 10. Train the Model
trainer.train()

# 11. Evaluate
eval_result = trainer.evaluate()
print(eval_result)

# 12. Inference Example
text = "This movie was absolutely wonderful and touching."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()

sentiment = "positive" if predicted_class == 1 else "negative"
print(f"Sentiment: {sentiment}")
```

---

### 🧪 Sample Output

```text
{'eval_loss': 0.315, 'eval_accuracy': 0.873, ...}
Sentiment: positive
```

---

### 🚀 Tips

* You can use `distilbert-base-uncased` for faster training.
* Fine-tune for more epochs if accuracy is low.
* Use `Trainer.push_to_hub()` to save and share your model.

Would you like this as a ready-to-run notebook file (`.ipynb`)?
