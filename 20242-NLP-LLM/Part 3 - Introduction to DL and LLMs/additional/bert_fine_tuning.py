# Complete BERT Fine-tuning Tutorial for Text Classification
# ========================================================

# Required installations
# !pip install transformers datasets scikit-learn torch accelerate wandb

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ========================================================
# 1. DATA PREPARATION
# ========================================================

class DataProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        
    def load_and_preprocess(self):
        """Load and preprocess the CSV data"""
        # Load data
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} rows from {self.csv_path}")
        
        # Basic data exploration
        print("\nDataset Info:")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Shape: {self.df.shape}")
        
        # Check for missing values
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        
        # Handle missing values
        self.df = self.df.dropna(subset=['title', 'content', 'label'])
        
        # Combine title and content
        self.df["text"] = self.df["title"].astype(str) + " " + self.df["content"].astype(str)
        
        # Keep only necessary columns
        self.df = self.df[["text", "label"]]
        
        # Convert labels to integers if they're strings
        if self.df['label'].dtype == 'object':
            label_mapping = {label: idx for idx, label in enumerate(self.df['label'].unique())}
            self.df['label'] = self.df['label'].map(label_mapping)
            print(f"Label mapping: {label_mapping}")
        
        # Check label distribution
        print(f"\nLabel distribution:\n{self.df['label'].value_counts()}")
        
        return self.df
    
    def split_data(self, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            self.df, 
            test_size=test_size, 
            random_state=42, 
            stratify=self.df['label']
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=val_ratio, 
            random_state=42, 
            stratify=train_val_df['label']
        )
        
        print(f"\nData split:")
        print(f"Train: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df

# ========================================================
# 2. TOKENIZATION AND DATASET CREATION
# ========================================================

class BERTDatasetCreator:
    def __init__(self, model_name="bert-base-uncased", max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def tokenize_function(self, examples):
        """Tokenize the input text"""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None
        )
    
    def create_datasets(self, train_df, val_df, test_df):
        """Convert pandas DataFrames to Hugging Face Datasets"""
        # Convert to Dataset objects
        train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
        test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
        
        # Tokenize datasets
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)
        
        # Set format for PyTorch
        columns_to_return = ["input_ids", "attention_mask", "label"]
        train_dataset.set_format(type="torch", columns=columns_to_return)
        val_dataset.set_format(type="torch", columns=columns_to_return)
        test_dataset.set_format(type="torch", columns=columns_to_return)
        
        # Create DatasetDict for better organization
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        return dataset_dict

# ========================================================
# 3. MODEL AND TRAINING SETUP
# ========================================================

class BERTClassifier:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.trainer = None
        
    def load_model(self):
        """Load the pre-trained BERT model"""
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        return self.model
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def setup_trainer(self, train_dataset, val_dataset, output_dir="./results"):
        """Setup the Trainer with training arguments"""
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to=None,  # Disable wandb if not configured
            seed=42,
        )
        
        # Create the Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        return self.trainer
    
    def train(self):
        """Fine-tune the model"""
        print("Starting training...")
        self.trainer.train()
        print("Training completed!")
        
    def evaluate(self, test_dataset):
        """Evaluate the model on test set"""
        print("Evaluating on test set...")
        eval_results = self.trainer.evaluate(test_dataset)
        return eval_results
    
    def predict(self, test_dataset):
        """Get predictions for test set"""
        predictions = self.trainer.predict(test_dataset)
        return predictions
    
    def save_model(self, save_path):
        """Save the fine-tuned model"""
        self.trainer.save_model(save_path)
        print(f"Model saved to {save_path}")

# ========================================================
# 4. EVALUATION AND VISUALIZATION
# ========================================================

class ModelEvaluator:
    def __init__(self, predictions, true_labels, label_names=None):
        self.predictions = np.argmax(predictions.predictions, axis=1)
        self.true_labels = true_labels
        self.label_names = label_names or [f"Class {i}" for i in range(len(np.unique(true_labels)))]
        
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_names, 
                   yticklabels=self.label_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
    def classification_report(self):
        """Print detailed classification report"""
        from sklearn.metrics import classification_report
        
        report = classification_report(
            self.true_labels, 
            self.predictions, 
            target_names=self.label_names,
            digits=4
        )
        print("Detailed Classification Report:")
        print(report)
        
    def plot_training_history(self, trainer):
        """Plot training and validation metrics"""
        # Extract training history
        logs = trainer.state.log_history
        
        train_loss = [log['train_loss'] for log in logs if 'train_loss' in log]
        eval_loss = [log['eval_loss'] for log in logs if 'eval_loss' in log]
        eval_accuracy = [log['eval_accuracy'] for log in logs if 'eval_accuracy' in log]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(train_loss, label='Training Loss')
        ax1.plot(eval_loss, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(eval_accuracy, label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# ========================================================
# 5. INFERENCE PIPELINE
# ========================================================

class BERTInference:
    def __init__(self, model_path, tokenizer_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
    def predict_single(self, text):
        """Predict sentiment for a single text"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].tolist()
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        return results

# ========================================================
# 6. MAIN EXECUTION PIPELINE
# ========================================================

def main():
    # Step 1: Load and preprocess data
    print("=== Step 1: Data Loading and Preprocessing ===")
    data_processor = DataProcessor("amazon_reviews.csv")
    df = data_processor.load_and_preprocess()
    train_df, val_df, test_df = data_processor.split_data()
    
    # Step 2: Create datasets
    print("\n=== Step 2: Dataset Creation ===")
    dataset_creator = BERTDatasetCreator(max_length=256)  # Reduced for speed
    datasets = dataset_creator.create_datasets(train_df, val_df, test_df)
    
    # Step 3: Model setup and training
    print("\n=== Step 3: Model Training ===")
    classifier = BERTClassifier(num_labels=len(df['label'].unique()))
    classifier.load_model()
    classifier.setup_trainer(datasets['train'], datasets['validation'])
    classifier.train()
    
    # Step 4: Evaluation
    print("\n=== Step 4: Model Evaluation ===")
    test_results = classifier.evaluate(datasets['test'])
    print(f"Test Results: {test_results}")
    
    # Get predictions for detailed analysis
    predictions = classifier.predict(datasets['test'])
    
    # Detailed evaluation
    evaluator = ModelEvaluator(
        predictions, 
        datasets['test']['label'],
        label_names=['Negative', 'Positive']  # Adjust as needed
    )
    evaluator.classification_report()
    evaluator.plot_confusion_matrix()
    evaluator.plot_training_history(classifier.trainer)
    
    # Step 5: Save model
    print("\n=== Step 5: Saving Model ===")
    classifier.save_model("./fine_tuned_bert_model")
    
    # Step 6: Test inference
    print("\n=== Step 6: Testing Inference ===")
    inference = BERTInference("./fine_tuned_bert_model")
    
    test_texts = [
        "This product is amazing! I love it so much.",
        "Terrible quality, waste of money.",
        "It's okay, nothing special but not bad either."
    ]
    
    for text in test_texts:
        result = inference.predict_single(text)
        print(f"Text: {text}")
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("---")

# ========================================================
# 7. HYPERPARAMETER TUNING (OPTIONAL)
# ========================================================

def hyperparameter_search():
    """Example of hyperparameter tuning with Optuna"""
    try:
        import optuna
        
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-4)
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
            warmup_steps = trial.suggest_int('warmup_steps', 100, 1000)
            
            # Setup model and training
            classifier = BERTClassifier()
            classifier.load_model()
            
            # Modified training arguments
            training_args = TrainingArguments(
                output_dir=f"./trial_{trial.number}",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                warmup_steps=warmup_steps,
                num_train_epochs=1,  # Reduced for speed
                evaluation_strategy="epoch",
                save_strategy="no",
                logging_steps=100,
                report_to=None,
            )
            
            trainer = Trainer(
                model=classifier.model,
                args=training_args,
                train_dataset=datasets['train'],
                eval_dataset=datasets['validation'],
                compute_metrics=classifier.compute_metrics,
            )
            
            trainer.train()
            eval_results = trainer.evaluate()
            
            return eval_results['eval_f1']
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        
        print("Best hyperparameters:", study.best_params)
        print("Best F1 score:", study.best_value)
        
    except ImportError:
        print("Optuna not installed. Skipping hyperparameter tuning.")

# ========================================================
# Run the main pipeline
# ========================================================

if __name__ == "__main__":
    main()
    
    # Uncomment to run hyperparameter tuning
    # hyperparameter_search()