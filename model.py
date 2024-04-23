import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load sentiment dataset
df = pd.read_csv('sentiment.csv')
df = df.head(5000)

# Data Preprocessing
def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a non-null string
        return text.lower()
    else:
        return ""  # Return an empty string for NaN values

df['Review_Title'] = df['Review_Title'].apply(preprocess_text)

# Convert sentiments to numerical values
sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Function for training and evaluation
def train_and_evaluate(model, train_loader, test_dataset, num_epochs, epsilon=None):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Save the average loss for the epoch
        losses.append(np.mean(epoch_losses))

        # Evaluation
        model.eval()
        with torch.no_grad():
            logits = model(test_dataset.tensors[0], attention_mask=test_dataset.tensors[1])
            predictions = np.argmax(logits.logits.numpy(), axis=1)
            accuracy = accuracy_score(test_dataset.tensors[2], predictions)
            f1 = f1_score(test_dataset.tensors[2], predictions, average='macro')
            accuracies.append(accuracy)
            print(f"Accuracy after Epoch {epoch + 1}: {accuracy:.4f}, Macro F1 Score: {f1:.4f}")

    return losses, f1, accuracies, predictions

# Traditional BERT Approach
tokenizer_bert = BertTokenizer.from_pretrained('bert-large-uncased')
train_encodings_bert = tokenizer_bert(list(train_df['Review_Title']),
                                      truncation=True, padding=True, return_tensors='pt')
test_encodings_bert = tokenizer_bert(list(test_df['Review_Title']),
                                     truncation=True, padding=True, return_tensors='pt')

train_dataset_bert = TensorDataset(train_encodings_bert['input_ids'],
                                   train_encodings_bert['attention_mask'],
                                   torch.tensor(train_df['Sentiment'].values))

test_dataset_bert = TensorDataset(test_encodings_bert['input_ids'],
                                  test_encodings_bert['attention_mask'],
                                  torch.tensor(test_df['Sentiment'].values))

model_bert = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=len(df['Sentiment'].unique()))
losses_bert, f1_bert, accuracies_bert, predictions_bert = train_and_evaluate(model_bert, DataLoader(train_dataset_bert, batch_size=32, shuffle=True), test_dataset_bert, num_epochs=3)

# Whole Word Masking BERT Approach
tokenizer_wwm = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
train_encodings_wwm = tokenizer_wwm(list(train_df['Review_Title']),
                                    truncation=True, padding=True, return_tensors='pt')
test_encodings_wwm = tokenizer_wwm(list(test_df['Review_Title']),
                                   truncation=True, padding=True, return_tensors='pt')

train_dataset_wwm = TensorDataset(train_encodings_wwm['input_ids'],
                                  train_encodings_wwm['attention_mask'],
                                  torch.tensor(train_df['Sentiment'].values))

test_dataset_wwm = TensorDataset(test_encodings_wwm['input_ids'],
                                 test_encodings_wwm['attention_mask'],
                                 torch.tensor(test_df['Sentiment'].values))

model_wwm = BertForSequenceClassification.from_pretrained('bert-large-uncased-whole-word-masking', num_labels=len(df['Sentiment'].unique()))
losses_wwm, f1_wwm, accuracies_wwm, predictions_wwm = train_and_evaluate(model_wwm, DataLoader(train_dataset_wwm, batch_size=32, shuffle=True), test_dataset_wwm, num_epochs=3)

# Whole Word Masking BERT with Adversarial Training
model_wwm_adv = BertForSequenceClassification.from_pretrained('bert-large-uncased-whole-word-masking', num_labels=len(df['Sentiment'].unique()))
losses_wwm_adv, f1_wwm_adv, accuracies_wwm_adv, predictions_wwm_adv = train_and_evaluate(model_wwm_adv, DataLoader(train_dataset_wwm, batch_size=32, shuffle=True), test_dataset_wwm, num_epochs=3, epsilon=0.1)

# Plotting the loss vs epoch
plt.plot(range(1, 4), losses_bert, label='Traditional BERT')
plt.plot(range(1, 4), losses_wwm, label='Whole Word Masking BERT')
plt.plot(range(1, 4), losses_wwm_adv, label='Adversarial Training BERT')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Epoch')
plt.legend()
plt.show()

# Print final metrics
print(f"Final Metrics - Traditional BERT: Macro F1 Score={f1_bert:.4f}, Accuracy={accuracies_bert[-1]:.4f}")
print(f"Final Metrics - Whole Word Masking BERT: Macro F1 Score={f1_wwm:.4f}, Accuracy={accuracies_wwm[-1]:.4f}")
print(f"Final Metrics - Whole Word Masking BERT with Adversarial Training: Macro F1 Score={f1_wwm_adv:.4f}, Accuracy={accuracies_wwm_adv[-1]:.4f}")

# Model Prediction of Whole Word Masking BERT with Adversarial Training

review_example = "Chevy Silverado is amazing!"
sentiment_mapping = {2: 'positive', 1: 'neutral', 0: 'negative'}
input_encoding = tokenizer_wwm(review_example, truncation=True, padding=True, return_tensors='pt')
predicted_sentiment = sentiment_mapping[np.argmax(model_wwm_adv(**input_encoding).logits.detach().numpy())]
print(f"The given review is: {predicted_sentiment}")
