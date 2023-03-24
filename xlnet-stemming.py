# Load necessary libraries
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW

import os

import pandas as pd
import urllib.request
import tarfile
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

# Download IMDB dataset
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
urllib.request.urlretrieve(url, "aclImdb_v1.tar.gz")

# Extract IMDB dataset
with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar:
    tar.extractall()

# Load IMDB dataset
imdb_df = pd.DataFrame(columns=['review', 'sentiment'])
for dataset in ['train', 'test']:
    for sentiment in ['pos', 'neg']:
        path = f'aclImdb/{dataset}/{sentiment}'
        for filename in os.listdir(path):
            with open(f'{path}/{filename}', 'r') as file:
                review = file.read()
            sentiment_value = 1 if sentiment == 'pos' else 0
            imdb_df = pd.concat([imdb_df, pd.DataFrame({'review': review, 'sentiment': sentiment_value}, index=[0])], ignore_index=True)

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Function to tokenize and stem the review text
def stem_review(review):
    stemmer = PorterStemmer()
    tokens = word_tokenize(review)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

# Apply the stem_review function to the 'review' column of the DataFrame
imdb_df['stemmed_review'] = imdb_df['review'].apply(stem_review)

# Display the DataFrame
print(imdb_df.head())
# Count the number of instances
num_instances = len(imdb_df)

# Count the number of positive and negative instances
num_positive = imdb_df['sentiment'].value_counts()[1]
num_negative = imdb_df['sentiment'].value_counts()[0]

# Print the results
print(f'Total number of instances: {num_instances}')
print(f'Number of positive instances: {num_positive}')
print(f'Number of negative instances: {num_negative}')

# Convert sentiment values to numeric
# Sonvert sentiment values to numeric
import numpy as np
from sklearn.model_selection import train_test_split
imdb_df['sentiment'] = pd.to_numeric(imdb_df['sentiment'], errors='coerce')
imdb_df = imdb_df.dropna()

# Preprocess dataset
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

input_ids = []
attention_masks = []

for review in imdb_df['stemmed_review']:
    encoded_dict = tokenizer.encode_plus(
                        review,                     
                        add_special_tokens = True,
                        max_length = 256,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(imdb_df['sentiment'].values, dtype=torch.long)

train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, 
                                                            random_state=42, test_size=0.2)
train_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=42, test_size=0.2)

batch_size = 16
epochs = 3
# Define XLNet model for sequence classification
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)

# Set the device to run the model on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Create the DataLoader for training data
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Create the DataLoader for validation data
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
# Define the training loop
for epoch in range(epochs):
    # Set the model to training mode
    model.train()

    # Track the training loss and accuracy
    total_train_loss = 0
    total_train_accuracy = 0

    # Iterate over the training data
    for step, batch in enumerate(train_dataloader):
        # Clear the gradients
        model.zero_grad()

        # Move the batch to the device
        batch_inputs = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch_inputs[0],
                  'attention_mask': batch_inputs[1],
                  'labels': batch_inputs[2]}

        # Perform the forward pass
        outputs = model(**inputs)
        loss = outputs[0]
        logits = outputs[1]

        # Perform the backward pass and update the parameters
        loss.backward()
        optimizer.step()

        # Track the training loss and accuracy
        total_train_loss += loss.item()
        preds = torch.argmax(logits, dim=1).flatten()
        accuracy = (preds == inputs['labels']).float().mean()
        total_train_accuracy += accuracy.item()

        # Print training progress
        if step % 50 == 0:
            print(f'Epoch: {epoch + 1}, Batch: {step}, Training Loss: {total_train_loss / (step + 1)}, Training Accuracy: {total_train_accuracy / (step + 1)}')

# Define the evaluation loop
model.eval()
total_val_loss = 0
total_val_accuracy = 0

# Iterate over the validation data
for batch in val_dataloader:
    # Move the batch to the device
    batch_inputs = tuple(t.to(device) for t in batch)
    inputs = {'input_ids': batch_inputs[0],
              'attention_mask': batch_inputs[1],
              'labels': batch_inputs[2]}

    # Disable gradient calculation
    with torch.no_grad():
        # Perform the forward pass
        outputs = model(**inputs)
        loss = outputs[0]
        logits = outputs[1]

    # Track the validation loss and accuracy
    total_val_loss += loss.item()
    preds = torch.argmax(logits, dim=1).flatten()
    accuracy = (preds == inputs['labels']).float().mean()
    total_val_accuracy += accuracy.item()

# Calculate the average validation loss and accuracy
avg_val_loss = total_val_loss / len(val_dataloader)
avg_val_accuracy = total_val_accuracy / len(val_dataloader)
print(f'Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}')


