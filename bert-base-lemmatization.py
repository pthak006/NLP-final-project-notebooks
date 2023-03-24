import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#import necessary libraries for lemmatizaation
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4') 

# define function to perform lemmatization
def preprocess_and_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    
    # Join the lemmatized tokens back into a single string
    lemmatized_text = ' '.join(lemmatized_tokens)
    
    return lemmatized_text

# wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# tar -xf aclImdb_v1.tar.gz

import os

# Define a function to load the IMDB dataset from disk
def load_imdb_dataset():
    imdb_dir = 'aclImdb'
    train_dir = os.path.join(imdb_dir, 'train')
    test_dir = os.path.join(imdb_dir, 'test')

    train_texts = []
    train_labels = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    train_texts.append(f.read())
                train_labels.append(0 if label_type == 'neg' else 1)

    test_texts = []
    test_labels = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(test_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    test_texts.append(f.read())
                test_labels.append(0 if label_type == 'neg' else 1)

    return train_texts, train_labels, test_texts, test_labels

# Load the IMDB dataset from disk
train_texts, train_labels, test_texts, test_labels = load_imdb_dataset()

# Apply the preprocess_and_lemmatize function to the training and testing texts
train_texts = [preprocess_and_lemmatize(text) for text in train_texts]
test_texts = [preprocess_and_lemmatize(text) for text in test_texts]
from sklearn.model_selection import train_test_split

# Combine the training and testing texts and labels
texts = train_texts + test_texts
labels = train_labels + test_labels

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Define a function to tokenize the input data
def tokenize_data(texts, labels, tokenizer, max_len):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels

# Tokenize the training and testing data
train_input_ids, train_attention_masks, train_labels = tokenize_data(train_texts, train_labels, tokenizer, max_len=512)
test_input_ids, test_attention_masks, test_labels = tokenize_data(test_texts, test_labels, tokenizer, max_len=512)

# Load the pre-trained BERT model and adjust the number of labels
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

# Set the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_input_ids) * 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Define a function to train the model
def train_model(model, train_input_ids, train_attention_masks, train_labels, test_input_ids, test_attention_masks, test_labels, optimizer, scheduler, num_epochs=3):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    train_data = torch.utils.data.TensorDataset(train_input_ids, train_attention_masks, train_labels)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

    test_data = torch.utils.data.TensorDataset(test_input_ids, test_attention_masks, test_labels)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for batch in train_loader:
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)

            optimizer.zero_grad()
            outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_attention_masks)
            logits = outputs.logits
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            train_acc += torch.sum(preds == batch_labels).item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

        model.eval()
        eval_loss, eval_acc = 0.0, 0.0
        for batch in test_loader:
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_attention_masks)
                logits = outputs.logits
                loss = loss_fn(logits, batch_labels)

            eval_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            eval_acc += torch.sum(preds == batch_labels).item()

        eval_loss /= len(test_loader)
        eval_acc /= len(test_loader.dataset)

        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {train_loss:.3f} | Training Accuracy: {train_acc:.3f}')
        print(f'Validation Loss: {eval_loss:.3f} | Validation Accuracy: {eval_acc:.3f}\n')

train_model(model, train_input_ids, train_attention_masks, train_labels, test_input_ids, test_attention_masks, test_labels, optimizer, scheduler, num_epochs=3)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

test_data = torch.utils.data.TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False)

model.eval()
test_acc = 0.0
for batch in test_loader:
    batch_input_ids = batch[0].to(device)
    batch_attention_masks = batch[1].to(device)
    batch_labels = batch[2].to(device)

    with torch.no_grad():
        outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_attention_masks)
        logits = outputs.logits

    _, preds = torch.max(logits, dim=1)
    test_acc += torch.sum(preds == batch_labels).item()

test_acc /= len(test_loader.dataset)

print(f'Test Accuracy: {test_acc:.3f}')
