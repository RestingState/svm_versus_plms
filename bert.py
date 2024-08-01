import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('type', choices=['train', 'evaluate'])

    args = parser.parse_args()

    df = pd.read_csv('bbc-news-data.csv', sep='\t')
    df = df.drop(['filename'], axis=1)

    X = df['title'] + df['content']

    category_categorical = pd.Categorical(df['category'])
    y = category_categorical.codes

    # Split the dataset into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the texts
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

    class TextDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    # Create train and validation datasets
    train_dataset = TextDataset(train_encodings, train_labels.tolist())
    val_dataset = TextDataset(val_encodings, val_labels.tolist())

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.type == 'train':
        # Load pre-trained BERT model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

        # Define the optimizer
        optimizer = AdamW(model.parameters(), lr=1e-5)

        # Fine-tuning the model
        model.to(device)
        model.train()

        for epoch in range(3):  # Number of training epochs
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                optimizer.zero_grad()
                inputs = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        # Save the fine-tuned model
        model.save_pretrained('fine-tuned-bert')
    elif args.type == 'evaluate':
        model = BertForSequenceClassification.from_pretrained('fine-tuned-bert')

        model.to(device)
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                inputs = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == inputs['labels']).sum().item()
                total += inputs['labels'].size(0)

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()