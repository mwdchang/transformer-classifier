import os
import sys
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
)
from transform_classifier.common.source_data import SourceCodeDataset
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.DEBUG)


def setup_model(data_folder, num_classes):
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base", 
        num_labels=num_classes,
        use_safetensors=True # Needs torch v2.6+ to use torch.load()
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def train(model, data_loader, num_epoch, lr):
    optimizer = AdamW(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        logging.info(f"Epoch {epoch+1} - Loss: {total_loss / len(data_loader):.4f}")



def run_train(data_folder):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    dataset = SourceCodeDataset(data_folder, tokenizer)
    train_loader = DataLoader(dataset, batch_size=12, shuffle=True)

    model = setup_model(data_folder, num_classes=len(dataset.label_2_id))
    train(model, train_loader, num_epoch=1, lr=1e-5)

    checkpoint_folder = data_folder + ".model"
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
   
    # Save model and tokenizer
    model.save_pretrained(checkpoint_folder)
    tokenizer.save_pretrained(checkpoint_folder)
    with open(os.path.join(checkpoint_folder, "label_2_id.json"), "w") as f:
        json.dump(dataset.label_2_id, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python -m transform_classifier.pretrained.train <data_folder>")
    data_folder = sys.argv[1]
    run_train(data_folder)
   
