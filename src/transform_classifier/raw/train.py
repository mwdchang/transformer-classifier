import os
import logging
logging.basicConfig(level=logging.DEBUG)

import sys
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from .simple_transformer_classifier import SimpleTransformerClassifier
from transform_classifier.common.source_data import SourceCodeDataset


def setup_model(data_folder, vocab_size, num_classes):

    # Model setup
    classifier_config = {
        "vocab_size": vocab_size,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "num_classes": num_classes
    }

    model = SimpleTransformerClassifier(
        vocab_size = classifier_config["vocab_size"],
        embed_dim = classifier_config["embed_dim"],
        num_heads = classifier_config["num_heads"],
        num_layers = classifier_config["num_layers"],
        num_classes = classifier_config["num_classes"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return (model, classifier_config)


def train(model, data_loader, num_epoch, lr):
    logging.info(f"num_epoch={num_epoch}, lr={lr}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optimizer setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logging.info(f"Epoch {epoch+1}, Loss: {total_loss/len(data_loader):.4f}")



def run_train(data_folder):
    # Input setup
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    dataset = SourceCodeDataset(data_folder, tokenizer)
    train_loader = DataLoader(dataset, batch_size=12, shuffle=True)

    model, classifier_config = setup_model(data_folder, 
        vocab_size=tokenizer.vocab_size, 
        num_classes=len(dataset.label_2_id))
    train(model, train_loader, num_epoch=10, lr=4e-5)

    checkpoint_folder = data_folder + ".model"
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    # Save model and tokenizer
    torch.save(model.state_dict(), os.path.join(checkpoint_folder, "model.pt"))
    tokenizer.save_pretrained(checkpoint_folder)
    with open(os.path.join(checkpoint_folder, "label_2_id.json"), "w") as f:
        json.dump(dataset.label_2_id, f, indent=2)
    with open(os.path.join(checkpoint_folder, "classifier_config.json"), "w") as f:
        json.dump(classifier_config, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python -m transform_classifier.raw.train <data_folder>")
    data_folder = sys.argv[1]
    run_train(data_folder)

