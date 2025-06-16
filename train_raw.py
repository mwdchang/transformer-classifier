import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from simple_transformer_classifier import SimpleTransformerClassifier
from source_data import SourceCodeDataset

data_folder = "tests"
checkpoint_folder = data_folder + ".model"
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)

# Input setup
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
dataset = SourceCodeDataset(data_folder, tokenizer)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model setup
model = SimpleTransformerClassifier(
    vocab_size = tokenizer.vocab_size,
    embed_dim = 128,
    num_heads = 4,
    num_layers = 2,
    num_classes = 2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Optimizer setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# Training
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")


# Save model and tokenizer
# model.save_pretrained("./my_model")
torch.save(model.state_dict(), os.path.join(checkpoint_folder, "model.pt"))
tokenizer.save_pretrained(checkpoint_folder)
with open(os.path.join(checkpoint_folder, "label_2_id.json"), "w") as f:
    json.dump(dataset.label_2_id, f, indent=2)

