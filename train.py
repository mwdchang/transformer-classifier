import os
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
)
from source_data import SourceCodeDataset
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

dataset = SourceCodeDataset("test", tokenizer)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base", 
    num_labels=dataset.num_classes,
    use_safetensors=True # Needs torch v2.6+ to use torch.load()
)
model.to(device)


optimizer = AdamW(model.parameters(), lr=1e-5)
model.train()
for epoch in range(3):  # Number of epochs
    total_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")


# Save model and tokenizer
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")
with open(os.path.join("./my_model", "label2id.json"), "w") as f:
    json.dump(dataset.label_2_id, f, indent=2)

