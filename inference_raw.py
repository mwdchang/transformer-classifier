import os
import json
import torch
from transformers import AutoTokenizer 

from simple_transformer_classifier import SimpleTransformerClassifier

checkpoint_folder = "tests.model"

tokenizer = AutoTokenizer.from_pretrained(checkpoint_folder)
model = SimpleTransformerClassifier(
    vocab_size = tokenizer.vocab_size,
    embed_dim = 128,
    num_heads = 4,
    num_layers = 2,
    num_classes = 2
)

# Load model state
model.load_state_dict(torch.load(os.path.join(checkpoint_folder, "model.pt")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

id_2_label = {}
with open(os.path.join(checkpoint_folder, "label_2_id.json"), "r") as f:
    id_2_label = json.load(f)


# Inference example
def predict(code_snippet):
    encoding = tokenizer(
        code_snippet,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        predicted_class = torch.argmax(logits, dim=1).item()
        return id_2_label[str(predicted_class)]

# Example usage
print("prediction is " + predict("def foo(x): return x + 1"))
