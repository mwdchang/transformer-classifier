import sys
import os
import json
import torch
from transformers import AutoTokenizer 

from simple_transformer_classifier import SimpleTransformerClassifier


if len(sys.argv) != 3:
    raise ValueError("Usage: python inference_raw.py <path_to_checkpoint> <path_to_file>")

test_str = ""
with open(sys.argv[2], "r") as f:
    test_str = f.read()


checkpoint_folder = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(checkpoint_folder)

classifier_config = {}
with open(os.path.join(checkpoint_folder, "classifier_config.json"), "r") as f:
    classifier_config = json.load(f)

model = SimpleTransformerClassifier(
    vocab_size = classifier_config["vocab_size"],
    embed_dim = classifier_config["embed_dim"],
    num_heads = classifier_config["num_heads"],
    num_layers = classifier_config["num_layers"],
    num_classes = classifier_config["num_classes"]
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
# code_sample = """
# def add(a, b):
#     return a + b
# """

print("prediction is " + predict(test_str))
