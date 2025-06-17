import sys
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


if len(sys.argv) != 2:
    raise ValueError("Usage: python inference.py <path_to_file>")

test_str = ""
with open(sys.argv[1], "r") as f:
    test_str = f.read()

checkpoint_folder = "tests.model"

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint_folder,
    use_safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_folder)

id_2_label = {}
with open(os.path.join(checkpoint_folder, "label_2_id.json"), "r") as f:
    id_2_label = json.load(f)

model.eval()  # Set to eval mode


# code_sample = """
# def add(a, b):
#     return a + b
# """

# Tokenize input
inputs = tokenizer(
    test_str,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=512
)


# Forward pass (inference)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

print(f"Predicted class: {id_2_label[str(predicted_class)]}")
