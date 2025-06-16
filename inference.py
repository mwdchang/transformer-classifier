import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


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


code_sample = """
def add(a, b):
    return a + b
"""

# Tokenize input
inputs = tokenizer(
    code_sample,
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
