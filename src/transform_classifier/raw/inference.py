import logging
logging.basicConfig(level=logging.DEBUG)

import sys
import os
import json
import torch
from transformers import AutoTokenizer 
from .simple_transformer_classifier import SimpleTransformerClassifier


def setup_model(checkpoint_folder):
    classifier_config = {}
    with open(os.path.join(checkpoint_folder, "classifier_config.json"), "r") as f:
        classifier_config = json.load(f)

    id_2_label = {}
    with open(os.path.join(checkpoint_folder, "label_2_id.json"), "r") as f:
        id_2_label = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_folder)
    model = SimpleTransformerClassifier(
        vocab_size = classifier_config["vocab_size"],
        embed_dim = classifier_config["embed_dim"],
        num_heads = classifier_config["num_heads"],
        num_layers = classifier_config["num_layers"],
        num_classes = classifier_config["num_classes"]
    )

    model.load_state_dict(torch.load(os.path.join(checkpoint_folder, "model.pt")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return (model, tokenizer, id_2_label)


def predict(model, tokenizer, id_2_label, text):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoding = tokenizer(
        text,
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


def run_inference(checkpoint_folder, text):
    model, tokenizer, id_2_label = setup_model(checkpoint_folder)
    result = predict(model, tokenizer, id_2_label, text)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python -m transform_classifier.raw.inference <path_to_check_point> <path_to_file>")

    checkpoint_folder = sys.argv[1]

    text = ""
    with open(sys.argv[2], "r") as f:
        text = f.read()
    label = run_inference(checkpoint_folder, text)
    logging.info(f"Prediction={label}")
