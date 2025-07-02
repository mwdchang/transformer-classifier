import sys
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
logging.basicConfig(level=logging.DEBUG)


def setup_model(checkpoint_folder):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_folder,
        use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_folder)
    id_2_label = {}
    with open(os.path.join(checkpoint_folder, "label_2_id.json"), "r") as f:
        id_2_label = json.load(f)

    return (model, tokenizer, id_2_label)
    
    
def predict(model, tokenizer, id_2_label, text):
    model.eval()  # Set to eval mode
    
    # Tokenize input
    inputs = tokenizer(
        text,
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
    return id_2_label[str(predicted_class)]


   
def run_inference(checkpoint_folder, text):
    model, tokenizer, id_2_label = setup_model(checkpoint_folder)
    result = predict(model, tokenizer, id_2_label, text)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python -m transform_classifier.pretrained.inference <path_to_check_point> <path_to_file>")

    checkpoint_folder = sys.argv[1]
    text = ""
    with open(sys.argv[2], "r") as f:
        text = f.read()

    label = run_inference(checkpoint_folder, text)
    logging.info(f"Prediction={label}")
