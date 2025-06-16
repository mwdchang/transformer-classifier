from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import json

class SourceCodeDataset(Dataset):
    def __init__(self, folder_path, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = 0
        self.label_2_id = {}

        for label, class_dir in enumerate(os.listdir(folder_path)):
            self.num_classes = self.num_classes + 1
            self.label_2_id[label] = class_dir
        
            class_path = os.path.join(folder_path, class_dir)
            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                self.samples.append((code, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code, label = self.samples[idx]
        encoding = self.tokenizer(
            code,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }

