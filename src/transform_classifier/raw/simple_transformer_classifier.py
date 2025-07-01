import torch
import torch.nn as nn

class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, max_length=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.max_length = max_length


    def forward(self, input_ids, attention_mask=None):
        # input_ids shape: (batch_size, seq_len)
        embeddings = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        
        # Transformer expects seq_len first: (seq_len, batch_size, embed_dim)
        embeddings = embeddings.permute(1, 0, 2)
        
        # Optionally apply attention mask if provided
        if attention_mask is not None:
            # transformer expects mask with shape (seq_len, seq_len)
            # Here, we create a key_padding_mask instead for padding positions
            # key_padding_mask: (batch_size, seq_len), True for padded tokens
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        encoded = self.transformer_encoder(embeddings, src_key_padding_mask=key_padding_mask)
        
        # Take encoding of the first token (like [CLS]) for classification
        encoded = encoded[0]  # (batch_size, embed_dim)
        
        logits = self.classifier(encoded)
        return logits

