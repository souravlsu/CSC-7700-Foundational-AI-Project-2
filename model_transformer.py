import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, 
                 num_layers=6, hidden_dim=512, dropout=0.1):
        super(TransformerLanguageModel, self).__init__()
        
        # Storing embedding dimension for scaling later
        self.embed_dim = embed_dim

        # Creating token embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Adding positional encodings to preserve sequence order
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        # Defining a single Transformer encoder layer and stacking it
        encoder_layers = TransformerEncoderLayer(
            embed_dim, num_heads, hidden_dim, dropout, batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        # Mapping the output to vocabulary logits
        self.fc = nn.Linear(embed_dim, vocab_size)

        # Initializing weights
        self.init_weights()

    def init_weights(self):
        # Initializing weights and biases uniformly
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask=None):
        """
        Forwarding inputs through embedding, positional encoding, Transformer layers, and output head.
        """
        # x: (batch_size, seq_len)
        x = self.embedding(x) * math.sqrt(self.embed_dim)  # Scaling embeddings
        x = self.pos_encoder(x)

        if mask is None:
            # Creating a causal mask to block future tokens during training
            mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)

        # Creating a padding mask to ignore padded tokens
        padding_mask = (x == 0).all(dim=-1)

        # Passing through the Transformer encoder
        output = self.transformer(x, mask=mask, src_key_padding_mask=padding_mask)

        # Mapping hidden states to vocabulary space
        logits = self.fc(output)

        # Returning logits and None to match interface with other models
        return logits, None

    def generate(self, prompt_ids, tokenizer, max_seq_length=50, temperature=1.0):
        """
        Generating tokens autoregressively from a given prompt.
        """
        self.eval()  # Setting model to evaluation mode
        generated = prompt_ids.copy()
        input_ids = torch.tensor(prompt_ids).unsqueeze(0).to(next(self.parameters()).device)

        for _ in range(max_seq_length):
            with torch.no_grad():
                # Forwarding through the model
                logits, _ = self.forward(input_ids)
                next_token_logits = logits[0, -1] / temperature  # Adjusting for temperature
                probs = torch.softmax(next_token_logits, dim=-1)  # Converting to probabilities
                next_token = torch.multinomial(probs, num_samples=1).item()  # Sampling next token
            
            # Stopping on EOS
            if next_token == tokenizer.eos_id():
                break

            # Appending new token and continuing
            generated.append(next_token)
            input_ids = torch.tensor([generated]).to(input_ids.device)

        # Decoding the generated IDs to text
        return tokenizer.DecodeIds(generated)

    def prompt(self, text_prompt, tokenizer, max_seq_length=50, temperature=1.0):
        """
        Encoding a text prompt and calling generate().
        """
        input_ids = tokenizer.EncodeAsIds(text_prompt)
        return self.generate(input_ids, tokenizer, max_seq_length, temperature)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Applying dropout to the positional encoding
        self.dropout = nn.Dropout(p=dropout)

        # Computing positional encodings once in log space
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # (d_model/2)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Applying sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Applying cosine to odd indices

        # Registering buffer so it's saved with the model but not trained
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Adding positional encodings to input embeddings
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)
