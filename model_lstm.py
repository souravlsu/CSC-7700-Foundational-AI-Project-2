import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super(LSTMLanguageModel, self).__init__()

        # Creating an embedding layer for converting token IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Defining the stacked LSTM layers for sequential modeling
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )

        # Mapping LSTM outputs to vocabulary space
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forwarding input sequences through embedding, LSTM, and output layers.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len)
            hidden (tuple or None): Optional (h_n, c_n) LSTM hidden states

        Returns:
            logits (Tensor): Output logits for vocabulary prediction
            hidden (tuple): Final hidden and cell states
        """
        # Embedding the input token IDs
        x = self.embedding(x)  # -> (batch_size, seq_len, embed_dim)

        # Passing through LSTM layers (with or without initial hidden state)
        if hidden is None:
            output, hidden = self.lstm(x)
        else:
            output, hidden = self.lstm(x, hidden)

        # Projecting LSTM output to vocab size
        logits = self.fc(output)  # -> (batch_size, seq_len, vocab_size)
        return logits, hidden

    def generate(self, prompt_ids, tokenizer, max_seq_length=50, temperature=1.0):
        """
        Generating tokens autoregressively from a prompt using the LSTM model.

        Args:
            prompt_ids (List[int]): Token ID list for the initial prompt
            tokenizer (SentencePieceProcessor): Tokenizer instance
            max_seq_length (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature for diversity

        Returns:
            str: Decoded generated text
        """
        self.eval()  # Setting model to evaluation mode
        input_ids = torch.tensor(prompt_ids).unsqueeze(0).to(next(self.parameters()).device)
        generated = prompt_ids.copy()
        hidden = None  # Initializing LSTM hidden state

        for _ in range(max_seq_length):
            with torch.no_grad():
                # Forwarding through the model
                logits, hidden = self.forward(input_ids, hidden)

                # Sampling next token using softmax and temperature scaling
                next_token_logits = logits[0, -1] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            # Stopping generation if EOS token is predicted
            if next_token == tokenizer.eos_id():
                break

            # Appending generated token and preparing next input
            generated.append(next_token)
            input_ids = torch.tensor([[next_token]]).to(input_ids.device)

        # Decoding token IDs into a human-readable string
        return tokenizer.DecodeIds(generated)

    def prompt(self, text_prompt, tokenizer, max_seq_length=50, temperature=1.0):
        """
        Encoding a string prompt and generating text from it.

        Args:
            text_prompt (str): Raw string input
            tokenizer (SentencePieceProcessor): Tokenizer instance

        Returns:
            str: Generated sequence from the prompt
        """
        # Encoding text prompt into token IDs
        input_ids = tokenizer.EncodeAsIds(text_prompt)

        # Generating sequence from encoded prompt
        return self.generate(input_ids, tokenizer, max_seq_length, temperature)
