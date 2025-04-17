import torch
import torch.nn as nn

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=1):
        super(RNNLanguageModel, self).__init__()
        
        # Creating an embedding layer for converting token IDs to vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Defining the RNN layer for sequential processing
        self.rnn = nn.RNN(
            embed_dim, hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            nonlinearity='tanh'
        )
        
        # Mapping RNN output to vocabulary space
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forwarding inputs through embedding, RNN, and output layers.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len)
            hidden (Tensor or None): Optional hidden state for RNN

        Returns:
            logits (Tensor): Output logits for each token position
            hidden (Tensor): Final hidden state
        """
        # Converting input token IDs to embeddings
        x = self.embedding(x)  # -> (batch_size, seq_len, embed_dim)

        # Processing sequence through RNN
        if hidden is None:
            output, hidden = self.rnn(x)  # No initial hidden state
        else:
            output, hidden = self.rnn(x, hidden)  # Using provided hidden state

        # Mapping RNN output to vocabulary logits
        logits = self.fc(output)  # -> (batch_size, seq_len, vocab_size)
        return logits, hidden

    def generate(self, prompt_ids, tokenizer, max_seq_length=50, temperature=1.0):
        """
        Generating tokens autoregressively from a prompt using the RNN.

        Args:
            prompt_ids (List[int]): List of token IDs for the prompt
            tokenizer (SentencePieceProcessor): Tokenizer instance
            max_seq_length (int): Maximum length of generated sequence
            temperature (float): Sampling temperature

        Returns:
            str: Decoded text from generated token IDs
        """
        self.eval()  # Setting model to evaluation mode
        input_ids = torch.tensor(prompt_ids).unsqueeze(0).to(next(self.parameters()).device)
        generated = prompt_ids.copy()
        hidden = None  # Initializing hidden state

        for _ in range(max_seq_length):
            with torch.no_grad():
                # Forwarding through the model
                logits, hidden = self.forward(input_ids, hidden)

                # Sampling next token from probability distribution
                next_token_logits = logits[0, -1] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            # Stopping generation if EOS is produced
            if next_token == tokenizer.eos_id():
                break

            # Appending generated token and continuing
            generated.append(next_token)
            input_ids = torch.tensor([[next_token]]).to(input_ids.device)

        # Decoding generated token IDs to string
        return tokenizer.DecodeIds(generated)

    def prompt(self, text_prompt, tokenizer, max_seq_length=50, temperature=1.0):
        """
        Encoding a prompt string and generating text from it.

        Args:
            text_prompt (str): Input text string
            tokenizer (SentencePieceProcessor): Tokenizer instance

        Returns:
            str: Generated text
        """
        # Converting text prompt to token IDs
        input_ids = tokenizer.EncodeAsIds(text_prompt)

        # Generating tokens based on the input IDs
        return self.generate(input_ids, tokenizer, max_seq_length, temperature)
