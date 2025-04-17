import json
import torch
from torch.utils.data import Dataset
import sentencepiece as spm

class TextDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer_model_path, max_len=128):
        """
        Initializing the dataset by loading prompt-completion pairs from a JSONL file,
        tokenizing the text using a SentencePiece model, and preparing samples for
        next-token prediction.
        """
        self.samples = []  # Storing tokenized samples
        self.tokenizer = spm.SentencePieceProcessor()  # Creating a SentencePiece tokenizer instance
        self.tokenizer.load(tokenizer_model_path)  # Loading the tokenizer model from file

        # Reading the JSONL file line by line
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # Checking if both 'prompt' and 'completion' fields are present
                if 'prompt' in data and 'completion' in data:
                    # Concatenating prompt and completion into one sequence
                    full_text = data['prompt'] + " " + data['completion']
                    # Tokenizing the text into a list of IDs
                    ids = self.tokenizer.EncodeAsIds(full_text)
                    # Keeping only sequences with more than one token and truncating to max_len
                    if len(ids) > 1:
                        self.samples.append(ids[:max_len])

    def __getitem__(self, idx):
        """
        Returning a single sample by splitting the token sequence into input and target.

        Args:
            idx (int): Index of the sample.

        Returns:
            input_ids (Tensor): Token IDs excluding the last token.
            target_ids (Tensor): Token IDs excluding the first token.
        """
        ids = self.samples[idx]
        input_ids = ids[:-1]  # Removing the last token for input
        target_ids = ids[1:]  # Removing the first token for target
        return torch.tensor(input_ids), torch.tensor(target_ids)

    def __len__(self):
        """
        Returning the number of samples in the dataset.
        """
        return len(self.samples)

def collate_fn(batch):
    """
    Collating a batch by padding sequences to equal length.

    Args:
        batch (list): List of (input_ids, target_ids) tuples.

    Returns:
        padded_inputs (Tensor): Padded input sequences [batch_size, max_seq_len].
        padded_targets (Tensor): Padded target sequences [batch_size, max_seq_len].
    """
    # Unpacking inputs and targets from batch
    inputs, targets = zip(*batch)
    # Padding input sequences to uniform length
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    # Padding target sequences to uniform length
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets
