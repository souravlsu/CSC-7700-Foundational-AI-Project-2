import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sentencepiece as spm
from nltk.translate.bleu_score import corpus_bleu
from dataset import TextDataset, collate_fn
import argparse
from pathlib import Path
from model_lstm import LSTMLanguageModel
from model_rnn import RNNLanguageModel
from model_transformer import TransformerLanguageModel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Registering all supported model architectures
MODEL_REGISTRY = {
    "lstm": LSTMLanguageModel,
    "rnn": RNNLanguageModel,
    "transformer": TransformerLanguageModel
}

def load_best_model(model_type, device="cpu"):
    """
    Finding and returning the most recent 'best' trained model checkpoint.
    """
    model_path = f"results/{model_type}_*/best_{model_type}_model.pt"
    model_files = sorted(Path(".").glob(model_path))
    if not model_files:
        raise FileNotFoundError(f"No best model found for {model_type}")
    return model_files[-1]

def evaluate(args):
    # Verifying model type
    assert args.model in MODEL_REGISTRY, f"Unsupported model type: {args.model}"

    # Selecting computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the SentencePiece tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_model)
    vocab_size = sp.get_piece_size()

    # Instantiating and loading the model
    model_class = MODEL_REGISTRY[args.model]
    model = model_class(vocab_size)
    
    # Loading pre-trained model weights
    model_file = load_best_model(args.model)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Loaded best model from: {model_file}")

    # Loading the test dataset
    test_ds = TextDataset(args.test_path, args.tokenizer_model, max_len=args.max_len)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initializing variables for metric computation
    total_loss = 0
    total_tokens = 0
    predictions = []
    references = []

    # Disabling gradient computation for evaluation
    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forwarding through the model
            logits, _ = model(inputs)

            # Computing total cross-entropy loss, ignoring padding
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=0, reduction='sum')
            total_loss += loss.item()

            # Counting valid (non-padding) tokens
            total_tokens += (targets != 0).sum().item()

            # Decoding predictions and references for BLEU computation
            pred_ids = torch.argmax(logits, dim=-1).tolist()
            ref_ids = targets.tolist()

            for p, r in zip(pred_ids, ref_ids):
                # Decoding and filtering padding tokens
                pred_tokens = sp.DecodeIds([i for i in p if i != 0])
                ref_tokens = [sp.DecodeIds([i for i in r if i != 0]).split()]
                predictions.append(pred_tokens.split())
                references.append(ref_tokens)

    # Calculating perplexity (PPL) and BLEU score
    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    bleu = corpus_bleu(references, predictions)

    # Printing evaluation metrics
    print(f"\n📊 Evaluation Results ({args.model.upper()}):")
    print(f"Perplexity (PPL): {ppl:.2f}")
    print(f"BLEU Score: {bleu:.4f}")
    print(f"Test Samples: {len(test_ds)}")
    print(f"Tokens Processed: {total_tokens}\n")

if __name__ == "__main__":
    # Parsing command-line arguments
    parser = argparse.ArgumentParser(description="Evaluating language models")
    parser.add_argument("--model", 
                       type=str, 
                       choices=["lstm", "rnn", "transformer"], 
                       required=True,
                       help="Model type to evaluate")
    parser.add_argument("--test_path", 
                       type=str, 
                       default="data/test.jsonl",
                       help="Path to test.jsonl file")
    parser.add_argument("--tokenizer_model", 
                       type=str, 
                       default="tokenizer/spm.model",
                       help="Path to SentencePiece model")
    parser.add_argument("--batch_size", 
                       type=int, 
                       default=128,
                       help="Evaluation batch size")
    parser.add_argument("--max_len", 
                       type=int, 
                       default=128,
                       help="Max input sequence length")
    
    # Running evaluation
    args = parser.parse_args()
    evaluate(args)
