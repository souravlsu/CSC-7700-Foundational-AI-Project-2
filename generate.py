import torch
import argparse
import sentencepiece as spm
from pathlib import Path
from model_lstm import LSTMLanguageModel
from model_rnn import RNNLanguageModel
from model_transformer import TransformerLanguageModel

# Registering available model architectures
MODEL_REGISTRY = {
    "lstm": LSTMLanguageModel,
    "rnn": RNNLanguageModel,
    "transformer": TransformerLanguageModel
}

def load_latest_model(model_type, device="cpu"):
    """
    Finding and loading the most recent trained model checkpoint.
    """
    model_path = f"results/{model_type}_*/best_{model_type}_model.pt"
    model_files = sorted(Path(".").glob(model_path))
    if not model_files:
        raise FileNotFoundError(f"No trained model found for {model_type}. Train first!")
    return model_files[-1]

def generate_text(args):
    # Checking model type validity
    assert args.model in MODEL_REGISTRY, f"Unsupported model type: {args.model}"

    # Selecting computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the trained SentencePiece tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_model)

    # Initializing the model with vocabulary size
    model = MODEL_REGISTRY[args.model](sp.get_piece_size())

    # Loading the most recent model checkpoint
    model_path = load_latest_model(args.model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Loaded model from: {model_path}")

    # Printing input prompt
    print(f"\n📝 Prompt: '{args.prompt}'")

    # Generating text using the model
    output = model.prompt(
        args.prompt,
        sp,
        max_seq_length=args.max_len,
        temperature=args.temperature
    )

    # Displaying generated completion
    print(f"\n🧠 {args.model.upper()} Completion:")
    print("-" * 50)
    print(output)
    print("-" * 50)

if __name__ == "__main__":
    # Parsing command-line arguments for text generation
    parser = argparse.ArgumentParser(
        description="Generating text using trained language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "rnn", "transformer"],
        help="Type of model to use for generation"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt to start generation"
    )
    parser.add_argument(
        "--tokenizer_model",
        type=str,
        default="tokenizer/spm.model",
        help="Path to SentencePiece tokenizer model"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=50,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (1.0=normal, lower=more conservative)"
    )

    # Running the main generation function
    args = parser.parse_args()
    generate_text(args)
