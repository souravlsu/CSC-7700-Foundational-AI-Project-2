import sentencepiece as spm
from pathlib import Path
import os

# Creating the tokenizer/ directory if it doesn't already exist
os.makedirs("tokenizer", exist_ok=True)

# Defining path to combined.txt and locating all .txt files inside the data/ directory
data_dir = Path("data")
combined_path = data_dir / "combined.txt"
raw_txts = list(data_dir.glob("*.txt"))

# Raising an error if no input .txt files are found for tokenizer training
if not raw_txts:
    raise FileNotFoundError("No .txt files found in 'data/' directory.")

# Merging the content of all .txt files into a single combined.txt file
with open(combined_path, "w", encoding="utf-8") as outfile:
    for txt_file in raw_txts:
        with open(txt_file, "r", encoding="utf-8") as infile:
            content = infile.read().strip()
            # Appending content if the file is not empty
            if content:
                outfile.write(content + "\n")

# Verifying that combined.txt was created and is not empty
if not combined_path.exists() or combined_path.stat().st_size == 0:
    raise ValueError("combined.txt was not created correctly or is empty.")

print(f"✅ Combined {len(raw_txts)} files into {combined_path}")
print("📦 Starting tokenizer training...")

# Training the SentencePiece tokenizer using BPE on the combined.txt file
spm.SentencePieceTrainer.Train(
    input=str(combined_path),
    model_prefix="tokenizer/spm",   # Saving the model as tokenizer/spm.model and tokenizer/spm.vocab
    vocab_size=10000,               # Setting vocabulary size
    model_type="bpe",               # Choosing Byte-Pair Encoding (BPE) as the model type
    pad_id=0,                       # Assigning special token IDs
    unk_id=1,
    bos_id=2,
    eos_id=3
)

print("✅ Tokenizer training complete.")
print("📄 Generated files: tokenizer/spm.model and tokenizer/spm.vocab")
