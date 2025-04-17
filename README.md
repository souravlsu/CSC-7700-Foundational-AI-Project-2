# 🧠 Language Modeling with RNN, LSTM, and Transformer

This project implements and compares three neural network-based language models — **Vanilla RNN**, **LSTM**, and **Transformer** — for text generation tasks using custom training data and a SentencePiece tokenizer.

---

## 📁 Project Structure

```
├── data/                    # Raw .txt files and JSONL datasets
│   ├── train.jsonl
│   ├── test.jsonl
│   └── *.txt                # Raw text files for tokenizer
│
├── tokenizer/              # Trained SentencePiece model
│   ├── spm.model
│   └── spm.vocab
│
├── model_rnn.py            # RNN model implementation
├── model_lstm.py           # LSTM model implementation
├── model_transformer.py    # Transformer model implementation
│
├── dataset.py              # Dataset class and collation function
├── train.py                # Training script
├── evaluate.py             # Evaluation script (PPL, BLEU)
├── generate.py             # Text generation script
│
├── tokenizer_train.py      # Script for training the SentencePiece tokenizer
├── requirements.txt        # Required Python packages
└── README.md
```

---

## 🧪 Requirements

Make sure Python 3.8+ is installed. Then run:

```bash
pip install -r requirements.txt
```

---

## 📝 Preparing the Dataset and Tokenizer

1. Put raw `.txt` files into the `data/` folder.
2. Run the tokenizer training script:

```bash
python tokenizer_train.py
```

This generates `tokenizer/spm.model` and `tokenizer/spm.vocab`.

3. Ensure `train.jsonl` and `test.jsonl` are present in the `data/` directory with the following format:

```json
{"prompt": "Once upon a time", "completion": "there was a dragon."}
```

---

## 🚀 Training a Model

```bash
python train.py --model lstm     # or "rnn", "transformer"
```

- Model checkpoints and loss plots will be saved to `results/`.

---

## 📊 Evaluating a Model

```bash
python evaluate.py --model lstm
```

This reports:

- **Perplexity (PPL)**
- **BLEU Score**
- **Total tokens processed**

---

## 💬 Generating Text

```bash
python generate.py --model lstm --prompt "The wizard said"
```

Optional flags:

- `--temperature`: Sampling diversity (default: 1.0)
- `--max_len`: Maximum number of tokens to generate (default: 50)

---

## 🛠 Supported Models

| Model Type   | Layers | Hidden Size | Notes                    |
|--------------|--------|-------------|--------------------------|
| `rnn`        | 1      | 512         | Vanilla RNN (tanh)       |
| `lstm`       | 2      | 512         | Better for longer memory |
| `transformer`| 6      | 512         | BPE + positional encoding|

---

## 📈 Example Output

```text
📝 Prompt: 'The wizard said'
🧠 LSTM Completion:
--------------------------------------------------
The wizard said he would return to the castle by morning, and everyone believed him.
--------------------------------------------------
```

---

## 📌 Notes

- Models ignore padding tokens during loss calculation.
- Text generation uses greedy sampling via temperature-adjusted multinomial.

---
