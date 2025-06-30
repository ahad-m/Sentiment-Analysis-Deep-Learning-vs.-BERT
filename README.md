# Sentiment Analysis: Deep Learning vs. BERT

This project compares the effectiveness of various deep learning models (CNN, LeNet, RBM, DBN) and the BERT model for sentiment analysis on the IMDb movie review dataset. The goal is to classify movie reviews as positive or negative and evaluate each model's performance.

## Dataset & Preprocessing

The project uses the IMDb Movie Reviews Sentiment Dataset (50,000 reviews, balanced positive/negative). Data undergoes extensive preprocessing:

*   **Cleaning**: Lowercasing, removing HTML tags, URLs, punctuation, numbers, and extra spaces.
*   **Tokenization & Stopword Removal**: Breaking text into words and removing common words.
*   **Word Embeddings (for DL models)**: Word2Vec (100-dim) for semantic representation.
*   **Sequence Padding**: Standardizing review lengths (200 tokens for DL, 256 for BERT).
*   **BERT Specifics**: Uses `BertTokenizerFast` for specialized tokenization and input formatting.

## Model Architectures

Five models were explored:

*   **Convolutional Neural Network (CNN)**: 1D convolutional layers to capture local text patterns.
*   **LeNet**: Adapted from image recognition, treats word embeddings as 2D inputs.
*   **Restricted Boltzmann Machine (RBM)**: Unsupervised feature extractor for text.
*   **Deep Belief Network (DBN)**: Stacked RBMs for hierarchical feature learning.
*   **Bidirectional Encoder Representations from Transformers (BERT)**: A pre-trained transformer model fine-tuned for sentiment classification, leveraging bidirectional context.

## Training & Evaluation

All models used Binary Cross-Entropy loss. Deep learning models used Adam optimizer (LR: 0.001), trained with early stopping and dropout. BERT fine-tuning typically uses `AdamW` with a small learning rate and linear warmup.

**Performance Comparison (Deep Learning Models):**

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| CNN   | 85.73%   | 83.01%    | 89.86% | 86.30%   |
| LeNet | 80.43%   | 82.10%    | 77.82% | 79.90%   |
| RBM   | 50.00%   | 50.00%    | 100.00%| 66.67%   |
| DBN   | 50.00%   | 0%        | 0%     | 0%       |

**Key Findings:**

*   **CNN** performed best among traditional deep learning models, demonstrating strong capability in text classification.
*   **RBM and DBN** showed limited effectiveness for this task.
*   **BERT (Expected Performance)**: While specific metrics for BERT were not detailed, transformer models like BERT are generally expected to significantly outperform traditional deep learning models due to their superior contextual understanding, pre-training on vast datasets, and ability to handle complex linguistic nuances (e.g., negation, sarcasm, OOV words).

## Conclusion

This project highlights the varying effectiveness of deep learning architectures for sentiment analysis. While CNN proved strong among traditional methods, the theoretical advantages of BERT (bidirectional context, pre-training, handling long-range dependencies) suggest its superior performance, marking the evolution towards more sophisticated transformer-based NLP solutions.


