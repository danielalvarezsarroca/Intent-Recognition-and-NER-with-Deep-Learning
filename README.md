# Intent Recognition & Named Entity Recognition (NER) with Deep Learning  
UPC – TVD Project (Daniel Álvarez & Albert Roca)

This repository contains two independent but complementary modules commonly used in dialogue systems for air-travel queries:

- Part 1 — Intent Recognition: sentence-level classification.
- Part 2 — Named Entity Recognition (NER): token-level semantic tagging.

Both parts follow an iterative experimental workflow where multiple neural architectures, preprocessing strategies and regularization techniques were tested and compared.

---

# Part 1 — Intent Recognition
Based on the report: P2_1_Álvarez_Roca.pdf :contentReference[oaicite:0]{index=0}

## Objective
Build a model capable of predicting the intent of airline-related user queries in the ATIS dataset. The process includes text preprocessing, vocabulary selection, embedding tuning, and experimentation with several neural architectures.

## Preprocessing and Data Preparation
Several preprocessing variants were evaluated:
- Lowercasing and removal of non-alphabetic characters.
- Lemmatization using spaCy.
- Vocabulary size experiments ranging from 40 to 800 words.
- Tokenization, padding and one-hot encoding of labels.

Main findings:
- Vocabulary size strongly affects performance. Best results reached around 560 words.

## Models Tested

### Baseline Sequential Model
Embedding → GlobalPooling → Dense → Softmax  
Achieved an initial accuracy of approximately 91.5% with a weighted F1-score of 0.898.

### Embedding Size Experiments
Embedding dimensions tested: 50, 100, 128, 200, 300.

- Accuracy improved consistently with larger embeddings.
- Best performance: 300 dimensions (≈ 93.2% accuracy, weighted F1 ≈ 0.921).

### Convolutional Networks (CNN)
Various kernel sizes and pooling strategies were tested.

- Best model: Conv1D kernel=3 + GlobalMaxPooling.  
  Accuracy ≈ 95.5%, weighted F1 ≈ 0.948.

### Recurrent Networks (LSTM / BiLSTM)
LSTM(64/128) and Bidirectional LSTM(64/128).

- Bidirectional models significantly outperformed unidirectional ones.
- Best configuration: BiLSTM(64), achieving ≈ 95.7% accuracy and weighted F1 ≈ 0.954.

### Regularization
Dropout values {0.2, 0.3, 0.5} were studied.

- Dropout = 0.3, applied after the recurrent layer, produced the best balance between generalization and stability.

### Class Balancing
Class weights were applied to address the skewed distribution of intent categories.

- Improved recall for minority classes.
- Slight decrease in global accuracy (from 95.16% to 91.67%).
- Useful when balanced behaviour across classes is required.

## Summary of Part 1
- Best model: BiLSTM(64) with Dropout 0.3.  
- CNNs also performed strongly, but RNNs captured context better.  
- Class balancing helps minority classes but reduces overall accuracy.

---

# Part 2 — Named Entity Recognition (NER)
Based on the report: P2_2_Álvarez_Roca.pdf :contentReference[oaicite:1]{index=1}

## Objective
Identify and classify named entities at token level for airline booking queries (cities, dates, times, etc.).  
This part includes extensive experimentation with CNNs, RNNs, Transformers, embeddings, dropout, and class weighting.

## Data Preparation
- Split: 4078 training sentences, 900 validation sentences, 893 test sentences.
- Vocabulary size: 831 unique tokens.
- Maximum sequence length: 46 tokens.
- Label encoding and one-hot representation for 119–120 distinct entity classes.
- Removal of sentences containing unseen labels.
- Token-level masking and class weighting to handle imbalance.

## Models and Experiments

### Baseline LSTM Model
Embedding(128) → LSTM(128, return_sequences=True) → Dense(softmax)

- Achieved macro-F1 ≈ 0.56 on the test set.
- Established as the baseline to improve upon.

---

## Experimental Blocks

### Embedding Size Experiments
Tested: 32, 64, 128, 256, 512, 1024, 2048, 4096 dimensions.

- Best: 512 dimensions (macro-F1 ≈ 0.7693).
- Very large embeddings caused instability or memory issues.
- Final practical choice: 256 dimensions.

### Convolutional Models
Kernel sizes tested: 3, 5, 7; pooling and no-pooling variants; TextCNN with parallel kernels.

- Best CNN: Conv1D kernel=7 without pooling (macro-F1 ≈ 0.6471).
- Pooling harmed performance due to loss of token alignment.

### Recurrent Models (LSTM / GRU / Bidirectional)
Uni- and bidirectional LSTM/GRU with 64 and 128 units.

Best pure recurrent models:
- BiGRU(128): ≈ 0.7712 macro-F1
- BiLSTM(128): ≈ 0.7545 macro-F1

Hybrid CNN+RNN models did not significantly improve results.

### Transformers
Custom Transformer layers with varying:
- number of heads (4–8)
- depth (1–3 layers)
- feed-forward dimensions (512–1024)

Best Transformer:
- 2 layers, 4 heads, FF=1024 → macro-F1 ≈ 0.5338

Transformers underperformed due to dataset size, imbalance, and training instability.

### Regularization with Dropout
Dropout values from 0.05 to 0.6 placed at different locations in the network.

- Best configuration: Dropout only after Dense layer at 0.05 (macro-F1 ≈ 0.7728).
- Improved the weak LSTM-64 model from 0.49 to 0.77 macro-F1.

### Class Balancing
Token-level class weights based on label frequency.

Effects:
- Transformers degraded significantly (macro-F1 ≈ 0.48 and below).
- BiLSTM(128) improved slightly (0.75 → 0.79).
- BiGRU worsened slightly (0.76 → 0.74).

---

## Summary of Part 2
- Best model: BiLSTM(128) with 256-dim embeddings, achieving macro-F1 close to 0.80.
- Dropout greatly improved generalization, particularly in smaller models.
- Transformers were unstable and performed worst due to data limitations.
- Bidirectional recurrent architectures were the most reliable choice.

---

# Conclusions

This repository demonstrates a complete experimental pipeline for:

- Intent Recognition: best results around F1 ≈ 0.95 with BiLSTM models.
- Named Entity Recognition: best results around macro-F1 ≈ 0.80 with BiLSTM architectures.

Key conclusions:

- Preprocessing and vocabulary selection have a strong impact on performance.
- Bidirectional recurrent networks outperform CNNs and Transformers on small, imbalanced datasets.
- Dropout regularization significantly reduces overfitting and improves weaker models.
- Class weighting helps certain architectures (BiLSTM) but harms others (Transformers).
- Transformers require far more data and tuning to match RNN performance in this setting.

---

# Suggested Repository Structure

