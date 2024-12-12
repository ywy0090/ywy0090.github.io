---
title: Summary of Different Embedding Models
date: 2024-12-11 09:00:00 +0800
categories: [Machine Learning, NLP]
tags: [embeddings, nlp, machine-learning, text-embeddings]
image: /assets/img/post/embedding/pexels-pixabay-256417.jpg  # You can add an appropriate image
toc: true
math: true    # Useful for explaining embedding dimensions
mermaid: false
---

# Introduction

Text embeddings are crucial in modern NLP, converting text into dense vector representations that capture semantic meaning. This post explores different embedding models, their architectures, and their specific use cases.

## Background

Before diving into specific models, it's important to understand that embeddings are learned representations of text in a continuous vector space, where similar texts are mapped to nearby points, enabling mathematical operations on text.

## Traditional Methods

### TF-IDF
- Based on bag-of-words representation
- No training required
- Weights terms based on frequency in document and corpus
- Simple but effective for many tasks

### Word2Vec
- Two main architectures:
  - Skip-gram: Predicts context words from target word
  - CBOW (Continuous Bag of Words): Predicts target word from context words
- Learns distributed word representations
- Captures semantic relationships between words

### GloVe (Global Vectors)
- Uses word co-occurrence matrix
- Combines benefits of matrix factorization and local context window methods
- Explicitly encodes meaning as vector offsets
- Trained on aggregated global word-word co-occurrence statistics

### FastText
- Extension of Word2Vec that handles subwords
- Uses character n-grams
- Better handles rare words and out-of-vocabulary words
- Particularly effective for morphologically rich languages

## Contextual Embeddings

### ELMo (Embeddings from Language Models)
- Uses bidirectional LSTM architecture
- Generates contextual embeddings
- Different representations at each layer
- Pre-trained on large text corpora

### CoVe (Contextualized Word Vectors)
- Uses deep LSTM layers
- Trained on machine translation task
- Generates context-aware representations
- Can be used as features for downstream tasks

## Transformer-Based Models

### BERT
- Based on Transformer architecture
- Bidirectional training
- Uses masked language modeling
- Revolutionized NLP with state-of-the-art results

### RoBERTa (Robustly Optimized BERT)
- Built on BERT architecture
- Key improvements:
  - Removes Next Sentence Prediction (NSP)
  - Uses larger batch sizes
  - Employs dynamic masking
  - Optimized learning rate

### ALBERT (A Lite BERT)
- Memory-efficient version of BERT
- Parameter reduction techniques:
  - Factorized embedding parameterization
  - Cross-layer parameter sharing
- Maintains performance while reducing model size

### XLNet
- Combines best of autoregressive and autoencoding approaches
- Key features:
  - Permutation-based training
  - Two-stream self-attention
  - Integrates ideas from Transformer-XL
- Handles dependencies and context better than BERT

## High Quality Sentence Embeddings

When looking to obtain high quality sentence embeddings (excluding OpenAI's embedding API), there are several strong options:

### For English Text
- The `all-mpnet-base-v2` model from sentence-transformers is recommended as a strong baseline
- Provides good performance across a wide range of tasks
- Easy to use through the sentence-transformers library

### For Chinese Text
Several models have been evaluated for Chinese sentence embeddings:

- `DMetaSoul/sbert-chinese-general-v2`
- `clip-ViT-B-32` 
- `sentence-transformers/all-mpnet-base-v2`
- `sentence-transformers/distilbert-base-nli-max-tokens`
- `paraphrase-multilingual-mpnet-base-v2`
