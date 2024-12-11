---
title: Summary of the Tokenizers in NLP
date: 2024-12-10 14:00:00 +0800
categories: [NLP, Tokenization]
tags: [nlp, tokenizer, bert, gpt, transformers, hugging-face]
toc: true
math: true
---

# Introduction

In Natural Language Processing (NLP), tokenization is a fundamental preprocessing step that converts raw text into machine-readable format. This post summarizes different types of tokenizers, their characteristics, and when to use them.

## Background

Before diving into specific tokenizers, it's important to understand why tokenization matters in NLP and how it has evolved from simple word-based approaches to more sophisticated subword tokenization methods.

## Types of Tokenizers

### 1. Word-Based Tokenizers

Word-based tokenization is the simplest and most intuitive approach to tokenization, where text is split into words based on delimiters (usually spaces and punctuation).

#### How it Works
- Splits text on whitespace and punctuation
- Treats each word as a separate token
- Often includes special handling for contractions and punctuation

#### Example
```python
text = "Hello, how are you?"
tokens = text.split()
print(tokens)
```

#### Advantages
- Simple and intuitive
- Works well for space-separated languages
- Preserves word meaning

#### Limitations
- Large vocabulary size
- Cannot handle out-of-vocabulary (OOV) words
- Poor handling of morphologically rich languages

### 2. Character-Based Tokenizers

Character tokenization breaks text down into individual characters, providing the finest granularity of tokenization.

#### How it Works
- Splits text into individual characters
- May include special tokens for spaces and punctuation
- Can use n-gram combinations

#### Example


#### Advantages
- No OOV problem
- Very small vocabulary size
- Works well for character-based languages (e.g., Chinese)
- Good for spelling correction tasks

#### Limitations
- Very long sequences
- Loses word-level semantics
- Computationally intensive

### 3. Subword Tokenization

![comparison of wordpiece and bpe](/assets/img/post/summary_tokenizers/cmp_wordpiece_bpe.png)
_comparison of wordpiece and bpe_{: .text-center }

Subword tokenization is a hybrid approach that breaks words into meaningful subword units, balancing vocabulary size and semantic meaning.

#### BPE (Byte Pair Encoding)
- Used by GPT models
- Iteratively merges most frequent character pairs
- Creates a vocabulary of subword units
- Originally adapted for NMT by [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909) to handle rare words

#### WordPiece (BERT)
- Similar to BPE but uses likelihood instead of frequency
- Adds '##' prefix for subword continuation
- Optimized for maintaining word meaning

WordPiece was originally developed by Google for use in Japanese and Korean processing [[1]](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf). Like BPE, it is also a data-driven approach to subword tokenization.

#### Unigram

The Unigram model, as described in [Kudo (2018)](https://arxiv.org/pdf/1804.10959.pdf), is a subword tokenization algorithm that uses a probabilistic approach to tokenization.

- **Training**: Begins with a large seed vocabulary (e.g., all possible subword units up to a certain length) and then iteratively prunes it down.
    - Each subword in the vocabulary is assigned a score based on its likelihood in the training data.
    - Less likely subwords are pruned from the vocabulary iteratively.
- **Tokenization**: For a new text, the model evaluates all possible segmentations and chooses the one with the highest likelihood based on the trained unigram model.

#### SentencePiece
- Language-agnostic approach
- Treats text as Unicode characters
- No pre-tokenization required
- Used by many multilingual models

SentencePiece treats the input as a raw input stream, thus including the space in the set of characters to use. It then uses the BPE or unigram algorithm to construct the appropriate vocabulary, as detailed in [Kudo & Richardson (2018)](https://arxiv.org/pdf/1808.06226.pdf).

Popular models using SentencePiece include [**ALBERT**](https://huggingface.co/docs/transformers/model_doc/albert), [**XLNet**](https://huggingface.co/docs/transformers/model_doc/xlnet), [**Marian**](https://huggingface.co/docs/transformers/model_doc/marian), and [**T5**](https://huggingface.co/docs/transformers/model_doc/t5).

#### Advantages of Subword Tokenization
- Balanced vocabulary size
- Handles OOV words effectively
- Good for morphologically rich languages
- Preserves semantic meaning
- Efficient for model training

#### Common Use Cases
- BERT: WordPiece tokenization
- GPT: BPE tokenization
- XLM-R: SentencePiece tokenization

## Practical Implementation

Here's a quick example using Hugging Face transformers:

```python
from transformers import AutoTokenizer

# Initialize BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Example text
text = "I love transformers! They're really powerful."

# 1. Basic tokenization
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# Output: ['i', 'love', 'transform', '##ers', '!', 'they', "'", 're', 'really', 'powerful', '.']

# 2. Convert tokens to IDs
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}")
# Output: [101, 1045, 2293, 19081, 2015, 999, 2027, 1005, 2024, 2428, 7860, 1012, 102]

# 3. Decode back to text
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded text: {decoded_text}")
# Output: "[CLS] i love transformers! they're really powerful. [SEP]"
```

Let's break down what's happening in this example:

1. **Basic Tokenization**:
   - The text is split into tokens
   - Notice how "transformers" is split into ["transform", "##ers"]
   - The "##" prefix indicates a subword continuation

2. **Token to ID Conversion**:
   - Each token is converted to its corresponding ID in BERT's vocabulary
   - Special tokens are added:
     - `[CLS]` (101): Start of sequence token
     - `[SEP]` (102): End of sequence token

3. **Decoding**:
   - The IDs can be converted back to text
   - Special tokens are included in the decoded output
   - Subwords are properly reconstructed

This example demonstrates BERT's WordPiece tokenization in action, showing how it handles:
- Word splitting into subwords
- Special token addition
- Token-to-ID conversion
- Text reconstruction

## Choosing the Right Tokenizer

Consider these factors when selecting a tokenizer:
- Language characteristics (morphology, writing system)
- Vocabulary size requirements
- Computational resources
- Task requirements (translation, classification, etc.)
- Model architecture compatibility


## Conclusion

![comparison of wordpiece, bpe, sentencepiece](/assets/img/post/summary_tokenizers/cmp_tokens.png)
_comparison of wordpiece, bpe, sentencepiece_{: .text-center }

## References

- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [SentencePiece Paper](https://arxiv.org/abs/1808.06226)


[1] Schuster, M., & Nakajima, K. (2012). Japanese and Korean voice search. 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

[2] Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics.

[3] Kudo, T. (2018). Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. arXiv preprint arXiv:1804.10959.

[4] Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. arXiv preprint arXiv:1808.06226.

[5] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

[6] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI Blog.



---
*Feel free to leave comments or questions below!*
