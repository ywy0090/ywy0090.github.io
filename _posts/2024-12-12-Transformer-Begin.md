---
title: "Understanding Transformer Architectures: From Seq2Seq to Modern Variants"
date: 2024-12-12 09:00:00 +0800
categories: [Machine Learning, Deep Learning]
tags: [transformers, attention, nlp, deep-learning, seq2seq]
image: /assets/img/post/transformers/transformers_001.jpg
toc: true
math: true
mermaid: true
---

# Introduction

Transformers have revolutionized natural language processing and beyond. This post explores the evolution from traditional sequence models to various Transformer architectures, explaining why they've become the dominant paradigm in NLP.

## What's Wrong with Seq2Seq?

### Traditional RNN/LSTM Limitations
- Sequential processing bottleneck
- Vanishing/exploding gradients
- Limited context window
- Difficulty with long-range dependencies

### Memory and Computation Issues
- O(n) sequential operations
- Limited parallelization
- Memory constraints for long sequences

## Encoder-Only Transformers

### Architecture Overview
- Bidirectional context
- Full self-attention
- Suitable for understanding tasks

### Structure Details

Key components in the image:
- Input layer: Tokenized sentences (Sentence 1, Sentence 2)
- Token embeddings: [CLS], Tok1...TokN, [SEP]
- Bidirectional encoding layers (shown in blue)
- Classification output at top (Class Label)

### Popular Examples
- BERT and variants (RoBERTa, DeBERTa)
- DistilBERT
- ALBERT

### Key Characteristics
- Good for: classification, sequence tagging (POS tagging, NER), sentiment analysis
- Typically requires fine-tuning for specific tasks
- Cannot generate text (only understand text)
- Bidirectional context awareness through full self-attention

### Limitations
- No text generation capability
- Requires task-specific fine-tuning
- Fixed-length input constraints

## Decoder-Only Transformers

### Architecture Overview
- Autoregressive processing
- Masked self-attention
- Focused on generation tasks

### Notable Models

Key examples include:
- **OpenAI Models**: GPT, GPT-2, GPT-3, GPT-4, ChatGPT
- **Google**: PaLM architecture
- **Deepmind**: Chinchilla
- **Meta**: LLaMA family

### Key Characteristics
- Specialized in text generation
- Unidirectional attention (left-to-right)
- Autoregressive nature
- Typically larger model sizes
- Pre-trained on vast amounts of text data

### Use Cases
- Text generation
- Conversational AI
- Code completion
- Creative writing
- Language translation (though not as optimal as encoder-decoder)

## Encoder-Decoder Transformers

### Architecture Overview

Key architectural components:
1. **Encoder Stack (Left)**:
   - Input Embedding layer
   - Positional Encoding
   - Nx encoder blocks containing:
     - Multi-Head Attention
     - Add & Norm layer
     - Feed Forward layer

2. **Decoder Stack (Right)**:
   - Output Embedding layer
   - Positional Encoding
   - Nx decoder blocks containing:
     - Masked Multi-Head Attention
     - Multi-Head Attention
     - Add & Norm layers
     - Feed Forward layer

3. **Output Layer**:
   - Linear transformation
   - Softmax for probability distribution

### Notable Models
- Original Transformer (Attention is all you need, 2017)
- T5
- BART

### Key Applications
- Machine translation
- Text summarization
- Image captioning (image is fed into encoder)

### Characteristics
- Complete sequence-to-sequence architecture
- Separate encoding and decoding components
- Cross-attention mechanism between encoder and decoder
- Suitable for tasks requiring both understanding and generation

## Attention Mechanisms

### Self-Attention
- Query, Key, Value concept
- Scaled dot-product attention
- Multi-head attention

![Attention Mechanism](/assets/img/post/transformers/att1.png)
_Self-attention mechanism showing Query, Key, Value interactions_
### Cross-Attention
- Connects encoder and decoder by allowing decoder to attend to encoder outputs
- Key components:
  - Queries come from decoder
  - Keys and Values come from encoder output
  - Enables decoder to focus on relevant source information
- Critical benefits:
  - Allows decoder to selectively focus on source sequence
  - Maintains source context throughout generation
  - Enables effective sequence-to-sequence learning
- Common applications:
  - Machine translation
  - Text summarization
  - Question answering
- Implementation details:
  - Usually follows self-attention in decoder
  - Uses similar scaled dot-product mechanism
  - Often employs multiple heads for diverse feature capture
  
### Hierarchical Attention
- Processes text at multiple levels (word, sentence, document)
- Captures document structure and relationships
- Key benefits:
  - Better handling of long documents
  - More interpretable attention patterns
  - Improved document classification
- Common implementations:
  - Word-level attention followed by sentence-level attention
  - Recursive hierarchical structures
  - Multi-granular attention mechanisms

![Hierarchical Attention](/assets/img/post/transformers/att2.png)
_Hierarchical attention architecture showing multiple attention levels_

### Multi-Step Attention
- Processes attention in sequential steps
- Each step refines and builds upon previous attention outputs
- Key advantages:
  - More complex reasoning patterns
  - Better handling of long-range dependencies
  - Improved information flow
- Common implementations:
  - Cascaded attention layers
  - Progressive refinement attention
  - Iterative attention mechanisms

![Multi-Step Attention](/assets/img/post/transformers/att3.png)
_Multi-step attention showing progressive refinement of attention patterns_

### Variants
- Local attention
- Sparse attention
- Linear attention
- Grouped-query attention
- Sparse attention
- ALIBI (Sparse attention)

## Different Transformer Variants

### Efficiency-Focused
- Performer (Linear attention)
- Longformer (Sparse attention)
- BigBird (Sparse random attention)

### Task-Specific
- ViT (Vision Transformer)
- Audio Transformers
- Graph Transformers

### Architectural Innovations
- Perceiver
- Transformer-XL
- PaLM architecture

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. arXiv. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

