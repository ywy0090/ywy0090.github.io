---
title: Supervised Fine-Tuning Approaches for Large Language Models
date: 2025-03-03 09:00:00 +0800
categories: [AI, Machine Learning]
tags: [llm, fine-tuning, sft, nlp, deep learning]
image: /assets/img/post/llm-sft-approaches/return_11906833.png  # Optional: Add if you want a header image
toc: true                  # Optional: Table of Contents
math: true                # Set to true for mathematical notations
mermaid: false           # Optional: Set to true if using mermaid diagrams
---

# Introduction

Supervised Fine-Tuning (SFT) has become a critical step in adapting Large Language Models (LLMs) for specific tasks and aligning them with human preferences. This post explores different SFT approaches, their trade-offs, and implementation considerations for researchers and practitioners.

## Background

Large Language Models like GPT-4, Claude, and Llama are initially trained on vast corpora through self-supervised learning. However, to make these models truly useful and safe, they require additional training through supervised fine-tuning on high-quality, human-labeled data that demonstrates desired behaviors and outputs.

## Full-Weight Fine-Tuning

The traditional approach involves updating all model parameters during fine-tuning. While this provides maximum flexibility, it has significant drawbacks:

- Requires extensive computational resources
- Storage demands equal to the full model size
- Risk of catastrophic forgetting
- Challenging to merge multiple fine-tuned versions

## Low-Rank Adaptation (LoRA)

LoRA reduces parameter count by learning low-rank decomposition matrices:

\[W = W_0 + BA\]

Where:
- \(W_0\) is the frozen pretrained weights
- \(B\) and \(A\) are low-rank decomposition matrices
- Typically reduces trainable parameters by 10000x

Key benefits:
- Significantly reduced memory requirements
- Faster training and inference
- Multiple adaptations can be merged

## Quantized LoRA (Q-LoRA)

Q-LoRA combines quantization with LoRA to enable fine-tuning on consumer hardware:

- 4-bit quantization of the base model
- Paged attention for memory efficiency
- NF4 (Normal Float 4) quantization format
- Double quantization to reduce memory footprint

## LoRA with Gradient Accumulation (LoRA-GA)

An enhancement to standard LoRA that improves training stability:

- Accumulates gradients over multiple forward passes
- Reduces memory spikes during training
- Enables larger effective batch sizes
- Better handles limited GPU memory scenarios



## Prefix Tuning

Prepends trainable continuous prompts to inputs:

- Freezes the entire LLM
- Only trains the prefix tokens
- Typically uses 100-500 tokens
- Maintains strong performance with <1% of parameters
![Prefix Tuning Approach](/assets/img/post/llm-sft-approaches/prefix_tuning.png)
_Figure: Illustration of Prefix Tuning approach where trainable prefix tokens are prepended to inputs while the LLM parameters remain frozen._
[Prefix Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/pdf/2101.00190)

## P-Tuning

A soft prompting method that learns continuous embeddings:

- Optimizes virtual tokens instead of discrete prompts
- Can be applied at multiple layers
- More flexible than traditional prompt tuning
- Particularly effective for knowledge-intensive tasks
P-Tuning uses an LSTM model to predict virtual token embeddings:

- LSTM parameters are randomly initialized at the start of training
- All LLM parameters remain frozen during training
- Only the LSTM weights (prompt encoder) are updated
- LSTM parameters are shared across tasks being tuned simultaneously
- Outputs unique virtual token embeddings for each specific task

### Implementation Details

The P-Tuning architecture consists of:

- A prompt encoder (LSTM) that generates continuous embeddings
- Virtual tokens inserted among discrete input tokens
- Each virtual token embedding is a vector of size `hidden_size`
- The number of virtual tokens is controlled by `total_virtual_tokens` parameter

### Training Process

During training:
1. The LSTM generates embeddings for virtual tokens
2. These embeddings are inserted at specified positions in the input sequence
3. The combined sequence is processed by the frozen LLM
4. Loss is calculated based on the task objective
5. Gradients update only the LSTM parameters

### Advantages

- More expressive than fixed prompt templates
- Requires minimal parameter updates (only LSTM weights)
- Can be applied to various NLP tasks with consistent architecture
- Particularly effective for knowledge-intensive tasks like fact retrieval

[P-Tuning: GPT Understands, Too](https://arxiv.org/pdf/2103.10385)

## Prompt Tuning

A parameter-efficient approach that prepends trainable continuous vectors to the input:

- Only trains a small set of continuous prompt embeddings
- Keeps all LLM parameters frozen
- Each task has its own dedicated prompt embedding matrix
- Significantly reduces the number of trainable parameters

### Implementation Details

Prompt embeddings are initialized as a 2D matrix:
- Size: `total_virtual_tokens × hidden_size`
- Each task has its own separate embedding matrix
- No parameter sharing between tasks during training or inference

### Initialization Methods

There are two primary ways to initialize the prompt embeddings:

1. **Random Initialization**:
   - Initialize embedding parameters from a random distribution
   - Simple but may require more training iterations to converge

2. **Vocabulary-based Initialization** (recommended):
   - Initialize from existing vocabulary embeddings
   - Provide a string of words in the model's configuration
   - The string is tokenized and adjusted to match the desired number of virtual tokens
   - Vocabulary embeddings are copied to initialize the soft prompt matrix
   - Original vocabulary embeddings remain unchanged during training

### Advantages

- Extremely parameter-efficient (often <0.1% of model parameters)
- Task-specific prompts can be easily swapped during inference
- Minimal storage requirements for multiple tasks
- Competitive performance compared to full fine-tuning for many tasks



## Comparison of Approaches

| Method | Parameter Efficiency | Memory Usage | Training Speed | Performance |
|--------|---------------------|--------------|----------------|-------------|
| Full-Weight | Low | High | Slow | Best |
| LoRA | High | Low | Fast | Very Good |
| Q-LoRA | Very High | Very Low | Medium | Good |
| LoRA-GA | High | Low | Medium | Very Good |
| Prefix Tuning | Very High | Very Low | Fast | Good |
| P-Tuning | Very High | Very Low | Fast | Good |


## VRAM Requirements Comparison
## VRAM Requirements for SFT Approaches

When fine-tuning LLMs, VRAM requirements vary significantly based on the approach used. Here are some practical examples:

### 7B Parameter Models

For a typical 7B parameter model:

- **Full Fine-tuning**: ~112GB VRAM (7B parameters × 16 bytes per parameter in FP16)
- **LoRA Fine-tuning**: ~35GB VRAM (approximately 5× less than full fine-tuning)

### Mid-sized Models

For medium-sized models like MBART (610M parameters):

- **Full Fine-tuning**:
  - Single batch: ~0.45GB VRAM
  - Batch size 32: ~24GB VRAM (across 4 GPUs)
  - Represents approximately 250% memory overhead compared to model size

### Memory Scaling Factors

As a general rule of thumb:
- Full fine-tuning requires approximately 16× the model size in VRAM
- LoRA reduces this to approximately 5× the model size
- QLoRA can further reduce to 2-3× the model size

These requirements can vary based on:
- Sequence length
- Batch size
- Optimizer choice (Adam variants require more memory)
- Gradient accumulation steps
- Mixed precision training settings


The following table provides a comparison of VRAM requirements for different fine-tuning approaches across various model sizes:

| Method | Precision | 7B | 13B | 30B | 70B | 110B |
|--------|-----------|-----|-----|-----|-----|------|
| Full   | 16-bit    | 67GB | 125GB | 288GB | 672GB | 1056GB |
| LoRA   | 16-bit    | 15GB | 28GB | 63GB | 146GB | 229GB |
| QLoRA  | 8-bit     | 9GB | 17GB | 38GB | 88GB | 138GB |
| QLoRA  | 4-bit     | 5GB | 9GB | 20GB | 46GB | 72GB |

_Table: VRAM requirements for different fine-tuning methods across model sizes. Note the dramatic reduction in memory requirements when using parameter-efficient methods like QLoRA with 4-bit precision._

This comparison clearly demonstrates why parameter-efficient fine-tuning methods have become essential for working with large language models, especially when resources are limited. The 4-bit QLoRA approach enables fine-tuning of even 110B parameter models on consumer-grade hardware, which would be impossible with full fine-tuning.

_Source: [How much VRAM do you need for LLM fine-tuning?](https://modal.com/blog/how-much-vram-need-fine-tuning)_

