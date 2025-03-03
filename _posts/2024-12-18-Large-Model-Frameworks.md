---
title: Large Model Distributed Training Frameworks (Transformer, DeepSpeed, FSDP)
date: 2024-12-18 09:00:00 +0800
categories: [Machine Learning, Inference]
tags: [large-models, training, transformer, deepspeed, fsdp]
toc: true                  # Optional: Table of Contents
math: false               # Optional: Set to true if using mathematical notations
mermaid: false           # Optional: Set to true if using mermaid diagrams
---

# Introduction

Brief introduction to what this post is about and why it matters (2-3 sentences).

## Background

Provide necessary context or background information that readers need to understand the topic.

## Hugging Face Transformer

### Overview
- Hugging Face Transformers provides a unified architecture for training and deploying transformer models
- Core training components include:
  - Trainer class that handles training loop, optimization, and device management
  - Training arguments for configuring hyperparameters, batch sizes, learning rates etc.
  - Built-in support for distributed training across multiple GPUs/machines
  - Automatic mixed precision training for improved performance
  - Gradient accumulation and gradient clipping
- Integrates seamlessly with popular training frameworks:
  - Native PyTorch support
  - TensorFlow compatibility 
  - JAX/Flax implementations
- Extensive pre-trained model support with easy fine-tuning capabilities

### Training with PyTorch Trainer

- Provides an optimized Trainer class specifically for Transformer models
- Eliminates need to write custom training loops
- Key features:
  - Comprehensive logging capabilities
  - Built-in gradient accumulation
  - Mixed precision training support
  - Distributed training across devices
- Simplified model initialization:
  - Automatic loading of pretrained weights
  - Memory-efficient loading with `torch_dtype="auto"`
  - Configurable through model's config.json
- Handles training workflow complexities:
  - Batch processing
  - Device management
  - Optimization steps
  - Model checkpointing
- Highly customizable training arguments
  - Learning rates
  - Batch sizes
  - Training epochs
  - Evaluation strategies

### Training with TensorFlow/Keras

- Seamless integration with TensorFlow's Keras API for model training
- Advantages of using Keras:
  - High-level, user-friendly API
  - Built-in training loops and metrics
  - Easy model compilation and fitting
  - Native TensorFlow optimization
- Key features:
  - Custom training loops possible with GradientTape
  - Automatic mixed precision support
  - TPU compatibility out-of-the-box
  - Distributed training capabilities
- Simple workflow:
  - Convert models to TF format
  - Compile with optimizer and loss
  - Fit model with dataset
  - Monitor training metrics
- Benefits:
  - Production-ready deployment
  - TensorFlow Serving integration
  - TensorFlow Lite conversion
  - TensorBoard visualization

### Training with Native PyTorch
- Provides full control over training loop implementation
- Key steps:
  - Data preparation:
    - Remove unnecessary columns from dataset
    - Rename label column to match model expectations
    - Convert dataset format to PyTorch tensors
  - DataLoader setup:
    - Create training and evaluation dataloaders
    - Configure batch size and shuffling
  - Model initialization:
    - Load pretrained model with correct number of labels
    - Move model to appropriate device (CPU/GPU)
  - Training configuration:
    - Setup optimizer (typically AdamW)
    - Configure learning rate scheduler
    - Define number of epochs
  - Custom training loop:
    - Iterate through epochs and batches
    - Forward pass and loss calculation
    - Backward pass and optimization
    - Learning rate scheduling
    - Progress tracking
- Benefits:
  - Complete flexibility in training process
  - Fine-grained control over optimization
  - Custom loss functions and metrics
  - Direct access to model internals
- Considerations:
  - Requires more boilerplate code
  - Manual implementation of training features
  - Need to handle device management
  - Memory management responsibility

### PEFT (Parameter-Efficient Fine-Tuning)
- A library for efficiently adapting large pretrained models to downstream tasks
- Enables fine-tuning without updating all model parameters
- Key benefits:
  - Significantly reduces computational and storage costs
  - Achieves performance comparable to full fine-tuning
  - Makes LLM training accessible on consumer hardware
  - Only fine-tunes a small number of extra parameters
- Ideal for adapting large language models with limited resources

### TRL (Transformer Reinforcement Learning)
- A comprehensive library for training transformer language models with Reinforcement Learning
- Provides end-to-end tools for the complete RLHF (Reinforcement Learning from Human Feedback) pipeline:
  - Supervised Fine-tuning (SFT) for initial model training
  - Reward Modeling (RM) for learning reward functions
  - Proximal Policy Optimization (PPO) for policy improvement
- Seamlessly integrates with Hugging Face transformers library
- Enables training of language models that better align with human preferences
- Key components include:
  - SFTTrainer for supervised fine-tuning
  - RewardTrainer for training reward models
  - PPOTrainer for reinforcement learning optimization
  - Utilities for data processing and model evaluation
- Built on top of PyTorch and transformers for maximum compatibility


## DeepSpeed

### Overview
- A deep learning optimization library developed by Microsoft for efficient distributed training
- Enables training of large-scale models with trillions of parameters
- Key features include:
  - Memory optimization through ZeRO stages
  - Pipeline parallelism for model distribution
  - 3D parallelism combining data, pipeline and tensor parallelism
  - Automatic mixed precision training
  - Dynamic loss scaling
  - Gradient accumulation and clipping
- Highly configurable through JSON configuration files
- Integrates with popular frameworks like PyTorch and Hugging Face
- Provides significant speedup and memory reduction compared to standard distributed training

### Features
- ZeRO (Zero Redundancy Optimizer) for memory optimization.
  - [Zero: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054.pdf)
- Mixed precision training for faster computation.
- Advanced model parallelism techniques.


### Training
- Supports efficient large-scale model training through:
  - Distributed training across multiple GPUs and nodes
  - ZeRO optimizer stages for memory optimization
  - Gradient accumulation for larger effective batch sizes
  - Automatic mixed precision training
  - Dynamic loss scaling for numerical stability
- Configurable training parameters via JSON configuration
- Integration with popular training frameworks and libraries

### Inference
- DeepSpeed-Inference optimizes model serving through:
  - Tensor parallelism for distributed inference
  - Continuous batching for improved throughput
  - Kernel optimizations for faster computation
  - Quantization support for reduced memory footprint
  - CPU, GPU, and multi-GPU inference support
- Specialized inference kernels for transformer architectures
- Dynamic batch size handling for varying workloads

### Compression
- Provides multiple compression techniques:
  - Quantization (INT8, FP16, mixed precision)
  - Pruning for model sparsification
  - Knowledge distillation support
  - Layer reduction and fusion
  - Weight sharing mechanisms
- Maintains model accuracy while reducing:
  - Memory usage
  - Computation requirements
  - Model size
- Compression Improves Inference Latency
![DeepSpeed Compression Improves Inference Latency](/assets/img/post/llm_train_frameworks/dp2.png)




- Configurable compression pipelines for different use cases


| Category       | Methods                         | Targets                     |
|----------------|---------------------------------|-----------------------------|
| Quantization   | INT8/INT4                       | Activations                 |
|                | INT8/INT4/Ternary/Binary        | Weights                     |
| Sparsification | Head pruning                    | Attention head (Transformer)|
|                | Sparse/Row pruning              | Weights                     |
|                | Channel pruning                 | Conv2D weights              |
| Layer Reduction| Arbitrary subset of network layers | Layers                  |
| Distillation   | Output logits, feature map, attn. map | Layers               |



## PyTorch FSDP (Fully Sharded Data Parallel)

### Overview
- A PyTorch distributed training feature that improves upon DistributedDataParallel (DDP) by sharding model components across GPUs
- Unlike DDP which replicates the full model on each worker, FSDP divides and distributes:
  - Model parameters
  - Optimizer states 
  - Gradients
- Key features include:
  - Automatic model sharding for reduced per-GPU memory usage
  - Optimized communication patterns that overlap with computation
  - Support for mixed precision training and gradient clipping
  - Flexible sharding strategies (full, mixed, hybrid)
  - Seamless integration with PyTorch ecosystem
- Enables training of larger models and batch sizes that wouldn't fit in DDP
- Achieves this memory efficiency with some additional communication overhead
![PyTorch FSDP Architecture](/assets/img/post/llm_train_frameworks/dp3.png)

### Benefits
- Reduces memory usage by sharding model states.
- Supports complex models with large parameter sizes.
- Integrates seamlessly with PyTorch's existing ecosystem.

### How FSDP Works

At a high level FSDP works as follows:

#### In Constructor
- Shards model parameters and each rank only keeps its own shard

#### In Forward Path
- Runs all_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit
- Runs forward computation
- Discards parameter shards it has just collected

#### In Backward Path
- Runs all_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit
- Runs backward computation
- Runs reduce_scatter to sync gradients
- Discards parameters

One way to view FSDP's sharding is to decompose the DDP gradient all-reduce into reduce-scatter and all-gather. Specifically, during the backward pass, FSDP reduces and scatters gradients, ensuring that each rank possesses a shard of the gradients. Then it updates the corresponding shard of the parameters in the optimizer step. Finally, in the subsequent forward pass, it performs an all-gather operation to collect and combine the updated parameter shards.

## PyTorch FSDP2

### Overview
- Next generation of PyTorch's Fully Sharded Data Parallel training system delivering up to 50% throughput speedup
- Completely rewritten from the ground up to improve performance and usability
- Key improvements include:
  - Support for float8 training with DTensor and torch.compile integration
  - Proven performance gains across model sizes from 1.8B to 405B parameters
  - Maintains training quality and loss convergence compared to FSDP1
  - Validated on large-scale training (up to 1T tokens)
  - Compatible with Meta LLaMa architecture family
  - Achieves faster training through optimized weight communication and compute

### Key Features
- Backward compatibility with FSDP1 while offering new optimizations
- Dynamic resharding for better memory management
- Enhanced support for:
  - Activation checkpointing
  - Mixed precision training
  - Custom sharding policies
  - Model state dict handling
- Improved integration with PyTorch ecosystem tools
- Better performance monitoring and profiling capabilities

### Float8 Training Support
- Implements float8 format introduced by NVIDIA, ARM, and Intel in 2022
- Key advantages:
  - Up to 2x improvement in training throughput on NVIDIA Hopper GPUs
  - Native float8 tensor core support
  - Maintains model quality despite lower precision
- Technical implementation details:
  - Uses torchtitan as training entry point
  - Leverages IBM's deterministic data loader
  - Implements float8 linear layers via torchao
  - Utilizes float8 all-gather from PyTorch nightlies
  - Uses tensor-wise scaling granularity
  - Integrates torch.compile for maximum performance
- Current status:
  - Core matrix multiplication operations enabled via NVIDIA libraries
  - Distributed training framework support through FSDP2
  - Weight communication between GPUs in float8
  - Attention computation in bf16 using SDPA (float8 in development)
  
### Training Loss Comparison
![FSDP1 vs FSDP2 Loss Comparison](/assets/img/post/llm_train_frameworks/fg2.png)
*Training loss comparison between FSDP1 and FSDP2 showing equivalent convergence behavior*

### Benchmark Results
Below are benchmark scores comparing float8 and bf16 trained models evaluated in FP16 after 1T tokens of FineWeb pre-training:

| Benchmark | Score (float8) | Score (bf16) |
|-----------|---------------|--------------|
| MMLU (5-shot) | 0.26 | 0.29 |
| ARC-e | 0.73 | 0.73 |
| ARC-c | 0.43 | 0.46 |
| Hellaswag | 0.65 | 0.67 |
| sciq | 0.89 | 0.88 |
| OpenBook QA | 0.43 | 0.43 |
| PIQA | 0.76 | 0.76 |
| Winogrande | 0.60 | 0.65 |
| Average | 0.59 | 0.60 |

The results demonstrate that float8 training maintains comparable performance to bf16 across most benchmarks, with only minor degradation on some tasks. This validates that the reduced precision does not significantly impact model quality while providing substantial training speedups.


### Benefits
- Reduced memory overhead compared to FSDP1
- More flexible configuration options for different training scenarios
- Improved stability for large-scale distributed training
- Better debugging experience with clearer error messages
- Seamless integration with existing PyTorch workflows


## Conclusion

Summarize the importance of these frameworks in advancing large model training and their impact on AI research and applications.


## References

1. [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) - Official PyTorch tutorial on Fully Sharded Data Parallel training
2. [Training using Float8 with FSDP2](https://pytorch.org/blog/training-using-float8-fsdp2/) - PyTorch blog post on Float8 training with FSDP2

3. [DeepSpeed Compression](https://www.microsoft.com/en-us/research/blog/deepspeed-compression-a-composable-library-for-extreme-compression-and-zero-cost-quantization/) - Microsoft Research blog post on DeepSpeed compression techniques
4. [Fine-tuning a Pretrained Model](https://huggingface.co/docs/transformers/training#fine-tune-a-pretrained-model) - Hugging Face documentation on fine-tuning pretrained models
