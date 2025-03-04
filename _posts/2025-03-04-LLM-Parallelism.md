---
title: Parallelism Techniques for Large Language Models
date: 2025-03-04 09:00:00 +0800
categories: [AI, High Performance Computing]
tags: [llm, parallelism, distributed training, model parallelism, deep learning]
image: /assets/img/post/250304/delivery_6045703.png
toc: true
math: true
mermaid: true
---

# Introduction

As Large Language Models (LLMs) continue to grow in size and complexity, efficient parallelization techniques have become essential for both training and inference. This post explores the various parallelism strategies that enable the development and deployment of billion and trillion parameter models on distributed computing systems.

## Background

Modern LLMs like GPT-4, Claude, and Llama 3 contain hundreds of billions of parameters, far exceeding the memory capacity of individual GPUs. To train and run these massive models effectively, researchers and engineers have developed sophisticated parallelization techniques that distribute computation across multiple devices and nodes.

## Data Parallelism

The simplest form of parallelism for deep learning:

- Replicates the entire model across multiple devices
- Each device processes a different batch of data
- Gradients are synchronized across devices
- Linear scaling with number of devices
- Limited by model size fitting on a single device

## Tensor Parallelism

Splits individual tensors across multiple devices:

- Divides neural network layers horizontally
- Each device holds a portion of each layer's weights
- Requires communication during forward and backward passes
- Enables training models larger than single-GPU memory
- Implemented in frameworks like Megatron-LM
- Particularly effective for transformer attention layers
- Reduces memory requirements per device
- Enables training of models that wouldn't fit on a single GPU

### Implementation Details

Tensor parallelism divides matrix operations across multiple devices:

1. **Attention Mechanism Splitting**: 
   - Query, Key, and Value projections are distributed
   - Each GPU computes a portion of the attention heads
   - Results are gathered through all-reduce operations

2. **Feed-Forward Network Splitting**:
   - The large feed-forward layers are sharded across GPUs
   - Each GPU computes a portion of the hidden dimension
   - Communication occurs only at layer boundaries

![Tensor Parallelism Concept](/assets/img/post/250304/tensor0.png)
_Figure: Conceptual illustration of Tensor Parallelism showing how individual layers are split horizontally across multiple GPUs, with each device handling a portion of the neurons in each layer._

![Tensor Parallelism Implementation](/assets/img/post/250304/tensor1.png)
_Figure: Detailed implementation of Tensor Parallelism in transformer models, showing how attention heads and feed-forward networks are distributed across devices with synchronized computation._


## Pipeline Parallelism

Partitions the model vertically across devices:

- Different layers run on different devices
- Introduces micro-batching to maintain device utilization
- Reduces communication overhead compared to tensor parallelism
- Handles activation memory efficiently
- Introduces pipeline bubbles that reduce efficiency
![Pipeline Parallelism Visualization](/assets/img/post/250304/pipeline.png)
_Figure: Illustration of Pipeline Parallelism where different layers of the model are distributed across multiple GPUs, with micro-batches flowing through the pipeline to maintain device utilization._
![Pipeline Parallelism Implementation](/assets/img/post/250304/pipeline1.png)
_Figure: Detailed implementation of Pipeline Parallelism showing how micro-batches flow through model stages across multiple GPUs, with forward and backward passes coordinated to maximize hardware utilization._

## 3D Parallelism

Combines multiple parallelism strategies:

- Data parallelism across node groups
- Pipeline parallelism within node groups
- Tensor parallelism within nodes
- Maximizes hardware utilization
- Enables training of trillion-parameter models
- Used in systems like DeepSpeed and Megatron-Turing NLG
![3D Parallelism Visualization](/assets/img/post/250304/3d.png)
_Figure: Visualization of 3D Parallelism combining data, pipeline, and tensor parallelism strategies for efficient training of trillion-parameter models._

## ZeRO (Zero Redundancy Optimizer)

ZeRO implements three progressive levels of optimization:

### ZeRO Stage 1: Optimizer State Partitioning

- The first step of sharding is applied to the Adam **optimizer states**, which is denoted as $$P_{os}$$ in the figure below. Here, "os" refers to **optimizer states**. Model parameters and gradients are still fully replicated on each GPU. At this stage, the model state memory required per GPU is $$4\Phi + \frac{12\Phi}{N}$$ bytes. When $$N$$ is large, this approaches $$4\Phi B$$, which is equivalent to $$\frac{1}{4}$$ of the original $$16\Phi B$$.

### ZeRO Stage 2: Gradient Partitioning

- If we continue to shard the **model gradients**, denoted as $$P_{os+g}$$ in the figure below, model parameters are still fully replicated on each GPU. At this stage, the model state memory required per GPU is $$2\Phi + \frac{2\Phi + 12\Phi}{N}$$ bytes. When $$N$$ is large, this approaches $$2\Phi B$$, which is equivalent to $$\frac{1}{8}$$ of the original $$16\Phi B$$.

### ZeRO Stage 3: Parameter Partitioning

- If we further shard the **model parameters**, denoted as $$P_{os+g+p}$$ in the figure below, the model state memory required per GPU is $$\frac{16\Phi}{N}$$ bytes. When $$N$$ is large, this approaches 0.

![ZeRO Optimization Stages](/assets/img/post/250304/weight.png)
_Figure: Visualization of ZeRO's three optimization stages showing how optimizer states, gradients, and parameters are progressively sharded across GPUs to reduce memory requirements per device._


### ZeRO Considerations for Multi-Node Training

When scaling to multi-node environments, communication overhead becomes a critical factor:

- **ZeRO Stage 1** is ideal for multi-node setups as it minimizes communication while providing significant memory benefits
- **ZeRO Stage 2** introduces moderate communication overhead but still performs well across nodes
- **ZeRO Stage 3** is not a good choice either for the same reason - more inter-node communications required

### ZeRO-Offload

ZeRO-Offload extends the memory optimization capabilities:

- Offloads optimizer states (Stage 1) to CPU memory
- Leverages high CPU RAM availability (often 10x GPU memory)
- Enables training of larger models on limited GPU hardware
- Performs computation on GPU, only using CPU for storage
- Minimal performance impact with proper overlap of computation and communication


An optimization strategy that eliminates memory redundancy:

- Partitions optimizer states, gradients, and parameters
- Three progressive levels of memory optimization
- Combines advantages of data and model parallelism
- Minimal communication overhead
- Enables training larger models with data parallelism


## Comparison of Approaches

| Method | Memory Efficiency | Communication Cost | Implementation Complexity | Scalability |
|--------|-------------------|-------------------|--------------------------|-------------|
| Data Parallelism | Low | Medium | Low | Limited by model size |
| Tensor Parallelism | High | High | Medium | Good for wide layers |
| Pipeline Parallelism | High | Low | High | Good for deep models |
| 3D Parallelism | Very High | Medium | Very High | Excellent |
| ZeRO | High | Medium | Medium | Very Good | 