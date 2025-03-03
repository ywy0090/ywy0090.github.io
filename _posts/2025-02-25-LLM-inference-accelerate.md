---
title: Accelerating LLM Inference - Techniques and Best Practices
date: 2025-02-25 09:00:00 +0800
categories: [AI, Performance Optimization]
tags: [llm, inference, optimization, gpu, acceleration, machine learning]
image: /assets/img/post/llm_inf_acc/high-speed_2560047.png  # You'll need to adjust this path
toc: true
math: true
mermaid: true
---

# Introduction

Large Language Models (LLMs) are revolutionizing AI applications across industries, but their computational demands pose significant challenges for deployment. This post explores cutting-edge techniques to accelerate LLM inference, enabling faster responses and more cost-effective implementations.

## Background

LLM inference is the process of generating predictions from a trained model. While training requires massive computational resources, inference efficiency determines the practical utility of LLMs in production environments. The computational bottlenecks include token generation speed, memory bandwidth, and hardware utilization efficiency.

# Overview

1. Apply various *parallelism* to scale up the model across a large number of GPUs. Smart parallelism of model components and data makes it possible to run a model of trillions of parameters.
2. Memory *offloading* to offload temporarily unused data to the CPU and read them back when needed later. This helps with memory usage but causes higher latency.
3. Smart batching strategy; E.g. [EffectiveTransformer](https://github.com/bytedance/effective_transformer) packs consecutive sequences together to remove padding within one batch.
4. Network *compression* techniques, such as *pruning, quantization, distillation*. A model of smaller size, in terms of parameter count or bitwidth, should demand less memory and run faster.
5. Improvement specific to a target model architecture. Many *architectural changes*, especially those for attention layers, help with transformer decoding speed.

# Quantization

1. *Post-Training Quantization (PTQ)*: A model is first trained to convergence and then we convert its weights to lower precision without more training. It is usually quite cheap to implement, in comparison to training.
2. *Quantization-Aware Training (QAT)*: Quantization is applied during pre-training or further fine-tuning ([paper](https://arxiv.org/pdf/1712.05877)). QAT is able to attain better performance but requires extra computation resources and access to representative training data.

## Quantization Techniques

### Post-Training Quantization (PTQ)
Post-training quantization converts a pre-trained model to lower precision without additional training. This approach is computationally efficient and doesn't require access to the original training dataset.

Key PTQ methods include:

1. *Weight-Only Quantization*: Only model weights are quantized while activations remain in higher precision. This approach offers a good balance between performance and accuracy.

2. *Dynamic Quantization*: Quantization parameters are calculated on-the-fly during inference. While flexible, this can add computational overhead.

3. *Static Quantization*: Quantization parameters are pre-computed using a calibration dataset, resulting in faster inference but potentially lower accuracy if the calibration data isn't representative.

4. *Activation-Aware Quantization*: Considers both weights and activation statistics when determining quantization parameters, often yielding better results than weight-only approaches.

### Quantization-Aware Training (QAT)
QAT incorporates quantization effects during the training process, allowing the model to adapt to lower precision. This typically produces better results than PTQ but requires more computational resources.


## Practical Steps for Model Quantization to INT8

To effectively quantize a model to INT8, follow these steps ([quantization theory](https://huggingface.co/docs/optimum/en/concept_guides/quantization#theory)):

1. *Identify Quantization Targets*: Choose which operators to quantize. Focus on operations that dominate computation time, such as linear projections and matrix multiplications.

2. *Try Dynamic Quantization First*: Implement post-training dynamic quantization as a baseline. If performance meets requirements, you can stop here.

3. *Implement Static Quantization*: If dynamic quantization isn't fast enough, try post-training static quantization. This requires:
   - Adding observers to your model at quantization points
   - Selecting representative calibration data
   - Running calibration to determine optimal quantization parameters

4. *Calibration*: Run your model with observers on calibration data to collect statistics about activation ranges.

5. *Convert to Quantized Model*: Replace floating-point operations with their INT8 counterparts and remove observers.

6. *Evaluate Performance*: Assess both speed improvement and accuracy. If accuracy degradation is unacceptable, proceed to QAT.

7. *Quantization-Aware Training (if needed)*: Implement QAT by simulating quantization effects during training, allowing the model to adapt to lower precision.

## Quantization Resources

For those looking to deepen their understanding of quantization techniques, here are some valuable resources:

1. *Academic Research*: ["The Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"](https://arxiv.org/abs/1712.05877) - This seminal paper introduces fundamental concepts for quantization-aware training and efficient integer-only inference.

2. *Beginner's Guide*: ["The Basics of Quantization in Machine Learning (ML) for Beginners"](https://iq.opengenus.org/basics-of-quantization-in-ml/) - A comprehensive blog post explaining quantization concepts in accessible terms.

3. *Precision Formats*:
   - [bfloat16 floating-point format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) - A 16-bit format optimized for deep learning that preserves the dynamic range of 32-bit floats.
   - [Single-precision floating-point format](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) - Details on the standard 32-bit floating-point representation (FP32).

Understanding these resources will provide a solid foundation for implementing effective quantization strategies for LLM inference acceleration.


# Pruning

Pruning reduces model size by removing unnecessary parameters, which can significantly accelerate inference while maintaining model quality. Here are key pruning approaches for LLMs:

1. *Unstructured Pruning*: Individual weights are removed based on their magnitude or importance. While effective at preserving accuracy, unstructured pruning often requires specialized hardware or software to realize speed improvements.

2. *Structured Pruning*: Removes entire channels, attention heads, or layers. This approach directly translates to computational speedups on standard hardware but can lead to larger accuracy drops if not done carefully.

3. *Magnitude-Based Pruning*: The simplest approach that removes weights with the smallest absolute values, assuming they contribute least to the model's performance.

4. *Movement Pruning*: A technique specialized for fine-tuning scenarios that removes weights that are moving toward zero during training.

5. *SparseGPT/Wanda*: Recent algorithms specifically designed for LLMs that identify and prune weights based on their impact on the activation patterns.

Combining pruning with quantization often yields the best results, creating sparse-quantized models that are both smaller and faster.

# Distillation

Knowledge distillation transfers learning from a large, compute-intensive model (teacher) to a smaller, more efficient model (student). This approach is particularly effective for LLMs, enabling significant acceleration with minimal quality loss.

![Knowledge Distillation Process](/assets/img/post/llm_inf_acc/distill_model.png)
*Figure: Illustration of knowledge distillation from a large teacher model to a smaller student model* [Source](https://www.researchgate.net/figure/Fig-2-Generic-architecture-of-knowledge-distillation-using-a-teacher-student-model_fig2_355180688)

## Key Distillation Approaches for LLMs

1. *Response-Based Distillation*: The student model learns to mimic the final output probabilities of the teacher model. For LLMs, this typically involves matching next-token predictions across diverse prompts.

2. *Feature-Based Distillation*: The student learns to reproduce the internal representations (hidden states, attention patterns) of the teacher model, not just its outputs. This can capture more nuanced knowledge.

3. *Layer-wise Distillation*: Knowledge is transferred layer by layer, with each student layer trained to match the corresponding teacher layer. This structured approach helps maintain architectural alignment.

4. *Self-Distillation*: A model serves as both teacher and student, with different sections or configurations of the same model teaching each other. This can be done iteratively to progressively reduce model size.

5. *Task-Specific Distillation*: Rather than general-purpose knowledge transfer, the distillation focuses on performance for specific downstream tasks, creating specialized, efficient models.

## Notable Distillation Techniques for LLMs

- *DistilBERT/DistilGPT*: Pioneer approaches that demonstrated how transformer-based models could be effectively distilled to much smaller versions while retaining ~95% of their capabilities.

- *MiniLM/TinyBERT*: Techniques that focus on distilling attention mechanisms, which are central to transformer performance but computationally expensive.

- *Sequence-Level Knowledge Distillation*: The teacher generates multiple high-quality outputs for each input, which the student then learns from, effectively expanding the training data.

Distillation can be combined with quantization and pruning in a multi-stage optimization pipeline, often yielding models that are 4-10× faster than the original with minimal performance degradation.

