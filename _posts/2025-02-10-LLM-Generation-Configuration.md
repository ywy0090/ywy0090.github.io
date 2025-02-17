---
title: LLM Generation Configuration and Parameters
date: 2025-02-10 09:00:00 +0800
categories: [Machine Learning, LLM]
tags: [llm, text-generation, inference, ai]
image: /assets/img/post/llm_generation/modeling_2181544.png  # Optional: Add relevant image
toc: true
math: true
mermaid: true
---

# Introduction

Understanding and configuring Large Language Model (LLM) generation parameters is crucial for optimal performance. This post explores key configuration options, their effects on generation quality, and best practices for different use cases.

## Generation Parameters

### Length Control Parameters
- max_length (int, defaults to 20):
  - Maximum length of generated tokens
  - Overridden by max_new_tokens if set
- max_new_tokens (int, optional):
  - Maximum number of tokens to generate
  - Ignores prompt token count
- min_length (int, defaults to 0):
  - Minimum sequence length
  - Overridden by min_new_tokens if set
- min_new_tokens (int, optional):
  - Minimum number of new tokens
  - Ignores prompt token count
- max_time (float, optional):
  - Maximum computation time in seconds
  - Completes current pass after timeout

### Beam Search Parameters
- early_stopping (bool or str, defaults to False):
  - Controls beam search stopping conditions
  - Options:
    - True: Stops when num_beams candidates complete
    - False: Uses heuristic for unlikely better candidates
    - "never": Only stops when no better candidates possible
  - Affects beam-based generation methods

### Logit Manipulation
- temperature (float, defaults to 1.0):
  - Modulates next token probabilities
  - Higher values (>1.0):
    - More random/diverse output
    - Increased creativity
  - Lower values (<1.0):
    - More focused/deterministic
    - Better for factual responses
- top_k (int, defaults to 50):
  - Keeps k highest probability tokens
  - Filters vocabulary for sampling
- top_p (float, defaults to 1.0):
  - Nucleus/cumulative probability sampling
  - Keeps smallest set of tokens summing to top_p
  - Lower values: More deterministic
  - Higher values: More diverse

### Special Tokens
- pad_token_id (int, optional):
  - ID for padding token
  - Used for batch processing
- bos_token_id (int, optional):
  - Beginning-of-sequence token ID
  - Marks start of generation
- eos_token_id (Union[int, List[int]], optional):
  - End-of-sequence token ID(s)
  - Can specify multiple end tokens
  - Controls generation stopping

### Temperature
- Controls randomness in generation
- Range: 0.0 to 2.0 (typically)
- Lower values (e.g., 0.1):
  - More deterministic
  - Good for factual/logical tasks
- Higher values (e.g., 0.8):
  - More creative/diverse
  - Better for creative writing

### Top-k Sampling
- Limits vocabulary to k most likely tokens
- Implementation:
```python
def top_k_sampling(logits, k):
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### Top-p (Nucleus) Sampling
- Dynamically selects tokens based on cumulative probability
- Also known as nucleus sampling
- Implementation:
```python
def top_p_sampling(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
```

### Repetition Penalty
- Prevents repetitive text generation through penalized sampling
- Discounts scores of previously generated tokens
- Mathematical formulation:
  $$
  p_i = \frac{\exp(x_i/(T \cdot I(i \in g)))}{\sum_j \exp(x_j/(T \cdot I(j \in g)))}
  $$
  where:
  - $I(c)$ is the indicator function:
    $$
    I(c) = \begin{cases} 
    \theta & \text{if } c \text{ is True} \\
    1 & \text{otherwise}
    \end{cases}
    $$
  - $g$ is the list of generated tokens
  - $T$ is the temperature
  - $\theta$ is the penalty factor (typically ≈ 1.2)

Key characteristics:
- Similar to coverage mechanisms in training
- Applied during inference, not training
- Requires well-trained base distribution
- Optimal settings:
  - θ ≈ 1.2 balances truthful generation and repetition prevention
  - θ = 1.0 reduces to standard sampling
  - Higher values more aggressively prevent repetition

Implementation considerations:
- Only effective if model has learned reliable distributions
- Can be combined with other sampling methods
- More computationally efficient than training-time solutions

### Length Settings
- max_length: Maximum generation length
- min_length: Minimum generation length
- early_stopping: Whether to stop before max_length
- no_repeat_ngram_size: Prevents repetition of n-grams

## Common Configurations

### Chat Completion
```python
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "max_new_tokens": 512,
    "do_sample": True
}
```

### Code Generation
```python
generation_config = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 0,
    "repetition_penalty": 1.2,
    "max_new_tokens": 1024,
    "do_sample": True
}
```

### Creative Writing
```python
generation_config = {
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 100,
    "repetition_penalty": 1.05,
    "max_new_tokens": 2048,
    "do_sample": True
}
```

## Best Practices

### Parameter Selection
1. Task-based configuration:
   - Factual: Low temperature, high top-p
   - Creative: High temperature, moderate top-p
   - Code: Low temperature, high repetition penalty

2. Length considerations:
   - Set appropriate max_length for task
   - Use early_stopping when possible
   - Consider context window limitations

3. Sampling strategy:
   - Greedy (temperature=0): Deterministic output
   - Pure sampling: Most diverse
   - Top-k + Top-p: Balanced approach

### Performance Optimization
```python
generation_config = {
    # Speed optimization
    "use_cache": True,
    "num_beams": 1,  # Disable beam search for faster inference
    
    # Memory optimization
    "max_new_tokens": 512,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    
    # Quality settings
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}
```

## Troubleshooting

Common issues and solutions:
1. Repetitive output
   - Increase repetition_penalty
   - Adjust temperature
   - Check no_repeat_ngram_size

2. Low quality/irrelevant output
   - Adjust temperature
   - Review prompt engineering
   - Check context window usage

3. Slow generation
   - Reduce max_new_tokens
   - Enable use_cache
   - Optimize batch size

## Conclusion

Effective LLM generation requires careful parameter tuning based on specific use cases. Understanding these configurations helps achieve optimal results for different applications.

## References

1. [Hugging Face Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)
2. [OpenAI API Parameters Guide](https://platform.openai.com/docs/api-reference/completions)
3. [The Nucleus Sampling Paper](https://arxiv.org/abs/1904.09751) 