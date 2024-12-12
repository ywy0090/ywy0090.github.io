---
title: Local Hosting Qwq 32B with llama.cpp on Dual 4090s
date: 2024-12-09 12:00:00 +0800
categories: [LLM, Deployment]
tags: [llama.cpp, gpu, rtx-4090, local-llm, cuda, ubuntu]
toc: true
math: false
---

## Background

llama.cpp is a powerful C++ implementation of Meta's LLaMA model, optimized for efficient inference on consumer hardware. Originally created by Georgi Gerganov, it has become one of the most popular solutions for running Large Language Models locally.

Key benefits of llama.cpp include:
- Exceptional performance through careful C++ optimization
- Low memory requirements compared to Python implementations
- Support for quantization (4-bit, 5-bit, 8-bit) to reduce VRAM usage
- Multi-GPU support for parallel inference
- Cross-platform compatibility (Linux, Windows, MacOS)

GPU acceleration is particularly important for llama.cpp as it can significantly reduce inference latency. With CUDA support, users can achieve:
- Up to 10x faster inference compared to CPU-only execution
- Ability to run larger models that wouldn't fit in system RAM
- Better token generation speeds for real-time applications
- Efficient handling of multiple concurrent requests

## Hardware Setup

### System Specifications
- Operating System: Ubuntu 24.04 LTS
- CUDA Version: 12.1
- GPUs: Dual NVIDIA RTX 4090
- VRAM: 24GB × 2 (Total 48GB)

### Prerequisites
- NVIDIA Driver compatible with CUDA 12.1
- CMake and build tools
- Git for source code management

### Software Requirements
- Python 3.8+
- C compiler
  - Linux: gcc or clang
  - Windows: Visual Studio or MinGW
  - MacOS: Xcode



## Installation and Configuration

### Using Pre-built Wheel

The easiest way to install llama.cpp with CUDA support is using the pre-built wheel. This method automatically handles all the CUDA dependencies and compilation settings.

1. check if CUDA_HOME is set:
```bash
echo $CUDA_HOME
```
2. Install the CUDA version:
check the cuda version:
```bash
nvcc --version
```
then set the CUDACXX to the cuda version:
```bash
export CUDACXX=/usr/local/cuda-12.1/bin/nvcc
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir\
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

3. Verify the installation:

```python
from llama_cpp import Llama
prompt = "what is 1+1 ?"
llm = Llama(
    model_path="./model_path/gguf",
    n_gpu_layers=-1,  
    n_ctx=1024*32,    
)
output = llm(
    prompt,           
    max_tokens=1024,  
    stop=["Q:"],     
    echo=False 
)
print(output)
```

This approach:
- Automatically includes CUDA support
- No manual compilation needed
- Optimized for your CUDA version
- Includes all necessary dependencies

### Getting Daily Llama CLI and Related Tools

To enhance your experience with llama.cpp, there are several useful command-line tools available:

1. Download CLI tools:
   - Visit [llama.cpp releases page](https://github.com/ggerganov/llama.cpp/releases)
   - Download the appropriate binary for your operating system and hardware:
     - For Windows: `llama-<version>-win-x64.zip`
     - For Linux: `llama-<version>-Linux-<arch>.zip` 
     - For macOS: `llama-<version>-macos-<arch>.zip`
   - Extract the downloaded archive to get access to tools like:
     - `llama-cli` - Main inference CLI
     - `llama-gguf-split` - Model split or merge tool
     - And other useful utilities


## Downloading the Qwen 32B Model

To run the Qwen 32B model with llama.cpp, you need to download the model files. Follow these steps:

1. Go to [QwQ-32B-Preview-GGUF on Hugging Face](https://huggingface.co/bartowski/QwQ-32B-Preview-GGUF) and select the model variant you want to download.
Ensure the file size matches the expected size to confirm a successful download.
2. Navigate to the directory where you want to store the model
3. If you downloaded a split model (multiple .gguf files), you'll need to merge them:
 like following
  ./llama-gguf-split --merge <first-split-file-path> <merged-file-path>
  for example 
  ./llama-gguf-split --merge QwQ-32B-Preview-Q4_K_M_m-00001-of-00003.gguf QwQ-32B-Preview-Q4_K_M.gguf
4. Run the following python code to verify the model is loaded correctly and can generate response:
```python
from llama_cpp import Llama
prompt = "what is 1+1 ?"
llm = Llama(
    model_path="./model_path/gguf",
    n_gpu_layers=-1,  
    n_ctx=1024*32,    
)
output = llm(
    prompt,           
    max_tokens=1024,  
    stop=["Q:"],     # Stop at new questions
    echo=False       # Don't echo the prompt
)
print(output)
```
5. Chat with the model, using following python code:
```python
import llama_cpp
import sys
import time
model_path = "./QwQ-32B-Preview-Q4_K_M.gguf"  # Replace with your model path
ctx_size = 2048  # Adjust based on your model and needs
n_gpu_layers = -1  # Set to a positive number to use GPU acceleration
llm = llama_cpp.Llama(
    model_path=model_path,
    n_ctx=ctx_size,
    n_gpu_layers=n_gpu_layers
)
def generate_response_stream(prompt, max_tokens=4096):
    """Generate a response using the Llama model with streaming output."""
    response = ""
    for token in llm(
        prompt,
        max_tokens=max_tokens,
        stop=["Human:"],
        stream=True
    ):
        chunk = token['choices'][0]['text']
        response += chunk
        yield chunk
def chat():
    """Main chat loop."""
    print("Welcome to the Llama Chat! (Type 'quit' to exit)")
    conversation_history = ""
    while True:
        user_input = input("Human: ").strip()
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        prompt = f"{conversation_history}Human: {user_input}\nAI:"
        print("AI: ", end="", flush=True)
        ai_response = ""
        for chunk in generate_response_stream(prompt):
            ai_response += chunk
            print(chunk, end="", flush=True)
            time.sleep(0.02)  # Add a small delay for a more natural typing effect
        print()  # New line after response
        conversation_history += f"Human: {user_input}\nAI: {ai_response}\n"
if __name__ == "__main__":
    chat()
```
