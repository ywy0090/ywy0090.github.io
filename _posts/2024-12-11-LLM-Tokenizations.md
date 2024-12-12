---
title: Understanding LLM Tokenization Implementation
date: 2024-12-11 09:00:00 +0800
categories: [Machine Learning, NLP]
tags: [llm, tokenization, nlp, machine-learning]
image: /assets/img/post/summary_tokenizers/pexels-googledeepmind-25626434.jpg  # You can add an appropriate image
toc: true
math: true    # Might be useful for explaining algorithms
mermaid: false
---

# Introduction

Tokenization is a fundamental process in Large Language Models (LLMs) that converts raw text into numerical tokens for model processing. This post explores how tokenization is implemented in popular LLMs and the technical details behind different tokenization approaches.

## Background

Before diving into implementation details, we'll cover the basic concepts of tokenization, including subword tokenization methods like BPE (Byte-Pair Encoding) and WordPiece, which are commonly used in modern LLMs.

## Common Tokenization Approaches

In this post, we'll explore three widely-used tokenization methods in modern LLMs, with code examples from the Hugging Face Transformers library. 

1. Widespread adoption across popular models
   - BERT's WordPiece is used in many BERT variants
   - SentencePiece powers T5 and many multilingual models  
   - Byte-level BPE is employed by newer models like LLaMA and GPT

2. Distinct algorithmic approaches
   - WordPiece: Simple rule-based subword segmentation
   - SentencePiece: Statistical unigram language modeling
   - Byte-level BPE: Sophisticated byte-pair encoding

3. Progressive complexity
   - Starting with WordPiece's straightforward implementation
   - Moving to SentencePiece's more nuanced statistical approach
   - Culminating in byte-level BPE's advanced handling of encodings

Let's examine each method's implementation details and tradeoffs.
## Base Trie Class

The Trie data structure is fundamental for efficient token lookup in tokenizers. Here's how it's typically implemented:
```python
class Trie:
    """
    Trie in Python. Creates a Trie out of a list of words. The trie is used to split on `added_tokens` in one pass
    Loose reference https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self):
        self.data = {}
        self._tokens = set()

    def add(self, word: str):
        """
        Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
        The special key `""` is used to represent termination.

        This function is idempotent, adding twice the same word will leave the trie unchanged

        Example:

        """
        if not word:
            # Prevent empty string
            return

        self._tokens.add(word)
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[""] = 1

    def split(self, text: str) -> List[str]:
        """
        Will look for the words added to the trie within `text`. Output is the original string splitted along the
        boundaries of the words found.

        This trie will match the longest possible word first !

        Example:
        """
        # indexes are counted left of the chars index.
        # "hello", index 0, is left of h, index 1 is between h and e.
        # index 5 is right of the "o".

        # States are going to capture every possible start (indexes as above)
        # as keys, and have as values, a pointer to the position in the trie
        # where we're at. This is a partial match for now.
        # This enables to keep track of multiple matches while we're iterating
        # the string
        # If the trie contains, "blowing", and "lower" and we encounter the
        # string "blower", we need to split into ["b", "lower"].
        # This is where we need to keep track of multiple possible starts.
        states = OrderedDict()

        # This will contain every indices where we need
        # to cut.
        # We force to cut at offset 0 and len(text) (added later)
        offsets = [0]

        # This is used by the lookahead which needs to skip over
        # some text where the full match exceeded the place in the initial
        # for loop
        skip = 0
        # Main loop, Giving this algorithm O(n) complexity
        for current, current_char in enumerate(text):
            if skip and current < skip:
                # Prevents the lookahead for matching twice
                # like extra_id_100 and id_100
                continue

            # This will track every state
            # that stop matching, we need to stop tracking them.
            # If we look at "lowball", we're going to match "l" (add it to states), "o", "w", then
            # fail on "b", we need to remove 0 from the valid states.
            to_remove = set()
            # Whenever we found a match, we need to drop everything
            # this is a greedy algorithm, it will match on the first found token
            reset = False

            # In this case, we already have partial matches (But unfinished)
            for start, trie_pointer in states.items():
                if "" in trie_pointer:
                    # This is a final match, we need to reset and
                    # store the results in `offsets`.

                    # Lookahead to match longest first
                    # Important in case of extra_id_1 vs extra_id_100
                    # Here we are also actively looking for other earlier partial
                    # matches
                    # "[CLS]", "L", we need to match CLS even if L is special
                    for lookstart, looktrie_pointer in states.items():
                        if lookstart > start:
                            # This partial match is later, we can stop looking
                            break
                        elif lookstart < start:
                            # This partial match is earlier, the trie pointer
                            # was already updated, so index is + 1
                            lookahead_index = current + 1
                            end = current + 1
                        else:
                            # Here lookstart == start and
                            #      looktrie_pointer == trie_pointer
                            # It wasn't updated yet so indices are current ones
                            lookahead_index = current
                            end = current
                        next_char = text[lookahead_index] if lookahead_index < len(text) else None
                        if "" in looktrie_pointer:
                            start = lookstart
                            end = lookahead_index
                            skip = lookahead_index

                        while next_char in looktrie_pointer:
                            looktrie_pointer = looktrie_pointer[next_char]
                            lookahead_index += 1
                            if "" in looktrie_pointer:
                                start = lookstart
                                end = lookahead_index
                                skip = lookahead_index

                            if lookahead_index == len(text):
                                # End of string
                                break
                            next_char = text[lookahead_index]
                        # End lookahead

                    # Storing and resetting
                    offsets.append(start)
                    offsets.append(end)
                    reset = True
                    break
                elif current_char in trie_pointer:
                    # The current character being looked at has a match within the trie
                    # update the pointer (it will be stored back into states later).
                    trie_pointer = trie_pointer[current_char]

                    # Storing back the new pointer into the states.
                    # Partial matches got longer by one.
                    states[start] = trie_pointer
                else:
                    # The new character has not match in the trie, we need
                    # to stop keeping track of this partial match.
                    # We can't do it directly within the loop because of how
                    # python iteration works
                    to_remove.add(start)

            # Either clearing the full start (we found a real match)
            # Or clearing only the partial matches that didn't work.
            if reset:
                states = {}
            else:
                for start in to_remove:
                    del states[start]

            # If this character is a starting character within the trie
            # start keeping track of this partial match.
            if current >= skip and current_char in self.data:
                states[current] = self.data[current_char]

        # We have a cut at the end with states.
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                # This is a final match, we need to reset and
                # store the results in `offsets`.
                end = len(text)
                offsets.append(start)
                offsets.append(end)
                # Longest cut is always the one with lower start so the first
                # item so we need to break.
                break

        return self.cut_text(text, offsets)

    def cut_text(self, text, offsets):
        # We have all the offsets now, we just need to do the actual splitting.
        # We need to eventually add the first part of the string and the eventual
        # last part.
        offsets.append(len(text))
        tokens = []
        start = 0
        for end in offsets:
            if start > end:
                logger.error(
                    "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it"
                    " anyway."
                )
                continue
            elif start == end:
                # This might happen if there's a match at index 0
                # we're also preventing zero-width cuts in case of two
                # consecutive matches
                continue
            tokens.append(text[start:end])
            start = end

        return tokens
```

The Trie class implementation above provides an efficient data structure for token lookup and text segmentation. It maintains a tree-like structure where each node represents a character and paths from root to leaf nodes form complete tokens. The class offers two main methods: `add()` for inserting new tokens into the trie, and `split()` for segmenting input text based on the stored tokens. The split algorithm employs a greedy approach with lookahead to match the longest possible tokens first, tracking multiple potential matches simultaneously through a state machine. This implementation achieves O(n) complexity for text segmentation while handling edge cases like overlapping tokens and ensuring the longest matches are prioritized.

## Base Tokenizer Class

Most tokenizer implementations inherit from a base tokenizer class that provides common functionality. Here's a typical structure:

```python
class PreTrainedTokenizer(PreTrainedTokenizerBase):
def __init__(self, **kwargs):
        # 1. Init the parent class

        self.tokens_trie = Trie()

        # 2. init `_added_tokens_decoder` if child class did not
        if not hasattr(self, "_added_tokens_decoder"):
            self._added_tokens_decoder: Dict[int, AddedToken] = {}

        # 3. if a `added_tokens_decoder` is passed, we are loading from a saved tokenizer, we overwrite
        self._added_tokens_decoder.update(kwargs.pop("added_tokens_decoder", {}))
        self._added_tokens_encoder: Dict[str, int] = {k.content: v for v, k in self._added_tokens_decoder.items()}

        # 4 init the parent class
        super().__init__(**kwargs)

        # 4. If some of the special tokens are not part of the vocab, we add them, at the end.
        # the order of addition is the same as self.SPECIAL_TOKENS_ATTRIBUTES following `tokenizers`
        self._add_tokens(
            [token for token in self.all_special_tokens_extended if token not in self._added_tokens_encoder],
            special_tokens=True,
        )

        self._decode_use_source_tokenizer = False
```
The PreTrainedTokenizer class serves as the foundation for all tokenizer implementations in the Hugging Face Transformers library. It initializes essential components like the tokens trie for efficient token lookup and maintains two key dictionaries: 
`_added_tokens_decoder` for mapping token IDs to their corresponding AddedToken objects,
`_added_tokens_encoder` for the reverse mapping from token strings to IDs. 
The class handles special tokens by ensuring they are properly added to the vocabulary, even if not initially present, and maintains consistency in token handling across different model architectures. This base class provides critical functionality for token management, vocabulary handling, and special token processing that is inherited by specific tokenizer implementations.


## T5 Tokenization

### Key Features
- Unigram language model-based subword tokenization
- Treats the input as a sequence of Unicode characters
- No pre-tokenization required
- Built-in special token handling

### Implementation Details
- Vocabulary size: 32,000 tokens
- Uses `<pad>`, `</s>`, `<unk>` as special tokens
- Preserves whitespace by encoding it as `▁` (U+2581)

T5 uses SentencePiece tokenization, which is a language-independent subword tokenizer. 
T5 tokenizer utilizes Google's SentencePiece library for tokenization, which implements a unigram language model approach. This method treats tokenization as an unsupervised learning problem, where the vocabulary is optimized to maximize the likelihood of the training data. The algorithm iteratively refines the vocabulary by removing tokens that contribute least to the overall likelihood, making it particularly effective for handling multiple languages and out-of-vocabulary words. SentencePiece treats the input text as a sequence of Unicode characters, making it language-agnostic and capable of processing any language without specific pre-processing rules.

The core algorithm works as follows:

1. Initialize a large vocabulary (e.g., 100k tokens) from substrings in training corpus
2. Calculate loss for the corpus with current vocabulary:
   ```python
   def compute_loss(text, vocab):
       loss = 0
       for sentence in text:
           # Find optimal segmentation using Viterbi algorithm
           segments = viterbi_segment(sentence, vocab)
           # Calculate negative log likelihood
           loss += -sum(log(p(segment)) for segment in segments)
       return loss
   ```
3. For each token in vocabulary:
   - Compute loss when that token is removed
   - Store loss delta (impact of removal)
4. Remove tokens with smallest loss impact until target vocab size
5. Re-optimize token probabilities
6. Repeat steps 2-5 until convergence

For implementation details and the complete algorithm, refer to the [SentencePiece GitHub repository](https://github.com/google/sentencepiece).
Here's a simplified huggingface code of how T5's tokenizer processes text using SentencePiece:


```python
class T5Tokenizer(PreTrainedTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        legacy=None,
        **kwargs,
    ) -> None:
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        if additional_special_tokens is not None:
            extra_tokens = [x for x in additional_special_tokens if "<extra_id_" in str(x)]
            if len(extra_tokens) < 1:
                additional_special_tokens += [f"<extra_id_{i}>" for i in range(extra_ids)]
            elif extra_ids > 0 and extra_ids != len(extra_tokens):
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
                    " provided to T5Tokenizer. In this case the additional_special_tokens must include the extra_ids"
                    " tokens"
                )
        else:
            extra_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
            additional_special_tokens = extra_tokens

        # for legacy purpose, we keep this. Will be removed and tests updated. (when `added_tokens_decoder` is not passed as kwargs)
        self._added_tokens_decoder = {}
        for i in range(len(extra_tokens)):
            self._added_tokens_decoder[len(self.sp_model) - 1 + extra_ids - i] = AddedToken(
                f"<extra_id_{i}>", single_word=False, lstrip=True, rstrip=True, special=True, normalized=False
            )

        if legacy is None:
            logger.warning_once(
                f"You are using the default legacy behaviour of the {self.__class__}. This is"
                " expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you."
                " If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it"
                " means, and thouroughly read the reason why this was added as explained in"
                " https://github.com/huggingface/transformers/pull/24565"
            )
            legacy = True

        self.legacy = legacy
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))
        self.vocab_file = vocab_file
        self._extra_ids = extra_ids

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            legacy=legacy,
            **kwargs,
        )
```



## BERT Tokenization

### Key Features
- WordPiece vocabulary construction
- Case-sensitive
- Includes pre-tokenization step

### Implementation Details
- Vocabulary size: 30,522 tokens
- Special tokens: `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, `[UNK]`
- Handles unknown tokens with `##` prefix for subwords
BERT uses WordPiece tokenization, which is a subword tokenization algorithm developed by Google. Here's how the WordPiece algorithm works:
```python
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
```

## LLaMA Tokenization 

LLaMA uses a byte-level BPE tokenizer, which provides several advantages for handling multiple languages and special characters.

### Key Features
- Byte-level encoding ensures coverage of all possible inputs
- BPE (Byte-Pair Encoding) for subword tokenization
- No pre-tokenization required

### Implementation Details
```python
class LLaMATokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        use_default_system_prompt=False,
        spaces_between_special_tokens=False,
        legacy=None,
        add_prefix_space=True,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, normalized=False, special=True) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, normalized=False, special=True) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, normalized=False, special=True) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, normalized=False, special=True) if isinstance(pad_token, str) else pad_token

        if legacy is None:
            logger.warning_once(
                f"You are using the default legacy behaviour of the {self.__class__}. This is"
                " expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you."
                " If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it"
                " means, and thoroughly read the reason why this was added as explained in"
                " https://github.com/huggingface/transformers/pull/24565"
            )
            legacy = True

        self.legacy = legacy
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_default_system_prompt = use_default_system_prompt
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))
        self.add_prefix_space = add_prefix_space

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            use_default_system_prompt=use_default_system_prompt,
            spaces_between_special_tokens=spaces_between_special_tokens,
            legacy=legacy,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
```

```python
    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_spm_processor
    def get_spm_processor(self, from_slow=False):
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        if self.legacy or from_slow:  # no dependency on protobuf
            tokenizer.Load(self.vocab_file)
            return tokenizer
```
The LLaMA tokenizer initialization above shows several key components:

1. Configuration parameters:
   - Special tokens (`bos`, `eos`, `unk`, `pad`) with normalization flags
   - Control flags for token addition (`add_bos_token`, `add_eos_token`)
   - SentencePiece model parameters (`sp_model_kwargs`)
   - Legacy behavior handling for backwards compatibility

2. Token processing setup:
   - Converts string tokens to AddedToken objects with special flags
   - Initializes SentencePiece processor via `get_spm_processor()`
   - Configures prefix space and system prompt handling

Like T5, LLaMA also uses SentencePiece tokenization under the hood, though with some key differences in implementation. Both tokenizers leverage SentencePiece's efficient subword tokenization, but LLaMA operates on bytes rather than Unicode characters and uses a different vocabulary size and special token set.

Here's how the SentencePiece processor is typically initialized:

- Vocabulary size: 32,000 tokens
- Special tokens: `<s>`, `</s>`, `<unk>`, `<pad>`
- Handles UTF-8 encoding directly
- Merges are learned on bytes rather than unicode characters