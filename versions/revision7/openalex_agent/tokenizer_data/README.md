# Study tokenizer

The `o200k_base.tiktoken` public BPE ranks and segmentation pattern come from
[openai/tiktoken](https://github.com/openai/tiktoken), under the accompanying MIT
license. Ranks source: `https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken`.
SHA-256: `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`.

The ranks are packaged to prevent network downloads during tests or diagnostics.
The explicit study encoding is not asserted to match Terra's undocumented native
tokenizer or provider-internal message framing. Actual provider usage stays separate.
