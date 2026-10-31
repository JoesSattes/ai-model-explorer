# Third-party files

`llamacpp_tensor_mapping.py` is an unmodified copy of `gguf-py/gguf/tensor_mapping.py` from
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT License, Copyright (c) 2023-2024 The ggml authors).
It is not imported or executed. `../gguf_baseline.py` parses it with Python's `ast` module to build
`../gguf_roles.json`, the name table used by the llama.cpp baseline in `eval.js` and `realbench.js`.
