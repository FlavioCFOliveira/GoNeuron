# GGUF Compatibility Expert Memory

## GGUF Export in GoNeuron

- Implementation resides in `internal/net/net.go` and `internal/net/gguf.go`.
- `SaveGGUF(filename string)`: Exports model to GGUF v3 in F32.
- `SaveGGUFExt(filename string, tensorType GGMLType)`: Supports F32 and F16 export.
- Uses `NamedParamProvider` interface from layers to collect tensors.
- Mapping GoNeuron names to GGUF standard:
    - `weights` -> `weight`
    - `biases` -> `bias`
    - `gamma` -> `weight`
    - `beta` -> `bias`
    - Transformer sub-layers:
        - `mha_wQ_` -> `attn_q.`
        - `mha_wK_` -> `attn_k.`
        - `mha_wV_` -> `attn_v.`
        - `mha_wOut_` -> `attn_output.`
        - `ff1_` -> `ffn_up.`
        - `ff2_` -> `ffn_down.`
        - `norm1_` -> `attn_norm.`
        - `norm2_` -> `ffn_norm.`
- GGUF dimensions are exported in reverse order (GGUF requirement).
- Alignment is set to 32 bytes.

## Performance
- Minimal allocations by pre-calculating offsets.
- F16 conversion implemented manually for F32 to F16 export.

## Metadata
- `general.architecture`: `goneuron`
- `general.name`: `GoNeuron Model`
- `general.alignment`: `32`
- `goneuron.layer_count`: Number of layers.
- `goneuron.loss_type`: Go type name of the loss function.
