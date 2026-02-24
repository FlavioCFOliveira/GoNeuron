# Character-Level Poetry Generation

This example trains a Transformer model to generate poetry in the style of **Luís de Camões**, one of the greatest poets in the Portuguese language. It operates at the **character level**, learning the rhythm, vocabulary, and structural nuances of classical verse.

## Technical Objectives

1.  **Character Tokenization**: Treating individual characters as atomic units, allowing the model to learn morphology and punctuation directly.
2.  **Stylistic Transfer**: Capturing the unique linguistic "fingerprint" of a specific historical author.
3.  **Weight Exporting**: Saving trained weights to a binary format for reuse in lightweight inference applications.

## Key Implementation Details

### 1. Character-to-Vector Mapping
By using a character-level embedding, the model gains a small vocabulary size (e.g., ~100 unique characters) but must learn to construct entire words and sentences from scratch.

### 2. Positional Awareness
The `PositionalEncoding` layer is crucial here, as it allows the Transformer to understand the meter and line structure essential for classical poetry.

## Execution

```bash
go run main.go
```
