# Dialogue Bot Transformer

A character-level dialogue model trained to simulate interactive conversations. It demonstrates the use of Transformers for mapping queries to responses in a conversational context.

## Technical Objectives

1.  **Interactive Inference**: Implementing a real-time loop for user input and model response generation.
2.  **Sequence-to-Sequence Modeling**: Learning the relationship between different speakers' turns within a single Transformer context.

## Key Implementation Details

### 1. Token-by-Token Generation
The bot generates responses autoregressively, feeding its own predicted characters back as input until an end-of-sequence marker or maximum length is reached.

### 2. Conversational Context
The model is trained on dialogue pairs, learning to identify the transition from a user prompt to a bot response through learned sequence patterns.

## Execution

```bash
go run main.go
```
