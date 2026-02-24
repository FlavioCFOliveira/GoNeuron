# Portuguese Generative Question Answering

This example showcases a **Generative QA** system specifically designed for Portuguese. It uses a causal Transformer architecture to synthesize answers based on a provided context and question.

## Technical Objectives

1.  **Contextual Reasoning**: Training the model to associate specific context tokens (`[CTX]`) with question tokens (`[Q]`) to produce accurate answers (`[A]`).
2.  **Special Token Handling**: Demonstrating the use of structural markers within the sequence to guide the model's logic.
3.  **Domain Adaptation**: Fine-tuning or training on Portuguese-specific facts and linguistic patterns.

## Key Implementation Details

### 1. Context-Question Structure
The model processes a combined sequence:
`[CTX] {historical_fact} [Q] {pergunta} [A] {resposta}`
By training on this concatenated format, the Transformer learns to attend to the context portion when generating the answer portion.

### 2. Causal Masking
The use of the `causal` flag in the Transformer block ensures that during training, the model cannot "see" the answer while processing the question, maintaining the integrity of the generative task.

## Execution

```bash
go run main.go
```
