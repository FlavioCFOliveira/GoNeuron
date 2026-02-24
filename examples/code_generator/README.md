# Character-Level Code Generator

This example trains a Transformer to generate **Go source code**. It highlights the model's ability to learn complex syntactical structures, indentation, and keyword usage.

## Technical Objectives

1.  **Syntax Learning**: Capturing the rigid structural requirements of a programming language at the character level.
2.  **Contextual Logic**: Maintaining long-range dependencies (e.g., closing braces, matching parentheses).

## Key Implementation Details

### 1. Training on Source Code
The model is fed large blocks of Go source code, learning the statistical likelihood of character sequences like `func `, `if err != nil`, and package declarations.

### 2. Structural Integrity
By using multiple Transformer heads, the model can attend to both the local context (the current word) and the global context (the function or package scope).

## Execution

```bash
go run main.go
```
