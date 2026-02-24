# Portuguese Word Embeddings

This example focuses on the unsupervised learning of **Word Embeddings** for the Portuguese language. It demonstrates how to train a model that captures semantic relationships between words in a high-dimensional vector space.

## Technical Objectives

1.  **Semantic Vector Spaces**: Learning word representations where synonymous words are geometrically close.
2.  **Unsupervised Context Learning**: Predicting a target word from its surroundings or vice-versa.

## Key Implementation Details

### 1. Embedding Layer Training
The core of the example is the training of the `Embedding` layer weights. Once trained, these weights can be extracted and used as a pre-trained feature set for other Portuguese NLP tasks like sentiment analysis or NER.

### 2. Linguistic Specificity
By training on Portuguese text corpora, the model captures language-specific nuances, gender agreement, and morphological patterns that are often lost in English-centric pre-trained models.

## Execution

```bash
go run main.go
```
