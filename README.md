# Neural Symbolic Reasoner (NSR)

## Overview
The Neural Symbolic Reasoner (NSR) is a hybrid AI framework that bridges the gap between sub-symbolic connectionist systems (Neural Networks) and symbolic logic reasoning. It allows for the learning of robust representations from raw data while maintaining the interpretability and reasoning capabilities of formal logic.

## Key Features
- **Differentiable Logic Layer**: Integrates First-Order Logic (FOL) constraints directly into the neural network's loss function.
- **Symbolic Knowledge Injection**: Allows pre-existing knowledge bases (KB) to guide the learning process.
- **Explainable Inference**: Provides logic traces for decisions made by the neural components.
- **Robustness**: Shows improved performance in low-data regimes compared to pure deep learning models.

## Architecture
1.  **Perception Module**: A standard CNN/Transformer backbone for feature extraction.
2.  **Concept Layer**: Maps high-dimensional features to discrete symbolic concepts.
3.  **Reasoning Engine**: A differentiable Prolog-like engine that executes logic rules.
4.  **Feedback Loop**: Backpropagates reasoning errors to the perception module.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
See `examples/mnist_addition.py` for a classic DeepProbLog-style example.

## License
MIT

## Author
Olivier Robert-Duboille
