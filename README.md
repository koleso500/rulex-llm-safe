# Evaluating Logic Learning Machine model with SAFE-AI Metrics

This repository contains the experimental pipeline used in the paper:

**Evaluating Logic Learning Machine Model with SAFE-AI Metrics**  
World Conference on eXplainable AI (XAI 2026, Late-Breaking Track)

The project compares the Rulex Logic Learning Machine (LLM) with classical machine learning models using SAFE-AI metrics across three data modalities:

- Tabular — HMDA mortgage dataset
- Images — Brain Cancer MRI dataset
- Text — Financial PhraseBank dataset


## Rulex LLM model

The Logic Learning Machine (LLM) is available on the Rulex platform:

https://www.rulex.ai/

The same SAFE-AI evaluation pipeline was applied to the Logic Learning Machine model.  
However, the internal implementation of Logic Learning Machine is proprietary and therefore not publicly available in this repository.

## Installation

### CUDA 11.8 Installation (GPU)
```bash
pip install -r requirements-cuda.txt
```

## Running experiments

Run the experiments using:

```bash
python hmda.py
python images.py
python fin_text.py
