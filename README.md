# MBIC — Media Bias & Information Classifier

A Natural Language Processing (NLP) project that detects **bias in news headlines and short text** using a fine-tuned **DistilBERT transformer model**, with an interactive Streamlit dashboard and live news integration.

---

## Overview

Media content often contains subtle bias through emotionally charged or opinionated language.
This project aims to automatically classify text as:

* **Biased**
* **Non-biased**

The system combines transformer-based NLP with a real-time web interface for analysis and visualization.

---

## Problem Statement

Determine whether a given piece of text:

* Uses neutral, factual language → Non-biased
* Uses emotionally loaded or persuasive language → Biased

---

## Dataset

Dataset used: **MBIC — Media Bias Annotation Dataset**

* Official: https://zenodo.org/records/4474336
* Kaggle: https://www.kaggle.com/datasets/timospinde/mbic-a-media-bias-annotation-dataset

### Dataset Details

| Feature          | Value                   |
| ---------------- | ----------------------- |
| Total samples    | 1551                    |
| Train/Test split | 80% / 20%               |
| Classes          | 2 (Biased / Non-biased) |

### Label Mapping

```python
Non-biased → 0  
Biased     → 1
```

---

## Model

Model used:

```python
distilbert-base-uncased
```

### Why DistilBERT?

* Smaller and faster than BERT
* Retains strong performance
* Suitable for low-resource environments

---

## Architecture

```
Input Text
   ↓
Tokenizer (AutoTokenizer)
   ↓
Embeddings
   ↓
DistilBERT Encoder (6 layers)
   ↓
Classification Head
   ↓
Softmax
   ↓
Bias Probability
```

---

## Training Pipeline

File: `trainbias.py`

### Steps

1. Load dataset
2. Clean and preprocess text
3. Convert labels to numeric format
4. Perform train/test split
5. Tokenize text
6. Fine-tune DistilBERT
7. Save trained model

### Optimization

The encoder is frozen to reduce computation:

```python
for p in model.base_model.parameters():
    p.requires_grad = False
```

This allows training on low-memory systems.



---

## Data Preprocessing

File: `datasetcleaning.py`

* Remove missing values
* Clean text (normalize spacing)
* Map labels to integers
* Stratified train/test split
* Convert to HuggingFace Dataset format



---

## Evaluation Metrics

| Metric        | Score |
| ------------- | ----- |
| Accuracy      | 73.6% |
| F1 (macro)    | 0.68  |
| F1 (weighted) | 0.72  |

---

## Application (Streamlit)

File: `app.py`

### Features

**Live News**

* Fetches headlines via NewsAPI
* Classifies bias in real time

**Text Classification**

* Accepts user input
* Outputs label and probability

**Model Overview**

* Displays dataset, training configuration, and metrics

---

## News Fetching

API used:

```
https://newsapi.org/v2/everything
```

Workflow:

```
User Query → NewsAPI → Headlines → Model → Prediction
```

---

## Project Structure

```
project/
│
├── app.py
├── trainbias.py
├── datasetcleaning.py
├── labeled_dataset.xlsx
├── requirements.txt
├── Dockerfile
│
└── out/
    └── distilbert-mbic-binary/
        └── best/
            ├── model files
            ├── tokenizer
            └── config
```

---

## How It Works

```
Pretrained DistilBERT (HuggingFace)
           ↓
Fine-tuned on MBIC dataset
           ↓
Saved locally
           ↓
Loaded into Streamlit app
           ↓
Real-time bias classification
```

---

## Novelty

* Combines transformer-based NLP with live news analysis
* Focuses on **bias detection**, not just sentiment
* Real-time interactive dashboard
* Probability-based bias spectrum

---

## Limitations

* Small dataset (~1500 samples)
* Binary classification only
* English language only
* Limited contextual understanding

---

## Future Work

* Expand dataset size
* Multi-class bias classification
* Add explainability (LIME / SHAP)
* Support multiple languages
* Integrate LLM-based explanations
* Add RAG-based fact verification

---

## Technologies Used

* Transformers (DistilBERT)
* PyTorch
* Streamlit
* Pandas
* HuggingFace Datasets
* Evaluate
* Plotly / Altair

---

## License

For academic and educational use.

---

## Acknowledgements

* HuggingFace Transformers
* MBIC Dataset Authors
* Streamlit
