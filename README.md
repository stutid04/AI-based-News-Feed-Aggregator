🧭 MBIC — Media Bias \& Information Classifier



MBIC (Media Bias \& Information Classifier) is a \*\*Streamlit-based NLP application\*\* that detects and visualizes linguistic bias in news headlines and short texts using a \*\*locally fine-tuned DistilBERT model\*\*.



The app combines:

\- 🤖 Transformer-based text classification  

\- 📰 Live news ingestion via NewsAPI  

\- 📊 Interactive visual analytics (gauges, charts)  



---



\## 🚀 Features



\### 📰 Live News Bias Analysis

\- Fetches real-time news headlines using \*\*NewsAPI\*\*

\- Default auto-load: `technology` • `popularity` • `10 articles`

\- Each headline is:

&nbsp; - Classified as \*Biased / Non-biased\*

&nbsp; - Shown on a \*\*graded bias spectrum\*\*

\- Graceful failure handling:

&nbsp; - Displays \*\*“API offline”\*\* banner if NewsAPI is unavailable



\### 🔎 Custom Text Classification

\- Paste any headline or paragraph

\- Adjustable \*\*binary classification threshold\*\*

\- Outputs:

&nbsp; - Binary label (Biased / Non-biased)

&nbsp; - Probability scores

&nbsp; - Bias gauge visualization



\### 📊 Model Overview \& Results

\- Explains:

&nbsp; - Model architecture

&nbsp; - Dataset statistics

&nbsp; - Training configuration

&nbsp; - Evaluation metrics

\- Includes:

&nbsp; - Accuracy \& F1 scores

&nbsp; - Label distribution charts

&nbsp; - Sample predictions

&nbsp; - Limitations \& future work



---



\## 🧠 Model Details



\- \*\*Base Model:\*\* `distilbert-base-uncased`

\- \*\*Task:\*\* Binary text classification  

\- \*\*Labels:\*\*

&nbsp; - `Non-biased` → 0

&nbsp; - `Biased` → 1

\- \*\*Frameworks:\*\*  

&nbsp; - 🤗 Transformers  

&nbsp; - PyTorch  



\### Training Summary

\- Dataset size: ~1,550 labeled samples

\- Split: 80% train / 20% test

\- Optimizer: AdamW

\- Learning rate: 2e-5

\- Epochs: 4

\- Device: Apple MPS / CPU fallback



---



\## 📈 Evaluation Metrics



| Metric       | Score |

| Accuracy     | 73.6% |

| F1 (macro)   | 0.68  |

| F1 (weighted)| 0.72 |



The model learns bias cues such as:

\- Emotionally loaded verbs

\- Subjective framing

\- Accusatory language



---



\## 🗂️ Project Structure



```text

├── app.py                     # Streamlit application

├── README.md                  # Project documentation

├── labeled\_dataset.xlsx       # Training dataset

├── out/

│   └── distilbert-mbic-binary/

│       └── best/              # Saved fine-tuned model

└── .streamlit/

&nbsp;   └── secrets.toml           # NewsAPI key (not committed)



