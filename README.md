# 📘 SEO Content Quality & Duplicate Detector

## 📄 Overview

This project detects **SEO content quality** and checks for **duplicate or overlapping articles** using **Machine Learning** and **Natural Language Processing (NLP)**.

It evaluates web content for:

- ✅ Readability  
- ✅ Keyword density  
- ✅ Structure and text complexity  
- ✅ Semantic similarity with existing pages  

A fine-tuned **Random Forest model** with **SentenceTransformer embeddings** powers both **URL-based content** and **raw text** analysis.  
The visually enhanced **Streamlit dashboard** provides real-time insights, charts, interpretive feedback, and downloadable reports.

---

## 🧠 Tech Stack

| Category | Technologies |
|-----------|---------------|
| **Frontend** | Streamlit |
| **Backend / ML** | Python, Scikit-learn, Joblib |
| **NLP & Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Web Parsing** | BeautifulSoup, Requests |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Streamlit Metrics, Bar Charts |
| **Model Persistence** | Joblib |

---

## 📂 Project Structure

Lead Walnut/
│
├── data/
│ ├── data.csv
│ ├── extracted_content.csv
│ └── features.csv
│
├── models/
│ ├── quality_model_hybrid.pkl
│ ├── scaler.pkl
│ └── pca.pkl
│
├── streamlit_app/
│ ├── app.py
│ └── utils/
│ ├── parser.py
│ ├── features.py
│ └── scorer.py
│
├── notebooks/
│ └── seo_pipeline.ipynb
│
├── requirements.txt
└── README.md

## ⚙️ Installation Guide

### 1️⃣ Clone or Download

```bash
git clone https://github.com/<your-username>/seo-content-detector.git
cd seo-content-detector
```
### 2️⃣ Create Virtual Environment

```bash
conda create -n seo_detector python=3.9 -y
conda activate seo_detector
```
### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
cd streamlit_app
streamlit run app.py
```
The app will open automatically in your browser:
👉 http://localhost:8501

## 🧩 How It Works

### 🔹 Step 1 — Content Extraction
Uses **BeautifulSoup** to extract clean text from webpages (`parser.py`).

---

### 🔹 Step 2 — Feature Engineering
Calculates key metrics:
- Word count  
- Sentence count  
- Readability (**Flesch Reading Ease**)  
- Keyword density  
- SentenceTransformer embeddings (`features.py`)

---

### 🔹 Step 3 — Quality Scoring
The trained **Random Forest model** predicts content quality as **High**, **Medium**, or **Low** using textual + embedding features (`scorer.py`).

---

### 🔹 Step 4 — Duplicate Detection
Computes **cosine similarity** between embeddings and existing dataset pages to identify **duplicate** or **partially overlapping** content.

---

### 🔹 Step 5 — Streamlit Interface
Interactive dashboard with:
- 📝 Content preview  
- 📊 Probability bar chart  
- 💡 Interpretive summaries  
- 🪞 Duplicate detection results  
- 💾 Downloadable analysis report  

---

## 🧾 Example Output

| Input Type | Example |
|-------------|----------|
| **URL** | [https://www.cisa.gov/news-events/news/10-essential-cybersecurity-tips](https://www.cisa.gov/news-events/news/10-essential-cybersecurity-tips) |
| **Predicted Quality** | Low |
| **Top Match** | [https://www.varonis.com/blog/cybersecurity-tips](https://www.varonis.com/blog/cybersecurity-tips) |
| **Similarity Score** | 0.86 *(Near-duplicate detected)* |

---

## 🌟 Key Features

- 🌐 URL or raw text input  
- 🧠 ML-based quality detection (**High / Medium / Low**)  
- 🔍 Duplicate detection using embeddings  
- 📊 Real-time visualizations (bar charts, metrics)  
- 💾 Downloadable CSV reports  
- 🪞 Top 3 similar pages table  
- 🚫 Self-similarity filtering (skips same-page match)  

---

## 🚀 Deploy to Streamlit Cloud

1. Push this project to a **public GitHub repository**.  
2. Go to 👉 [https://share.streamlit.io](https://share.streamlit.io).  
3. Connect your repo and configure:
   - **Main file:** `streamlit_app/app.py`  
   - **Python version:** `3.9`  
4. Click **Deploy** ✅  

---

## 🧪 Model Performance

| Metric | Score |
|---------|-------|
| **Accuracy** | 0.88 |
| **Precision** | 0.89 |
| **Recall** | 0.88 |
| **F1-Score** | 0.88 |

> Tuned **Random Forest Classifier** trained on 81 web articles using hybrid textual + embedding features.

---
🖥️ **Live Demo:** [Click here to try it out on Streamlit Cloud](https://seo-content-detector-jwprwxvqsz9bwsamtpdefb.streamlit.app/)


## 🧑‍💻 Developed by

**Harsha K**  
*Lead Walnut — Data Science Project*  

> “Turning raw text into SEO insights with NLP & Machine Learning.”
