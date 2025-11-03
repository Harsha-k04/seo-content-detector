# 📘 SEO Content Quality & Duplicate Detector

## 🧩 Project Overview
This project analyzes web content to assess **SEO quality** and detect **duplicate or overlapping articles** using **Machine Learning** and **NLP**. It evaluates readability, keyword density, structure, and semantic similarity between webpages.  
A **Random Forest model** combined with **SentenceTransformer embeddings** powers a live **Streamlit app** for real-time analysis.

---

## ⚙️ Setup Instructions
```bash
git clone https://github.com/Harsha-k04/seo-content-detector.git
cd seo-content-detector
pip install -r requirements.txt
jupyter notebook notebooks/seo_pipeline.ipynb
```
## 🚀 Quick Start

Ensure `data/features.csv` and trained models (`models/quality_model_hybrid.pkl`, `scaler.pkl`, `pca.pkl`) are in place.

Run the app:
```bash
cd streamlit_app
streamlit run app.py
```
Enter a webpage URL or paste raw text to analyze its SEO quality and duplicate risk.
## 🌐 Deployed Streamlit URL

👉 [https://seo-content-detector.streamlit.app](https://seo-content-detector-jwprwxvqsz9bwsamtpdefb.streamlit.app/)

---

## 🧠 Key Decisions

- **Libraries:** Used `scikit-learn`, `SentenceTransformer`, and `BeautifulSoup` for a lightweight yet powerful pipeline.  
- **Parsing Approach:** Extracted clean visible text using `BeautifulSoup` and removed scripts/styles for better readability scoring.  
- **Similarity Threshold:** A cosine similarity score >0.85 marks near-duplicates, 0.70–0.85 indicates partial overlap.  
- **Model Choice:** Random Forest chosen for its interpretability and robust performance on small datasets (81 pages).  

---

## 📊 Results Summary

- **Model Accuracy:** 0.88  
- **F1-Score:** 0.88  
- **Duplicates Found:** 6 near-duplicates detected in 81 samples  
- **Sample Quality Scores:**  
  - High: 14  
  - Medium: 42  
  - Low: 25  

**Example:**

| Input | Predicted Quality | Match | Similarity |
|-------|-------------------|--------|-------------|
| https://www.cisa.gov/news-events/news/10-essential-cybersecurity-tips | Low | https://www.varonis.com/blog/cybersecurity-tips | 0.86 |

---

## ⚠️ Limitations

- Small dataset (81 samples) limits generalization.  
- Readability estimation doesn’t account for non-textual factors (images, layout).  
- Near-duplicate detection relies on embedding model precision; semantic nuance loss possible.  

---

## 👨‍💻 Developed by

**Harsha K**  
Lead Walnut — Data Science Project  

> “Turning raw text into SEO insights with NLP & Machine Learning.”
