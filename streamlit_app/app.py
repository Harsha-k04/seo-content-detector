import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from utils.parser import parse_url
from utils.features import compute_features
from utils.scorer import predict_quality

# === Load models ===
@st.cache_resource
def load_models():
    model = joblib.load('../models/quality_model_hybrid.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    pca = joblib.load('../models/pca.pkl')
    base_df = pd.read_csv('../data/features.csv')
    return model, scaler, pca, base_df

model, scaler, pca, df_base = load_models()

# === Helper for embeddings ===
def parse_embedding(x):
    try:
        return np.array(eval(x))
    except:
        return np.zeros(384)

# === Streamlit UI ===
st.set_page_config(page_title="SEO Content Quality & Duplicate Detector", layout="centered")
st.markdown("<style>div.block-container{padding-top:2rem;max-width:850px;margin:auto;} </style>", unsafe_allow_html=True)

st.title("🔍 SEO Content Quality & Duplicate Detector")
st.markdown("Analyze any webpage or text for **SEO quality**, **readability**, and **duplicate risk** — complete with live content preview and smart similarity insights.")

choice = st.radio("Choose Input Type:", ["URL", "Raw Text"])

# === URL Analysis ===
if choice == "URL":
    url = st.text_input("Enter webpage URL:")
    if st.button("Analyze URL"):
        with st.spinner("Fetching and analyzing content..."):
            text = parse_url(url)
            if text.strip():
                # === Content Preview ===
                st.subheader("📝 Content Preview")
                preview = text[:600] + "..." if len(text) > 600 else text
                st.text_area("Extracted Text", preview, height=180)

                # === Predict Quality ===
                pred, probs, new_embed = predict_quality(text, model, scaler, pca)

                st.subheader("📊 Quality Prediction")
                st.metric("Predicted Quality", pred)
                st.bar_chart({
                    "High": probs[model.classes_ == "High"][0],
                    "Medium": probs[model.classes_ == "Medium"][0],
                    "Low": probs[model.classes_ == "Low"][0],
                })

                # === Interpretive Summary ===
                if pred == "High":
                    st.success("✅ Excellent! Strong SEO quality — balanced readability, well-structured paragraphs, and good keyword mix.")
                elif pred == "Medium":
                    st.info("ℹ️ Fair quality — consider enhancing flow, adding subheadings, and improving sentence clarity.")
                else:
                    st.warning("⚠️ Low quality — short or hard-to-read text detected. Expand and simplify content for better SEO value.")

                # === Duplicate Check ===
                st.subheader("🪞 Duplicate Detection")

                base_embeddings = np.vstack(df_base['embedding'].apply(parse_embedding).values)
                base_reduced = pca.transform(base_embeddings)
                sim_scores = cosine_similarity(new_embed, base_reduced)[0]

                # --- FIX: avoid self-match ---
                if "url" in df_base.columns and url in df_base["url"].values:
                    self_idx = df_base.index[df_base["url"] == url].tolist()[0]
                    sim_scores[self_idx] = -1  # ignore same URL

                # --- Top 3 similar pages ---
                top_indices = np.argsort(sim_scores)[::-1][:3]
                top_similar = df_base.iloc[top_indices][['url']].copy()
                top_similar['Similarity Score'] = sim_scores[top_indices]

                most_similar = top_similar.iloc[0]['url']
                sim_val = top_similar.iloc[0]['Similarity Score']

                st.write(f"**Most similar page:** [{most_similar}]({most_similar})  \n**Similarity Score:** {sim_val:.3f}")
                if sim_val > 0.85:
                    st.error("🚨 Duplicate or near-duplicate content detected!")
                elif sim_val > 0.70:
                    st.warning("⚠️ Partial overlap detected — consider rephrasing sections to improve originality.")
                else:
                    st.success("✅ Unique content — no strong overlaps found.")

                with st.expander("🔗 Top 3 Similar Pages"):
                    st.dataframe(top_similar)

                # === Download Report ===
                report_data = pd.DataFrame({
                    "URL": [url],
                    "Predicted Quality": [pred],
                    "High Probability": [probs[model.classes_ == 'High'][0]],
                    "Medium Probability": [probs[model.classes_ == 'Medium'][0]],
                    "Low Probability": [probs[model.classes_ == 'Low'][0]],
                    "Most Similar Page": [most_similar],
                    "Similarity Score": [sim_val]
                })
                csv = report_data.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Analysis Report", csv, "seo_analysis_report.csv", "text/csv")
            else:
                st.error("Failed to fetch or parse webpage content.")

# === Raw Text Analysis ===
else:
    text_input = st.text_area("Paste or type text here:")
    if st.button("Analyze Text"):
        with st.spinner("Analyzing content..."):
            if text_input.strip():
                st.subheader("📝 Content Preview")
                preview = text_input[:600] + "..." if len(text_input) > 600 else text_input
                st.text_area("Analyzed Text", preview, height=180)

                pred, probs, _ = predict_quality(text_input, model, scaler, pca)
                st.subheader("📊 Quality Prediction")
                st.metric("Predicted Quality", pred)
                st.bar_chart({
                    "High": probs[model.classes_ == "High"][0],
                    "Medium": probs[model.classes_ == "Medium"][0],
                    "Low": probs[model.classes_ == "Low"][0],
                })

                if pred == "High":
                    st.success("✅ Excellent! The content is well-written and SEO-friendly.")
                elif pred == "Medium":
                    st.info("ℹ️ Moderate quality — refine structure and clarity.")
                else:
                    st.warning("⚠️ Low quality — improve readability and keyword usage.")

                report_data = pd.DataFrame({
                    "Input Type": ["Text"],
                    "Predicted Quality": [pred],
                    "High Probability": [probs[model.classes_ == 'High'][0]],
                    "Medium Probability": [probs[model.classes_ == 'Medium'][0]],
                    "Low Probability": [probs[model.classes_ == 'Low'][0]]
                })
                csv = report_data.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Text Analysis Report", csv, "text_analysis_report.csv", "text/csv")
            else:
                st.error("Please enter text before analyzing.")
