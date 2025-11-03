import numpy as np
from sentence_transformers import SentenceTransformer
from utils.features import compute_features

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def predict_quality(text, model, scaler, pca):
    """Predict content quality using trained model and embeddings."""
    feats = compute_features(text)
    embed = embed_model.encode([text])
    reduced_embed = pca.transform(embed)
    X = np.hstack([[feats['word_count'], feats['sentence_count'],
                    feats['flesch_reading_ease'], feats['keyword_density'],
                    feats['readability_bin']], reduced_embed[0]])
    X_scaled = scaler.transform([X])
    pred = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]
    return pred, probs, reduced_embed
