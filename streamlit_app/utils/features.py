import numpy as np
import textstat

def compute_features(text):
    """Compute numeric SEO + readability features from text."""
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    try:
        flesch = textstat.flesch_reading_ease(text)
    except:
        flesch = 0
    keyword_density = len(set(text.lower().split())) / max(word_count, 1) * 100
    readability_bin = np.digitize(flesch, [0, 30, 50, 70, 100])
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'flesch_reading_ease': flesch,
        'keyword_density': keyword_density,
        'readability_bin': readability_bin
    }
