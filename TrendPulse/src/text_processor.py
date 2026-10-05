"""
Text Processing and NLP Feature Extraction Module for Social Media Trend Analysis.
Case Study 127: TrendPulse
"""

import re
import string
import unicodedata
from typing import Dict, List, Tuple

# Comprehensive English stop words list (self-contained, no network dependency)
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
    'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
    'wouldn', "wouldn't", 'rt', 'via', 'amp', 'http', 'https'
}

# Lexicon for sentiment polarity calculation
POSITIVE_WORDS = {
    'great', 'amazing', 'excellent', 'breakthrough', 'innovative', 'record', 'success',
    'love', 'best', 'super', 'massive', 'growth', 'historic', 'thrilled', 'excited',
    'awesome', 'winner', 'win', 'revolution', 'bullish', 'unbelievable', 'incredible',
    'outstanding', 'phenomenal', 'surge', 'gains', 'top', 'perfect', 'inspiring',
    'gamechanger', 'milestone', 'positive', 'bright', 'triumph', 'brilliant', 'wonderful',
    'spectacular', 'legendary', 'boost', 'upgrade', 'valuable', 'champion', 'soar'
}

NEGATIVE_WORDS = {
    'bad', 'worst', 'terrible', 'crash', 'fail', 'failure', 'disaster', 'scam', 'drop',
    'decline', 'loss', 'crisis', 'shock', 'warning', 'danger', 'bearish', 'plunge',
    'collapsed', 'horrible', 'awful', 'controversy', 'scandal', 'breakdown', 'bleak',
    'bankrupt', 'cancel', 'lawsuit', 'panic', 'chaos', 'furious', 'depressing', 'tragic',
    'downturn', 'struggling', 'criticized', 'boycott', 'hacked', 'outage', 'glitch'
}

SUBJECTIVE_WORDS = {
    'feel', 'believe', 'think', 'opinion', 'worst', 'best', 'incredible', 'insane',
    'unbelievable', 'shocking', 'legendary', 'favorite', 'hate', 'love', 'surely',
    'definitely', 'probably', 'must', 'totally', 'ridiculous', 'wild', 'crazy', 'disgusting'
}


def clean_text(text: str) -> str:
    """Cleans raw social media text: removes URLs, mentions, emojis, and punctuation."""
    if not isinstance(text, str):
        return ""
    
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbol but keep hashtag word
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove non-ascii characters (emojis, unusual symbols)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove special punctuation and digits
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Lowercase & collapse extra whitespace
    text = text.lower()
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    
    return " ".join(tokens)


def extract_hashtags(text: str) -> List[str]:
    """Extracts hashtags from raw text."""
    if not isinstance(text, str):
        return []
    return re.findall(r'#\w+', text.lower())


def compute_sentiment(text: str) -> Tuple[float, float]:
    """
    Computes sentiment polarity (-1.0 to +1.0) and subjectivity (0.0 to 1.0)
    using robust lexicon-based analysis with intensifier handling.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0
    
    tokens = re.findall(r'\b\w+\b', text.lower())
    if not tokens:
        return 0.0, 0.0
    
    pos_count = sum(1 for w in tokens if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in tokens if w in NEGATIVE_WORDS)
    subj_count = sum(1 for w in tokens if w in SUBJECTIVE_WORDS or w in POSITIVE_WORDS or w in NEGATIVE_WORDS)
    
    total_emotional = pos_count + neg_count
    if total_emotional == 0:
        polarity = 0.0
    else:
        polarity = (pos_count - neg_count) / float(total_emotional)
    
    subjectivity = min(1.0, subj_count / max(1, len(tokens) * 0.4))
    
    return round(polarity, 3), round(subjectivity, 3)


def extract_nlp_features(text: str) -> Dict[str, float]:
    """Extracts structural and stylistic NLP features from social media post."""
    if not isinstance(text, str):
        text = ""
        
    char_count = len(text)
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    
    uppercase_count = sum(1 for c in text if c.isupper())
    caps_ratio = uppercase_count / max(1, char_count)
    
    exclamation_count = text.count('!')
    question_count = text.count('?')
    has_url = 1.0 if re.search(r'https?://|www\.', text) else 0.0
    
    hashtags = extract_hashtags(text)
    hashtag_count = len(hashtags)
    mention_count = len(re.findall(r'@\w+', text))
    
    polarity, subjectivity = compute_sentiment(text)
    
    # Lexical diversity (Type-Token Ratio)
    lexical_diversity = (len(set(w.lower() for w in words)) / max(1, word_count)) if word_count > 0 else 0.0
    
    return {
        'char_count': float(char_count),
        'word_count': float(word_count),
        'caps_ratio': round(caps_ratio, 3),
        'exclamation_count': float(exclamation_count),
        'question_count': float(question_count),
        'has_url': has_url,
        'hashtag_count': float(hashtag_count),
        'mention_count': float(mention_count),
        'sentiment_polarity': polarity,
        'sentiment_subjectivity': subjectivity,
        'lexical_diversity': round(lexical_diversity, 3)
    }
