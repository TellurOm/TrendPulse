"""
Data Preprocessing, Cleaning, and Feature Engineering Pipeline.
Transforms raw social media post text and metadata into machine learning feature matrices.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from text_processor import clean_text, extract_nlp_features


class TrendDataPreprocessor:
    """End-to-end preprocessor for social media trend data."""
    
    def __init__(self, max_tfidf_features: int = 300):
        self.max_tfidf_features = max_tfidf_features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.label_encoder = LabelEncoder()
        
        self.categorical_cols = ['platform', 'category', 'media_type']
        self.numerical_cols = [
            'log_author_followers', 'author_verified', 'hashtag_count',
            'char_count', 'word_count', 'sentiment_polarity', 'sentiment_subjectivity',
            'question_count', 'exclamation_count', 'has_url',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        self.feature_names = []
        self.is_fitted = False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineers statistical, temporal, and textual features."""
        df = df.copy()
        
        # Log-transform power-law follower count
        df['log_author_followers'] = np.log1p(df['author_followers'].fillna(100).clip(lower=0))
        
        # Cyclical encoding of hour and day of week
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
        
        # Clean text
        df['cleaned_text'] = df['post_text'].apply(clean_text)
        
        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Fits all encoders and scalers on dataset and returns feature matrices."""
        df_featured = self.engineer_features(df)
        
        # 1. TF-IDF features
        tfidf_features = self.tfidf_vectorizer.fit_transform(df_featured['cleaned_text']).toarray()
        tfidf_cols = [f"tfidf_{w}" for w in self.tfidf_vectorizer.get_feature_names_out()]
        
        # 2. One-hot encode categorical features
        cat_features = self.one_hot_encoder.fit_transform(df_featured[self.categorical_cols])
        cat_cols = list(self.one_hot_encoder.get_feature_names_out(self.categorical_cols))
        
        # 3. Scale numerical features
        num_features = self.scaler.fit_transform(df_featured[self.numerical_cols])
        
        # Concatenate all feature arrays
        X = np.hstack([num_features, cat_features, tfidf_features])
        self.feature_names = self.numerical_cols + cat_cols + tfidf_cols
        
        # Encode Target (virality_tier: Low, Moderate, Viral)
        # Ensure consistent label ordering
        self.label_encoder.fit(['Low', 'Moderate', 'Viral'])
        y = self.label_encoder.transform(df_featured['virality_tier'])
        
        self.is_fitted = True
        return X, y, df_featured

    def transform_single_post(self, post_text: str, platform: str, category: str,
                              media_type: str, followers: int, hour: int,
                              day_of_week: int, is_verified: int = 0) -> np.ndarray:
        """Transforms a single user input into model feature vector for real-time inference."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted or loaded before calling transform_single_post.")
            
        nlp_dict = extract_nlp_features(post_text)
        cleaned = clean_text(post_text)
        
        log_followers = np.log1p(max(0, followers))
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        day_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        day_cos = np.cos(2 * np.pi * day_of_week / 7.0)
        
        num_dict = {
            'log_author_followers': log_followers,
            'author_verified': float(is_verified),
            'hashtag_count': float(nlp_dict['hashtag_count']),
            'char_count': float(nlp_dict['char_count']),
            'word_count': float(nlp_dict['word_count']),
            'sentiment_polarity': float(nlp_dict['sentiment_polarity']),
            'sentiment_subjectivity': float(nlp_dict['sentiment_subjectivity']),
            'question_count': float(nlp_dict['question_count']),
            'exclamation_count': float(nlp_dict['exclamation_count']),
            'has_url': float(nlp_dict['has_url']),
            'hour_sin': float(hour_sin),
            'hour_cos': float(hour_cos),
            'day_sin': float(day_sin),
            'day_cos': float(day_cos)
        }
        
        num_df = pd.DataFrame([num_dict])[self.numerical_cols]
        num_scaled = self.scaler.transform(num_df)
        
        cat_df = pd.DataFrame([{
            'platform': platform,
            'category': category,
            'media_type': media_type
        }])[self.categorical_cols]
        cat_encoded = self.one_hot_encoder.transform(cat_df)
        
        tfidf_vec = self.tfidf_vectorizer.transform([cleaned]).toarray()
        
        X_vec = np.hstack([num_scaled, cat_encoded, tfidf_vec])
        return X_vec

    def save(self, filepath: str = "models/preprocessor_pipeline.joblib"):
        """Saves fitted preprocessor components."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler,
            'one_hot_encoder': self.one_hot_encoder,
            'label_encoder': self.label_encoder,
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols,
            'feature_names': self.feature_names
        }, filepath)
        print(f"Preprocessor pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/preprocessor_pipeline.joblib"):
        """Loads pre-trained preprocessor components."""
        data = joblib.load(filepath)
        instance = cls()
        instance.tfidf_vectorizer = data['tfidf_vectorizer']
        instance.scaler = data['scaler']
        instance.one_hot_encoder = data['one_hot_encoder']
        instance.label_encoder = data['label_encoder']
        instance.categorical_cols = data['categorical_cols']
        instance.numerical_cols = data['numerical_cols']
        instance.feature_names = data['feature_names']
        instance.is_fitted = True
        return instance


def prepare_and_save_data():
    """Runs data preprocessing on dataset and stores train/test splits."""
    data_path = "data/social_media_trends.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run dataset_generator.py first.")
        
    df = pd.read_csv(data_path)
    preprocessor = TrendDataPreprocessor(max_tfidf_features=250)
    X, y, df_featured = preprocessor.fit_transform(df)
    
    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    preprocessor.save("models/preprocessor_pipeline.joblib")
    
    # Save processed splits for rapid training
    np.savez_compressed(
        "data/processed_data_splits.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    
    # Save featured dataframe for EDA inspection
    df_featured.to_csv("data/processed_features.csv", index=False)
    print(f"Data preprocessed successfully. Feature matrix shape: {X.shape}")
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


if __name__ == '__main__':
    prepare_and_save_data()
