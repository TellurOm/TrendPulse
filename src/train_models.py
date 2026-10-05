"""
Model Training, Benchmarking, and Evaluation Module for Case Study 127.
Trains 5 diverse Supervised Classifiers + 1 Unsupervised LDA Topic Model.
Evaluates using Accuracy, Precision, Recall, Macro F1, ROC-AUC, and Confusion Matrices.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold


def train_and_evaluate_all_models():
    """Trains 5 supervised classifiers and 1 unsupervised LDA topic model."""
    print("Loading preprocessed data splits...")
    splits = np.load("data/processed_data_splits.npz")
    X_train = splits['X_train']
    X_test = splits['X_test']
    y_train = splits['y_train']
    y_test = splits['y_test']
    
    # Load preprocessor to get feature names and label encoder
    preprocessor_data = joblib.load("models/preprocessor_pipeline.joblib")
    feature_names = preprocessor_data['feature_names']
    label_encoder = preprocessor_data['label_encoder']
    target_names = list(label_encoder.classes_)
    
    # Define models dictionary
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        'Support Vector Machine (SVM)': SVC(
            C=1.5, kernel='rbf', probability=True, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=16, min_samples_split=4, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42
        ),
        'Multi-Layer Perceptron (MLP)': MLPClassifier(
            hidden_layer_sizes=(128, 64), activation='relu', max_iter=300,
            early_stopping=True, random_state=42
        )
    }
    
    results = {}
    saved_models = {}
    
    print("\n" + "="*70)
    print("STARTING SUPERVISED BENCHMARKING (5-Fold CV + Test Set Evaluation)")
    print("="*70)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n--> Training {name}...")
        t0 = time.time()
        
        # 5-fold cross validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        
        # Fit on full training set
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Inference speed test
        t_infer_start = time.time()
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        infer_time_ms = ((time.time() - t_infer_start) / len(X_test)) * 1000.0
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Multi-class ROC-AUC (One-vs-Rest)
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.0
            
        cm = confusion_matrix(y_test, y_pred).tolist()
        class_report = classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
        )
        
        # Extract Feature Importances if available
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:20]
            feature_importance = {
                feature_names[i]: float(importances[i]) for i in top_indices
            }
        elif hasattr(model, 'coef_'):
            # Average absolute coefficients across classes
            coef_mean = np.mean(np.abs(model.coef_), axis=0)
            top_indices = np.argsort(coef_mean)[::-1][:20]
            feature_importance = {
                feature_names[i]: float(coef_mean[i]) for i in top_indices
            }
            
        results[name] = {
            'accuracy': round(float(acc), 4),
            'precision_macro': round(float(prec_macro), 4),
            'recall_macro': round(float(rec_macro), 4),
            'f1_macro': round(float(f1_macro), 4),
            'f1_weighted': round(float(f1_weighted), 4),
            'roc_auc_macro': round(float(auc), 4),
            'cv_f1_mean': round(float(np.mean(cv_scores)), 4),
            'cv_f1_std': round(float(np.std(cv_scores)), 4),
            'training_time_sec': round(train_time, 3),
            'inference_latency_ms': round(infer_time_ms, 3),
            'confusion_matrix': cm,
            'classification_report': class_report,
            'top_feature_importance': feature_importance
        }
        
        print(f"    Test Accuracy : {acc:.4f}")
        print(f"    Macro F1-Score: {f1_macro:.4f}")
        print(f"    ROC-AUC (OvR) : {auc:.4f}")
        print(f"    CV F1 (5-Fold): {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        # Save model artifact
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_filename = f"models/{slug}.joblib"
        joblib.dump(model, model_filename)
        saved_models[name] = model_filename

    # Unsupervised Model: Latent Dirichlet Allocation (LDA) for Topic Discovery
    print("\n" + "="*70)
    print("TRAINING UNSUPERVISED TOPIC DISCOVERY MODEL (LDA, K=6 Topics)")
    print("="*70)
    
    df_raw = pd.read_csv("data/social_media_trends.csv")
    cleaned_texts = df_raw['post_text'].apply(lambda t: " ".join([w for w in t.lower().split() if len(w) > 3]))
    
    from sklearn.feature_extraction.text import CountVectorizer
    lda_vectorizer = CountVectorizer(max_features=400, stop_words='english', min_df=3)
    dtm = lda_vectorizer.fit_transform(cleaned_texts)
    
    num_topics = 6
    lda_model = LatentDirichletAllocation(
        n_components=num_topics, max_iter=25, learning_method='online', random_state=42
    )
    lda_model.fit(dtm)
    
    feature_words = lda_vectorizer.get_feature_names_out()
    lda_topics = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[:-11:-1]
        top_words = [feature_words[i] for i in top_words_idx]
        lda_topics[f"Topic #{topic_idx + 1}"] = {
            'keywords': top_words,
            'top_weights': [round(float(topic[i]), 2) for i in top_words_idx]
        }
        print(f"  Topic #{topic_idx + 1}: {', '.join(top_words[:7])}")
        
    joblib.dump(lda_model, "models/lda_topic_model.joblib")
    joblib.dump(lda_vectorizer, "models/lda_vectorizer.joblib")
    
    # Save benchmark metrics and topic summaries to JSON
    summary_output = {
        'supervised_benchmark': results,
        'unsupervised_topics': lda_topics,
        'target_classes': target_names,
        'num_train_samples': int(X_train.shape[0]),
        'num_test_samples': int(X_test.shape[0]),
        'num_features': int(X_train.shape[1]),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("models/model_metrics.json", "w") as f:
        json.dump(summary_output, f, indent=2)
        
    print("\n" + "="*70)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY IN models/")
    print("Evaluation metrics recorded in models/model_metrics.json")
    print("="*70)
    return summary_output


if __name__ == '__main__':
    train_and_evaluate_all_models()
