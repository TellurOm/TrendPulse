# MAJOR PROJECT DEFENSE PRESENTATION OUTLINE
## Case Study No. 127: Social Media Trend Analysis
### **Project Title: TrendPulse — Multi-Modal Social Media Trend Dynamics, Virality Prediction, and Conversational Topic Mining Framework**
**Presenter:** Tellur Om  
**Degree:** B.Tech / BE in Artificial Intelligence & Machine Learning  

---

### **Slide 1: Title & Introduction**
- **Slide Content:**
  - Project Title: *TrendPulse: Multi-Modal Social Media Trend Dynamics & Virality Intelligence Framework*
  - Case Study No. 127: Social Media Trend Analysis
  - Candidate: Tellur Om
  - Supervisor / Evaluator: Department of AIML
- **Speaker Talking Points:**
  - *"Respected evaluators, today I present our Major Project on Case Study 127: Social Media Trend Analysis. We address the fundamental challenge modern organizations face: how to extract meaningful, predictive patterns from massive volumes of public social media streams."*

---

### **Slide 2: Problem Statement & Motivation**
- **Slide Content:**
  - **Core Challenge:** Unstructured, volatile public discourse with extreme noise and power-law reach.
  - **Key Questions:**
    1. What content features (NLP, sentiment, format) differentiate viral breakout trends from low-traction posts?
    2. When are the optimal diurnal publishing windows to maximize engagement velocity?
    3. How can we automatically discover latent conversational clusters without manual annotation?
- **Speaker Talking Points:**
  - *"Organizations waste millions on content that goes unnoticed. Traditional keyword matching cannot predict reach. We need a dual approach: supervised virality forecasting and unsupervised conversational topic modeling."*

---

### **Slide 3: Formal ML Problem Formulation**
- **Slide Content:**
  - Multi-Modal Input Space: $\mathbf{x} = [\mathbf{x}_{text}, \mathbf{x}_{sentiment}, \mathbf{x}_{temporal}, \mathbf{x}_{author}, \mathbf{x}_{modality}] \in \mathbb{R}^{278}$
  - **Task 1: Supervised Virality Classification:**
    - Target: $Y \in \{\text{Low Reach (50%)}, \text{Moderate Traction (35%)}, \text{Viral Breakout (15%)}\}$
  - **Task 2: Unsupervised Topic Discovery:**
    - Latent Dirichlet Allocation (LDA) with $K=6$ conversational mixtures.
- **Speaker Talking Points:**
  - *"We formulate virality not as noisy regression, but as a robust 3-tier classification problem based on engagement velocity quantiles, supplemented by LDA topic clustering."*

---

### **Slide 4: Dataset & Quality Management**
- **Slide Content:**
  - 5,000 multi-platform public posts across `Twitter/X`, `Instagram`, `LinkedIn`, and `Reddit`.
  - 6 Diverse Verticals: Tech & AI, Finance/Crypto, Health, Pop Culture, Sports, Business.
  - **Quality Handling:**
    - Logarithmic transformation for Pareto-distributed follower counts: $\log(1 + x)$.
    - Regex cleaning for URLs, mentions, emojis, and punctuation.
    - Cyclical sine/cosine transformation for 24-hour and 7-day cyclical continuity.
- **Speaker Talking Points:**
  - *"Social media data is notoriously skewed. A naive linear model would be misled by accounts with millions of followers. We apply log scaling and continuous cyclical temporal projections to ensure mathematical validity."*

---

### **Slide 5: Exploratory Data Analysis — Key Insights**
- **Slide Content:**
  - **Observation 1:** Video and Carousel formats achieve $2.1\times$ to $2.4\times$ higher engagement velocity than static plain text.
  - **Observation 2:** Diurnal peaks occur at 12:00-14:00 (lunchtime) and 18:00-21:00 (evening leisure).
  - **Observation 3:** High emotional subjectivity combined with polarizing sentiment increases viral probability by $3.2\times$.
- **Speaker Talking Points:**
  - *"Our EDA revealed that algorithmic distribution heavily favors dwell time via video formats and conversational hooks like question marks."*

---

### **Slide 6: Feature Engineering Architecture**
- **Slide Content:**
  - High-Dimensional 278-D Feature Matrix:
    - 250 TF-IDF N-Gram features (unigrams & bigrams).
    - 14 Scaled numerical features (followers, sentiment polarity/subjectivity, text metrics, temporal sine/cosine).
    - 14 One-hot encoded categorical features (platform, domain, media type).
- **Speaker Talking Points:**
  - *"By fusing stylistic NLP signals, semantic TF-IDF vectors, temporal physics, and author credentials into a unified feature matrix, the model captures both what is being said and how/when it is being delivered."*

---

### **Slide 7: Machine Learning Model Comparison**
- **Slide Content:**
  - 5 Diverse Paradigms Evaluated under 5-Fold Stratified Cross-Validation:
    1. Logistic Regression (L2 Regularized)
    2. Support Vector Machine (RBF Kernel)
    3. Random Forest (150 Trees)
    4. Gradient Boosting (120 Estimators)
    5. Multi-Layer Perceptron (Feedforward Neural Net)
- **Speaker Talking Points:**
  - *"We rigorously benchmarked linear, kernel-based, bagging, boosting, and neural architectures to understand the performance landscape."*

---

### **Slide 8: Empirical Benchmark Results**
- **Slide Content:**
  - **Logistic Regression:** Accuracy 57.10% | Macro F1 50.02% | ROC-AUC 72.82% | CV F1 50.10% ± 1.34%
  - **Gradient Boosting:** Accuracy 54.80% | Macro F1 48.00% | ROC-AUC 72.27% | CV F1 49.79% ± 1.23%
  - **Random Forest:** Accuracy 56.20% | Macro F1 46.42% | ROC-AUC 71.95% | CV F1 44.53% ± 1.05%
  - **SVM:** Accuracy 54.30% | Macro F1 46.57% | ROC-AUC 71.70%
  - **MLP Neural Net:** Accuracy 54.70% | Macro F1 43.72% | ROC-AUC 71.69%
- **Speaker Talking Points:**
  - *"Regularized Logistic Regression achieved the highest balanced score across Macro F1 and ROC-AUC. In high-dimensional, sparse text representations, linear regularized hyperplanes avoid the overfitting that plagues tree ensembles and complex neural networks."*

---

### **Slide 9: Feature Importance & Virality Drivers**
- **Slide Content:**
  - Top Virality Drivers (Gini & Coefficient Analysis):
    1. `sentiment_subjectivity` (Emotional perspective)
    2. `log_author_followers` (Baseline account audience)
    3. `media_type_Video` (Format engagement bonus)
    4. `hour_sin` / `hour_cos` (Publication scheduling)
    5. High-impact keywords: *benchmark*, *breakthrough*, *investing*, *record*.
- **Speaker Talking Points:**
  - *"Our feature importance diagnostics prove that virality is not pure luck: subjective conviction, video format, and prime-time scheduling provide measurable, actionable lifts."*

---

### **Slide 10: Error Analysis & Model Diagnostics**
- **Slide Content:**
  - Confusion matrix analysis across classes:
    - Low Reach: High Precision (66.0%) and Recall (77.4%).
    - Moderate vs. Viral Boundary: Most errors occur between Moderate and Viral due to the continuous nature of algorithmic distribution.
    - High-follower flops (False Positives) vs. Niche account breakouts (False Negatives).
- **Speaker Talking Points:**
  - *"Our error analysis demonstrates that boundary ambiguity between moderate and viral posts reflects real-world power laws rather than model failure."*

---

### **Slide 11: Unsupervised Topic Discovery (LDA)**
- **Slide Content:**
  - 6 Extracted Conversational Clusters without manual labels:
    - Topic 1: Autonomous AI & Open-Source LLMs
    - Topic 2: Cinema, Box Office & Visual Media
    - Topic 3: Macro Crypto, Bitcoin & Market Volatility
    - Topic 4: Executive Leadership & Future of Work
    - Topic 5: Global Esports & Athletics Championships
    - Topic 6: Tech Career Advice & Portfolio Building
- **Speaker Talking Points:**
  - *"LDA demonstrates how organizations can autonomously track topical themes and narrative shifts across thousands of posts in real time."*

---

### **Slide 12: Streamlit Application Demonstration**
- **Slide Content:**
  - Interactive Web Dashboard (`app.py`):
    - Executive Overview & Problem Definition
    - Interactive Exploratory Trend Analytics
    - Model Benchmark Arena & Confusion Matrices
    - Real-Time Live Virality Predictor
    - AI Content Optimization & Virality Booster Heuristics
    - Topic Explorer & Academic Documentation Viewer
- **Speaker Talking Points:**
  - *"To make our machine learning model usable by non-technical marketing and PR teams, we built a responsive, dark-mode Streamlit dashboard with real-time inference and prescriptive content recommendations."*

---

### **Slide 13: Jupyter Notebook Execution**
- **Slide Content:**
  - Fully documented and executed `.ipynb` file (`Social_Media_Trend_Analysis_Major_Project.ipynb`).
  - 28 structured cells containing end-to-end data loading, statistical visualizations, preprocessing, model cross-validation, and error analyses.
- **Speaker Talking Points:**
  - *"All code, exploratory plots, cross-validation runs, and evaluation metrics are reproducible cell-by-cell in our Jupyter Notebook."*

---

### **Slide 14: System Limitations & Real-World Constraints**
- **Slide Content:**
  - Limitations:
    1. Dark Social Sharing: Private encrypted messaging cannot be tracked via public social APIs.
    2. Lack of Explicit Network Graph: Community follower network graphs require elevated API access.
    3. Dynamic Platform Algorithm Changes: Social recommendation weights shift over time.
- **Speaker Talking Points:**
  - *"We critically recognize the boundaries of our study: dark social sharing and evolving platform algorithms represent ongoing real-world challenges."*

---

### **Slide 15: Conclusion & Future Scope**
- **Slide Content:**
  - **Summary:** Successfully formulated, developed, and deployed an end-to-end social media trend analysis framework meeting all deliverables for Case Study 127.
  - **Future Roadmap:**
    - Fine-tuning transformer models (RoBERTa / ModernBERT) on domain-specific social corpora.
    - Graph Neural Networks (GNNs) to model diffusion cascades across follower graphs.
- **Speaker Talking Points:**
  - *"Thank you for your time. I am now ready for the viva voce examination and questions."*
