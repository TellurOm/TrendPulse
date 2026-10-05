# ACADEMIC MAJOR PROJECT REPORT
## Case Study No. 127: Social Media Trend Analysis

---

# **TrendPulse: Multi-Modal Social Media Trend Dynamics, Virality Prediction, and Conversational Topic Mining Framework**

**Course:** B.Tech / BE in Artificial Intelligence and Machine Learning  
**Student Author:** Tellur Om  
**Domain:** Natural Language Processing, Applied Machine Learning, Computational Social Science  
**Status:** Completed & Evaluated  

---

## **Abstract**
In the era of hyper-connected digital communication, public social networks generate continuous streams of unstructured opinion, multimedia content, and commercial discourse. Organizations seeking actionable intelligence from publicly available social media data face significant challenges: high noise-to-signal ratios, non-stationary topical drift, extreme power-law distributions in audience engagement, and multi-modal interactions. 

This major project presents **TrendPulse**, an end-to-end machine learning framework designed to systematically analyze, cluster, and forecast social media trend dynamics. Using a curated multi-platform benchmark dataset of 5,000 public posts across four major social ecosystems (`Twitter/X`, `Instagram`, `LinkedIn`, `Reddit`) and six distinct thematic domains, we engineer a 278-dimensional composite feature space combining natural language semantics (TF-IDF $n$-grams), lexical sentiment polarity and subjectivity, non-linear author reach scaling, and continuous cyclical temporal projections. 

We benchmark five supervised learning algorithms—Logistic Regression, Support Vector Machines (RBF Kernel), Random Forest, Gradient Boosting, and Multi-Layer Perceptron (MLP) neural networks—under 5-fold stratified cross-validation. Regularized Logistic Regression achieved top overall performance with **57.10% test accuracy**, **50.02% macro F1-score**, and **72.82% multi-class ROC-AUC (OvR)**, proving that normalized linear decision boundaries effectively navigate sparse high-dimensional NLP spaces. Furthermore, an unsupervised Latent Dirichlet Allocation (LDA) model successfully extracted six latent conversational clusters, enabling thematic discovery without ground-truth supervision. The complete framework is packaged into a high-performance interactive Streamlit dashboard and a fully executed Jupyter Notebook (`.ipynb`), delivering real-time virality predictions, feature diagnostics, and automated content optimization heuristics.

---

## **1. Introduction & Problem Definition**

### **1.1 Assigned Case Study**
- **Case Study Number:** 127
- **Problem Statement:** *"An organization wants to identify meaningful patterns in publicly available social-media-related data (With Proper Justification)."*
- **Student-Formulated Project Title:** **TrendPulse: Multi-Modal Social Media Trend Dynamics, Virality Prediction, and Conversational Topic Mining Framework**

### **1.2 Mathematical Problem Formulation**
Let each publicly observed social media post $i$ be parameterized by a multi-modal feature vector:
$$\mathbf{x}_i = \left[ \mathbf{x}_{text}^{(i)}, \mathbf{x}_{sentiment}^{(i)}, \mathbf{x}_{temporal}^{(i)}, \mathbf{x}_{author}^{(i)}, \mathbf{x}_{modality}^{(i)} \right] \in \mathbb{R}^{D}$$

Where:
- $\mathbf{x}_{text} \in \mathbb{R}^{250}$: Sparse TF-IDF vector representing unigrams and bigrams with sublinear term-frequency scaling.
- $\mathbf{x}_{sentiment} \in [-1, 1] \times [0, 1]$: Emotional polarity and subjective intensity.
- $\mathbf{x}_{temporal} = \left[\sin\left(\frac{2\pi h}{24}\right), \cos\left(\frac{2\pi h}{24}\right), \sin\left(\frac{2\pi d}{7}\right), \cos\left(\frac{2\pi d}{7}\right)\right]$: Continuous trigonometric projections of hour $h \in [0, 23]$ and day $d \in [0, 6]$.
- $\mathbf{x}_{author} = [\log(1 + \text{followers}), \text{verified status}]$: Logarithmic dampening of follower count power-law distributions.
- $\mathbf{x}_{modality} \in \{0, 1\}^4$: One-hot encoded format indicator (`Text`, `Image`, `Video`, `Carousel`).

The learning objective is partitioned into two distinct tasks:
1. **Supervised Virality Classification:**
   $$f: \mathcal{X} \rightarrow \mathcal{Y}, \quad \mathcal{Y} \in \{\text{Low}, \text{Moderate}, \text{Viral}\}$$
   Where the target classes represent engagement velocity quantiles:
   - $\text{Low Reach}$: Engagement in the 0th–50th percentile ($P \le Q_{0.50}$).
   - $\text{Moderate Traction}$: Engagement in the 50th–85th percentile ($Q_{0.50} < P < Q_{0.85}$).
   - $\text{Viral Breakout}$: Engagement in the top 15th percentile ($P \ge Q_{0.85}$).
2. **Unsupervised Conversational Topic Modeling:**
   Extracting $K=6$ latent topics using Latent Dirichlet Allocation (LDA), modeling each document as a Dirichlet mixture over topics and each topic as a Dirichlet mixture over vocabulary tokens:
   $$P(\mathbf{w}|\alpha, \beta) = \prod_{n=1}^{N} \sum_{z_n=1}^{K} P(w_n|z_n, \beta) P(z_n|\theta_d)$$

### **1.3 Enterprise & Organizational Justification**
Organizations require this predictive and analytical framework for several critical business functions:
1. **Public Relations (PR) & Crisis Mitigation:** Detecting emerging polarizing or negative discussions before viral algorithmic amplification causes brand damage.
2. **Publishing Schedule Optimization:** Identifying the exact diurnal windows (e.g., 12:00-14:00 and 18:00-21:00) where engagement velocity per follower is maximized.
3. **Resource Allocation in Content Creation:** Determining whether high-budget video or carousel production yields a statistically verified engagement lift over static text.
4. **Market & Competitor Intelligence:** Systematically mining consumer discussions across technology, finance, health, and entertainment without manual tagging.

---

## **2. Dataset Description & Data Quality Observations**

### **2.1 Dataset Overview**
- **Records:** 5,000 unique social media posts.
- **Platforms:** `Twitter/X` (25%), `Instagram` (25%), `LinkedIn` (25%), `Reddit` (25%).
- **Verticals:** Technology & AI, Finance & Crypto, Health & Wellness, Entertainment & Culture, Sports & Gaming, Business & Careers.
- **Target Variable:** `virality_tier` (Low: 49.96%, Moderate: 35.02%, Viral: 15.02%).

### **2.2 Data Quality Issues & Preprocessing Resolutions**
1. **Extreme Heavy-Tail Skewness:**
   - *Observation:* Raw audience engagement (likes, shares, comments) exhibits an extreme power-law distribution where $<15\%$ of posts account for $>70\%$ of cumulative engagement.
   - *Resolution:* Applied $\log(1 + x)$ transforms to continuous engagement and follower variables prior to scaling.
2. **Textual Noise & Non-Standard Syntax:**
   - *Observation:* Public social texts contain unstructured URLs, user handles (`@user`), emojis, capitalization emphasis, and unstandardized hashtag delimiters.
   - *Resolution:* Engineered a custom regex pipeline (`src/text_processor.py`) that normalizes casing, cleans URLs, strips mentions, extracts hashtags as discrete tokens, and filters English stopwords while preserving emotional punctuation count.
3. **Temporal Boundary Discontinuity:**
   - *Observation:* Hour 23:00 and Hour 00:00 are separated by 23 integer units in naive encoding despite being contiguous in real time.
   - *Resolution:* Engineered sine and cosine cyclical projections mapping 24-hour and 7-day cycles to unit circles.

---

## **3. Exploratory Data Analysis (EDA) Findings**

Key statistical observations from the exploratory data analysis include:
1. **Format-Driven Reach Multiplier:**
   Posts featuring `Video` and `Carousel` media formats achieve a **$2.1\times$ to $2.4\times$ higher median engagement** than plain text across all four social platforms.
2. **Diurnal Bimodal Engagement Peaks:**
   Engagement velocity displays pronounced surges during **12:00–14:00 (Lunchtime peak)** and **18:00–21:00 (Evening leisure peak)**, whereas posts published between 01:00 and 06:00 underperform by $>60\%$.
3. **The Sentiment Polarization Effect:**
   Neutral posts exhibit the lowest probability of viral breakout ($11.2\%$), whereas posts with high positive excitement ($>0.40$) or polarizing critique ($<-0.30$) combined with high subjectivity exceed a **$34.5\%$ viral breakout rate**.
4. **Conversational Dialogue Hooks:**
   Posts incorporating at least one conversational question mark (`?`) generate $2.3\times$ higher reply volumes, which social ranking algorithms weight twice as heavily as passive likes.

---

## **4. Machine Learning Methodology & Model Benchmarking**

### **4.1 Feature Space Composition (278 Features)**
- **14 Engineered Numerical Features:** $\log(\text{followers})$, verified flag, hashtag count, character count, word count, polarity, subjectivity, question count, exclamation count, URL flag, $\text{hour\_sin}$, $\text{hour\_cos}$, $\text{day\_sin}$, $\text{day\_cos}$.
- **14 Categorical One-Hot Features:** Platform, topic vertical, media format.
- **250 NLP Features:** TF-IDF unigram and bigram document frequencies with sublinear scaling.

### **4.2 Supervised Model Comparison (Held-Out Test Set)**

| Algorithm | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1 | Multi-Class ROC-AUC (OvR) | 5-Fold CV F1 (μ ± σ) | Training Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **57.10%** | **52.12%** | **49.29%** | **50.02%** | **55.72%** | **72.82%** | **50.10% ± 1.34%** | **1.83s** |
| **Gradient Boosting** | 54.80% | 49.69% | 47.28% | 48.00% | 54.07% | 72.27% | 49.79% ± 1.23% | 15.26s |
| **Support Vector Machine (SVM)** | 54.30% | 49.08% | 45.87% | 46.57% | 52.96% | 71.70% | 47.39% ± 1.05% | 16.09s |
| **Random Forest** | 56.20% | 53.94% | 45.78% | 46.42% | 53.74% | 71.95% | 44.53% ± 1.05% | 0.89s |
| **Multi-Layer Perceptron (MLP)** | 54.70% | 48.14% | 43.92% | 43.72% | 51.16% | 71.69% | 46.66% ± 1.66% | 0.67s |

### **4.3 Interpretation & Feature Importance Analysis**
- **Why Linear Models Performed Best:** In text classification where the feature space is sparse, high-dimensional, and noisy ($D=278$), $L_2$-regularized Logistic Regression creates a smooth, convex hyper-plane that avoids overfitting, whereas tree-based ensembles and neural networks split variance across sparse n-gram dimensions.
- **Primary Virality Drivers:**
  1. `sentiment_subjectivity` (0.135 relative weight): Content expressing strong perspectives out-engages purely factual statements.
  2. `log_author_followers` (0.115 relative weight): Initial follower reach serves as the foundational algorithmic springboard.
  3. `media_type_Video` (0.066 relative weight): Platform algorithms heavily favor video dwell time.
  4. Temporal components `hour_sin` and `hour_cos` (0.097 combined weight): Aligning with peak user activity windows.

### **4.4 Error Analysis & Failure Modes**
1. **Ambiguous Decision Boundaries (Moderate vs. Viral):**  
   The majority of misclassifications occurred between *Moderate* and *Viral*. Because virality in real social algorithms operates on continuous power laws, boundary cases near the 85th percentile threshold are sensitive to small stochastic fluctuations.
2. **High-Follower Underperformance (False Positives):**  
   The model occasionally over-predicted virality on accounts with $>500,000$ followers when the copy was bland, highlighting the perpetual tension between audience scale and content quality.
3. **Niche Account Breakouts (False Negatives):**  
   Accounts with $<1,000$ followers producing explosive viral hits were occasionally misclassified as *Low* because metadata signals could not foresee external retweet syndication.

---

## **5. Unsupervised Topic Discovery (LDA)**
Using Latent Dirichlet Allocation with $K=6$, the system discovered the following thematic clusters without human supervision:
- **Topic 1 (Autonomous AI & Compute):** `completely`, `open`, `championship`, `benchmark`, `weights`, `model`, `futuretech`.
- **Topic 2 (Cinema & Visual Media):** `day`, `office`, `record`, `looks`, `stunning`, `effects`, `practical`.
- **Topic 3 (Macro Crypto & Markets):** `bitcoin`, `investing`, `update`, `hyper`, `bull`, `economy`, `stockmarket`.
- **Topic 4 (Executive Strategy & Careers):** `leadership`, `high`, `performance`, `futureofwork`, `businessstrategy`, `productivity`.
- **Topic 5 (Global Esports & Sports):** `records`, `breaks`, `million`, `viewers`, `live`, `world`, `championship`.
- **Topic 6 (Viral Culture & Tech Portfolios):** `original`, `major`, `viral`, `truth`, `portfolio`, `problem`, `beat`.

---

## **6. Streamlit Interactive Application Architecture**
The complete framework is deployed as a modular, responsive Streamlit dashboard (`app.py`) featuring:
1. **Executive Overview:** Problem definition, mathematical formulation, and system architecture.
2. **Exploratory Data Analysis:** Interactive filterable cross-tabulations, heatmaps, and sentiment distributions.
3. **Benchmark Arena:** Interactive model comparisons, confusion matrix visualizers, and feature ranking charts.
4. **Real-Time Virality Predictor:** Live testing console with probability gauges, NLP attribute extraction, and automated AI optimization heuristics.
5. **Topic Explorer:** Visualizing latent LDA topic distributions and keyword weights.
6. **Academic Documentation & Viva Guide:** Built-in report viewer and viva defense preparation materials.

---

## **7. Conclusions & Future Roadmap**
1. **Academic Findings:** Social media virality can be effectively framed as a supervised multi-class problem combined with unsupervised topic discovery. Contextual modality, emotional subjectivity, and publication timing are stronger predictors of virality than raw character count.
2. **Limitations:** The model operates strictly on publicly accessible post metadata; private messaging shares and full community follower graphs are unavailable in standard public API streams.
3. **Future Enhancements:** Integrating contextual transformer embeddings (e.g. RoBERTa, DeBERTa) and Graph Neural Networks (GNNs) to model diffusion cascades across explicit social network graphs.
