# VIVA VOCE EXAMINATION QUESTIONS & ANSWERS
## AIML Major Project: Case Study No. 127
### **TrendPulse: Multi-Modal Social Media Trend Dynamics & Virality Intelligence Framework**
**Candidate:** Tellur Om  

---

### **Section A: Problem Definition & Mathematical Modeling**

#### **Q1: What is the formal problem statement of Case Study 127 and what is your student-formulated project title?**
- **Answer:**  
  The assigned problem statement from Case Study 127 is: *"An organization wants to identify meaningful patterns in publicly available social-media-related data (With Proper Justification)."*  
  Our student-formulated title is: **"TrendPulse: Multi-Modal Social Media Trend Dynamics, Virality Prediction, and Conversational Topic Mining Framework"**. We framed the problem as a hybrid supervised classification (virality tier prediction) and unsupervised topic discovery (Latent Dirichlet Allocation) task.

#### **Q2: Why did you frame virality prediction as a 3-class classification problem (Low, Moderate, Viral) rather than continuous regression?**
- **Answer:**  
  1. *Power-Law Distribution of Social Metrics:* Social media engagement metrics (likes, shares, comments) follow an extreme Pareto/Power-law distribution with a very heavy tail ($<15\%$ of posts generate $>70\%$ of engagement). In regression, models attempt to minimize squared error ($\text{MSE}$), which causes the loss function to be dominated by extreme viral outliers, destabilizing gradient updates.
  2. *Actionable Business Triage:* From an organizational perspective, decision-makers do not need to know whether a post will get 43,210 vs 45,600 likes; they need actionable categorical triage: will this post fail to gain traction (*Low*), achieve expected organic reach (*Moderate*), or explode into a breakout viral trend (*Viral*) that warrants brand monitoring or PR triage.
  3. *Zero-Shot Predictability:* Zero-shot post regression has massive variance, whereas quantiles create stable, mathematically well-conditioned decision boundaries.

#### **Q3: What was your formula for the composite engagement score?**
- **Answer:**  
  The composite engagement velocity is computed as:
  $$\text{Engagement Score} = \text{Likes} + (3.0 \times \text{Shares}) + (2.0 \times \text{Comments})$$
  We weight shares highest ($3.0\times$) because sharing content into one's personal feed represents the strongest social endorsement and primary viral diffusion vector. Comments are weighted $2.0\times$ because replying requires high cognitive engagement, which ranking algorithms prioritize over passive one-click likes ($1.0\times$).

---

### **Section B: Feature Engineering & Preprocessing**

#### **Q4: Why did you use cyclical sine and cosine encoding for temporal features?**
- **Answer:**  
  Standard linear integer encoding (e.g. Hour = $0, 1, \dots, 23$) creates an artificial numerical discontinuity: Hour 23 and Hour 0 are separated by a difference of 23 units, even though in physical time they are only 1 hour apart. By projecting time onto a unit circle using trigonometric transformations:
  $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$
  $$\text{day\_sin} = \sin\left(\frac{2\pi \cdot d}{7}\right), \quad \text{day\_cos} = \cos\left(\frac{2\pi \cdot d}{7}\right)$$
  The distance between Hour 23 and Hour 0 becomes mathematically small and continuous, allowing linear models and neural networks to capture cyclical diurnal patterns without artificial boundary cliffs.

#### **Q5: Why did you apply log-transformation to author follower counts?**
- **Answer:**  
  Follower counts span multiple orders of magnitude (from 50 followers to over 5,000,000). A standard linear scaler would compress 99% of normal accounts into a tiny band near 0, with a few mega-influencer accounts dominating the gradients. By applying $\log(1 + x)$, we transform the multiplicative power-law scale into an additive linear scale that aligns with the diminishing marginal returns of algorithmic distribution.

#### **Q6: How does TF-IDF vectorization work mathematically in your pipeline?**
- **Answer:**  
  Term Frequency-Inverse Document Frequency (TF-IDF) quantifies the importance of a word $t$ in a post $d$ relative to the entire corpus $D$:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
  Where:
  $$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}, \quad \text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
  This penalizes ubiquitous non-informative words while highlighting distinct topical buzzwords (e.g. *breakthrough*, *benchmark*, *bitcoin*, *championship*).

---

### **Section C: Model Development & Evaluation**

#### **Q7: Why did regularized Logistic Regression achieve higher test accuracy and Macro F1 than complex models like Random Forest and MLP?**
- **Answer:**  
  This is a classic manifestation of the **Curse of Dimensionality in Natural Language Processing**:
  1. *Sparsity in High Dimensions:* Our feature matrix has 278 dimensions, of which 250 are sparse TF-IDF n-grams. Linear models with $L_2$ regularization (Ridge penalty) estimate a single weight per feature and create a smooth global convex decision hyperplane, which is robust against noise.
  2. *Tree Partitioning Breakdown:* Decision tree ensembles (Random Forest) perform axis-aligned orthogonal splits. In sparse text representations where most features are zero for any given sample, individual trees split on rare words, leading to high-variance subtrees and sub-optimal partitions.
  3. *Neural Network Sample Efficiency:* Deep neural networks (MLPs) require substantial training samples to learn latent representations without memorizing sparse text features; with 4,000 training samples, the regularized linear model generalized with superior out-of-sample stability.

#### **Q8: Why is Macro F1 a more critical evaluation metric than simple Accuracy in this project?**
- **Answer:**  
  By real-world social network definition, the classes are imbalanced: *Low Reach* accounts for ~50%, *Moderate Traction* ~35%, and true *Viral Breakouts* only ~15%.  
  A trivial naive model predicting only *Low* and *Moderate* could achieve 85% accuracy while completely failing at identifying the most valuable class (Viral).  
  **Macro F1** calculates the unweighted harmonic mean of precision and recall independently for each class and averages them:
  $$\text{Macro F1} = \frac{\text{F1}_{\text{Low}} + \text{F1}_{\text{Moderate}} + \text{F1}_{\text{Viral}}}{3}$$
  This equally penalizes poor performance on the viral minority class, making it the most honest and rigorous academic benchmark.

#### **Q9: What is the difference between Macro Average, Micro Average, and Weighted Average?**
- **Answer:**  
  - **Macro Average:** Computes metric independently per class and takes simple arithmetic mean. Treats all classes equally regardless of support size. Best for detecting minority class performance.
  - **Micro Average:** Aggregates true positives, false positives, and false negatives globally across all classes before calculating metric. Equal to overall accuracy in multi-class single-label classification.
  - **Weighted Average:** Computes metric per class and weights each by its support (number of true instances). Can mask severe failures in small classes if the majority class performs well.

#### **Q10: What were the most significant features driving virality in your model?**
- **Answer:**  
  Based on Gini importance from Gradient Boosting and feature coefficients from Logistic Regression:
  1. `sentiment_subjectivity` & `sentiment_polarity`: Content with bold personal perspectives and emotional valence outperforms bland neutral statements.
  2. `log_author_followers`: Initial audience foundation provides the baseline algorithmic reach.
  3. `media_type_Video`: Video assets generate over $2.1\times$ higher dwell time, triggering platform recommendation algorithms.
  4. Cyclical temporal terms (`hour_sin`, `hour_cos`): Posts published during peak social usage windows (12:00-14:00 and 18:00-21:00) gain viral traction significantly faster.

---

### **Section D: Unsupervised Learning & Topic Discovery**

#### **Q11: What is Latent Dirichlet Allocation (LDA) and how does it discover topics?**
- **Answer:**  
  LDA is a generative probabilistic model for discrete data. It posits that:
  1. Each document is a mixture over a set of $K$ latent Dirichlet-distributed topics: $\theta_d \sim \text{Dirichlet}(\alpha)$.
  2. Each topic is a mixture over a vocabulary of words: $\phi_k \sim \text{Dirichlet}(\beta)$.
  3. For each word token in a post, a topic assignment $z_i$ is sampled from $\theta_d$, and a word $w_i$ is sampled from the corresponding topic distribution $\phi_{z_i}$.  
  Through variational inference or Gibbs sampling, LDA estimates the posterior topic distributions and top keyword weights without requiring any labeled target training data.

#### **Q12: How does unsupervised topic discovery add value when you already have supervised virality classification?**
- **Answer:**  
  Supervised virality models tell an organization *how far* a post will spread, but they do not explain *what* subjects are driving audience discussions. LDA operates without human bias to discover emerging themes (e.g. AI benchmarks, crypto market surges, workplace productivity), allowing marketing and PR teams to detect shifts in conversation before formulating content strategy.

---

### **Section E: Application & Deployment Architecture**

#### **Q13: How does the real-time inference pipeline work in your Streamlit application?**
- **Answer:**  
  When a user enters a draft post and selects metadata:
  1. `text_processor.py` extracts syntactic NLP attributes (character count, word count, questions, exclamations, sentiment polarity, and subjectivity).
  2. `preprocessor.py` transforms the raw input through the persisted `preprocessor_pipeline.joblib`: calculating cyclical temporal sine/cosine values, log-transforming follower counts, one-hot encoding categorical selections, and mapping cleaned text through the fitted 250-D TF-IDF vectorizer.
  3. The resulting 278-D vector is passed to the selected trained classifier (e.g. Logistic Regression or Gradient Boosting), which returns the predicted class label along with calibrated class probability distribution vectors.
  4. An AI content optimization rule-engine compares the post attributes against empirical virality drivers and outputs concrete suggestions (e.g. upgrading text to video, scheduling for peak hours, adding a conversational question hook).

#### **Q14: What are the main limitations of your framework and how would you address them in industry?**
- **Answer:**  
  1. *Dark Social & Encrypted Shares:* Private sharing via messaging apps (WhatsApp, Telegram, DMs) cannot be tracked via public social APIs.
  2. *Static Text Embeddings:* TF-IDF captures lexical presence but ignores syntactic word order and deep semantic context. In production, we would fine-tune transformer models such as ModernBERT or RoBERTa.
  3. *Follower Graph Dynamics:* True virality is propelled by network cascade topologies (who retweets whom). In industry, integrating Graph Neural Networks (GNNs) with follower network graphs would significantly improve outbreak forecasting.
