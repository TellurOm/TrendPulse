# Dataset Specification & Data Dictionary
## AIML Major Project | Case Study 127: Social Media Trend Analysis

---

### 1. Dataset Overview & Justification
- **Assigned Problem Statement:** An organization wants to identify meaningful patterns in publicly available social-media-related data (With Proper Justification).
- **Project Title:** *TrendPulse: Multi-Modal Social Media Trend Dynamics, Virality Prediction, and Conversational Topic Mining Framework*
- **Domain:** Natural Language Processing (NLP), Social Media Analytics, Predictive Modeling.
- **Dataset Size:** 5,000 post records across 4 major public social platforms (`Twitter/X`, `Instagram`, `LinkedIn`, `Reddit`) and 6 topic verticals.
- **Format:** Structured CSV (`data/social_media_trends.csv`) + Processed Feature Matrix (`data/processed_features.csv`).

---

### 2. Variable Definitions & Schema

| Variable Name | Data Type | Role | Description & Measurement Scale |
| :--- | :--- | :--- | :--- |
| `post_id` | String | Identifier | Unique post record key (`POST_00001` to `POST_05000`). |
| `platform` | Categorical | Input Feature | Source public social platform: `Twitter/X`, `Instagram`, `LinkedIn`, `Reddit`. |
| `category` | Categorical | Context Feature | Topic domain vertical: `Technology & AI`, `Finance & Crypto`, `Health & Wellness`, `Entertainment & Culture`, `Sports & Gaming`, `Business & Careers`. |
| `post_text` | Text | Primary NLP Input | Full text content of the public post including natural language copy, mentions, and hashtags. |
| `timestamp` | Datetime | Temporal Feature | Publication timestamp in ISO format (`YYYY-MM-DD HH:MM:SS`). |
| `hour_of_day` | Integer (0-23) | Temporal Feature | Hour of publication in 24-hour clock. |
| `day_of_week` | Integer (0-6) | Temporal Feature | Day of publication (0 = Monday, 6 = Sunday). |
| `author_followers` | Integer | Author Reach | Account follower count. Follows Pareto/Log-normal distribution ($\mu \approx 4,000$, range $50$ to $5,000,000+$). |
| `author_verified` | Binary (0/1) | Credibility Feature | Binary indicator if author profile holds platform verification. |
| `media_type` | Categorical | Content Modality | Format attached to post: `Text`, `Image`, `Video`, `Carousel`. |
| `hashtag_count` | Integer | Stylistic Feature | Total count of thematic hashtags in the post body. |
| `hashtags` | Text | Keyword Tokens | Space-separated list of hashtag keywords. |
| `char_count` | Integer | Syntactic Feature | Total character length of post text. |
| `word_count` | Integer | Syntactic Feature | Total word count. |
| `sentiment_polarity` | Float (-1.0 to 1.0) | Sentiment Feature | Lexicon-derived emotional valence: negative (< -0.15), neutral (-0.15 to 0.15), positive (> 0.15). |
| `sentiment_subjectivity` | Float (0.0 to 1.0) | Sentiment Feature | Degree of opinion, subjectivity, or emotional assertion vs. objective facts. |
| `question_count` | Integer | Conversational Feature | Count of question marks ('?') provoking audience dialogue. |
| `exclamation_count` | Integer | Conversational Feature | Count of exclamation marks ('!') conveying urgency or enthusiasm. |
| `has_url` | Binary (0/1) | Link Feature | Presence of external hyperlink. |
| `likes` | Integer | Raw Metric | Observed like reaction count. |
| `shares_retweets` | Integer | Raw Metric | Observed share / repost / retweet count. |
| `comments` | Integer | Raw Metric | Observed discussion reply count. |
| `raw_engagement` | Float | Derived Metric | Composite Engagement Index: $\text{Likes} + (3.0 \times \text{Shares}) + (2.0 \times \text{Comments})$. |
| `engagement_rate` | Float | Derived Metric | Engagement velocity normalized per 1,000 followers. |
| `virality_tier` | Categorical | **Target Label (Y)** | Ground truth engagement class: `Low` (bottom 50%), `Moderate` (50%-85%), `Viral` (top 15% breakout). |

---

### 3. Data Quality Observations & Handling

1. **Extreme Power-Law Skewness:**
   - *Observation:* Raw likes and shares exhibit heavy-tail Pareto distributions where 10-15% of posts account for >70% of total engagement.
   - *Mitigation:* We apply log-transformations ($\log(1 + x)$) to follower counts and engagement metrics during normalization.

2. **Text Noise and Formatting Inconsistencies:**
   - *Observation:* Raw social posts contain URLs, user handles (`@`), emojis, varied punctuation, and irregular casing.
   - *Mitigation:* Built a robust regex text cleaning pipeline in `src/text_processor.py` that strips non-informative tokens, extracts clean tokens, preserves hashtag terms, and removes stop words.

3. **Cyclical Nature of Temporal Features:**
   - *Observation:* Hour 23 and Hour 0 are numerically distant (23 vs 0) but chronologically contiguous.
   - *Mitigation:* Engineered trigonometric cyclical encodings:
     $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
     $$\text{day\_sin} = \sin\left(\frac{2\pi \cdot \text{day}}{7}\right), \quad \text{day\_cos} = \cos\left(\frac{2\pi \cdot \text{day}}{7}\right)$$

4. **Class Imbalance in Virality:**
   - *Observation:* By definition, true viral content represents an elite minority (~15%) of all public social posts.
   - *Mitigation:* Stratified 80/20 train/test splitting was strictly enforced. Class evaluation utilizes Macro F1 and ROC-AUC (OvR) to prevent false inflation from the majority class.
