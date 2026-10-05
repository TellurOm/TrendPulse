"""
Dataset Generator for Case Study 127: Social Media Trend Analysis.
Generates an academically rigorous, multi-domain benchmark dataset of 5,000+ posts
with realistic power-law engagement distributions, temporal dynamics, and NLP attributes.
"""

import os
import random
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict

from text_processor import compute_sentiment, extract_hashtags, extract_nlp_features

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

PLATFORMS = ['Twitter/X', 'Instagram', 'LinkedIn', 'Reddit']
CATEGORIES = [
    'Technology & AI',
    'Finance & Crypto',
    'Health & Wellness',
    'Entertainment & Culture',
    'Sports & Gaming',
    'Business & Careers'
]

MEDIA_TYPES = ['Text', 'Image', 'Video', 'Carousel']

# Domain-specific post templates and buzzwords
TOPIC_CORPUS = {
    'Technology & AI': {
        'templates': [
            "Just deployed an open-source autonomous AI agent pipeline! The benchmark performance surpassed GPT-4o by 14%. Complete architecture diagram and weights in thread. {hashtag}",
            "Why is nobody talking about the massive breakthrough in quantum computing neuromorphic chips announced today? {hashtag}",
            "Is Python still the king of machine learning in 2026, or is Mojo and Rust completely taking over? Thoughts? {hashtag}",
            "Unbelievable milestone: our deep learning model reached 99.2% accuracy on medical diagnostics with zero false negatives! {hashtag}",
            "Major security alert: A critical zero-day exploit found in popular web framework. Update your production servers immediately! {hashtag}",
            "The future of AI agents isn't larger LLMs, it's efficient multi-modal reasoning at the edge. Here is our 5-step breakdown: {hashtag}",
            "Devs, what is your biggest pet peeve with modern microservice architectures? For me, it's distributed tracing nightmare. {hashtag}",
            "Silicon Valley is sleeping on this new AI open-weights model that runs completely on-device without cloud latency. {hashtag}"
        ],
        'hashtags': ['#ArtificialIntelligence', '#MachineLearning', '#TechTrends', '#DeepLearning', '#DataScience', '#OpenSource', '#CloudComputing', '#FutureTech']
    },
    'Finance & Crypto': {
        'templates': [
            "Bitcoin breaks past resistance levels with massive institutional liquidity inflows! Bull run continuation confirmed or bull trap? {hashtag}",
            "The Federal Reserve interest rate decision will shock the markets tomorrow. Here is our macro hedge fund analysis. {hashtag}",
            "DeFi protocol hacked for $45M due to reentrancy vulnerability. Audit reports clearly warned about this 3 months ago! {hashtag}",
            "Why value investing principles from Warren Buffett are outperforming hyper-growth speculative tech stocks this quarter. {hashtag}",
            "Stock market update: S&P 500 records all-time high amidst record earnings revisions. What is your top pick? {hashtag}",
            "Financial literacy isn't taught in school, but understanding compound interest and ETF index funds will make you rich. {hashtag}",
            "Ethereum layer-2 gas fees dropped by 90% following the latest network upgrade. Mass adoption is closer than you think. {hashtag}",
            "Shocking inflation data released: consumer sentiment declines as energy prices surge. Are we heading into a recession? {hashtag}"
        ],
        'hashtags': ['#FinTech', '#Crypto', '#Bitcoin', '#StockMarket', '#Investing', '#Economy', '#Trading', '#WealthManagement']
    },
    'Health & Wellness': {
        'templates': [
            "Groundbreaking clinical trial reveals intermittent fasting coupled with strength training reverses metabolic syndrome markers! {hashtag}",
            "Stop drinking coffee immediately after waking up. Waiting 90 minutes optimizes adenosine levels and eliminates the afternoon crash. {hashtag}",
            "Mental health is just as vital as physical fitness. If you're feeling burned out, here is your reminder to rest and unplug. {hashtag}",
            "New neuroscience study confirms sleep deprivation damages cognitive retention by over 40%. Prioritize your 8 hours! {hashtag}",
            "5 scientifically proven habits that will dramatically reduce anxiety and boost focus within 14 days: {hashtag}",
            "The truth about ultra-processed foods: how hidden sugars and emulsifiers disrupt your gut microbiome. {hashtag}",
            "Longevity research breakthrough: cellular senescence therapies enter Phase 3 human trials with promising results. {hashtag}",
            "Hydration checklist for high performers: why electrolytes matter more than plain water when exercising. {hashtag}"
        ],
        'hashtags': ['#HealthAndWellness', '#Biohacking', '#FitnessGoals', '#MentalHealthMatters', '#Longevity', '#Nutrition', '#SleepScience', '#HealthyLiving']
    },
    'Entertainment & Culture': {
        'templates': [
            "The season finale was an absolute cinematic masterpiece! That unexpected cliffhanger left everyone speechless. {hashtag}",
            "Box office record shattered! The new sci-fi epic crossed $1 Billion faster than any release this decade. {hashtag}",
            "Controversial take: The original soundtrack carried the entire movie, the plot itself was mediocre at best. {hashtag}",
            "Historic night at the Global Music Awards! Independent artists sweep all major categories against major studio labels. {hashtag}",
            "Can we talk about the stunning cinematography and practical effects in this film? CGI looks cheap compared to this. {hashtag}",
            "Viral rumor confirmed: The legendary director is reuniting the original cast for a blockbuster sequel trilogy! {hashtag}",
            "Top 10 binge-worthy series to watch this weekend if you love psychological thrillers and mystery twists: {hashtag}",
            "Pop culture moment of the decade: live concert breaks global streaming records with 85 million concurrent viewers! {hashtag}"
        ],
        'hashtags': ['#PopCulture', '#MovieReview', '#Cinema', '#BoxOffice', '#MusicNews', '#BingeWatch', '#TrendingNow', '#Entertainment']
    },
    'Sports & Gaming': {
        'templates': [
            "Historic comeback! Trailing by 18 points in the final quarter, they pull off the greatest championship upset ever! {hashtag}",
            "The new Unreal Engine 5.5 open-world RPG gameplay teaser looks hyper-realistic. Is this the game of the year? {hashtag}",
            "World record shattered! Clocking sub-9.6s in the 100m sprint under torrential rain. Legendary performance! {hashtag}",
            "Transfer window drama: Record-breaking $180M deal agreed in the final seconds of the deadline day! {hashtag}",
            "Esports world championship breaks peak viewership records with over 6.4 million live viewers tuning in! {hashtag}",
            "Why tactical discipline and zonal marking won the derby today, breaking the opponent's counter-press completely. {hashtag}",
            "Next-gen gaming handheld benchmark: 120 FPS at 1080p on battery power for 4 hours straight. Impressive tech! {hashtag}",
            "Unbelievable buzzer-beater shot from half-court to clinch the playoff victory! Crowds are going wild in the arena! {hashtag}"
        ],
        'hashtags': ['#SportsHighlights', '#GamingCommunity', '#WorldRecord', '#Championship', '#Esports', '#MatchDay', '#GamerLife', '#Athletics']
    },
    'Business & Careers': {
        'templates': [
            "Leadership lesson from scaling a startup to $50M ARR without taking any venture capital funding: {hashtag}",
            "Resume advice that actually gets you interviews: Replace passive duty descriptions with quantified business impact. {hashtag}",
            "Why remote work is here to stay, despite executive mandates demanding full 5-day office returns. {hashtag}",
            "Top 7 high-income skills worth mastering this year if you want to future-proof your career against automation: {hashtag}",
            "Most meetings should have been an email, and most emails could have been resolved in a single Slack message. {hashtag}",
            "We just hit our company milestone: 100,000 paying business customers across 42 countries! Grateful to the entire team. {hashtag}",
            "The brutal truth about hiring in tech right now: portfolio projects and problem-solving beat fancy degrees every time. {hashtag}",
            "Work-life balance is not a luxury, it is the foundation of sustainable long-term high performance. {hashtag}"
        ],
        'hashtags': ['#CareerAdvice', '#Leadership', '#StartupGrowth', '#FutureOfWork', '#Entrepreneurship', '#RemoteWork', '#BusinessStrategy', '#Productivity']
    }
}


def generate_benchmark_dataset(num_samples: int = 5000, output_path: str = "data/social_media_trends.csv") -> pd.DataFrame:
    """Generates synthetic but statistically realistic social media trend dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    start_date = datetime.datetime(2026, 1, 1, 0, 0, 0)
    data = []
    
    for i in range(1, num_samples + 1):
        post_id = f"POST_{i:05d}"
        category = random.choice(CATEGORIES)
        platform = random.choice(PLATFORMS)
        
        # Author attributes (Log-normal distribution for realistic follower counts)
        # Median ~ 4,000 followers, heavy tail up to 5,000,000+
        followers = int(np.random.lognormal(mean=8.2, sigma=1.6))
        followers = max(50, min(5000000, followers))
        author_verified = 1 if (followers > 50000 and random.random() < 0.65) or random.random() < 0.05 else 0
        
        # Temporal attributes
        random_days = random.randint(0, 90)
        # Peak posting hours around lunch (12-14) and evening (18-21)
        hour_weights = [1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 9, 10, 11, 10, 9, 8, 9, 10, 12, 11, 9, 7, 4, 2]
        hour = random.choices(range(24), weights=hour_weights)[0]
        minute = random.randint(0, 59)
        post_time = start_date + datetime.timedelta(days=random_days, hours=hour, minutes=minute)
        day_of_week = post_time.weekday()
        
        # Media type distribution
        if platform == 'Instagram':
            media_type = random.choices(MEDIA_TYPES, weights=[0.05, 0.40, 0.35, 0.20])[0]
        elif platform == 'Twitter/X':
            media_type = random.choices(MEDIA_TYPES, weights=[0.45, 0.30, 0.20, 0.05])[0]
        elif platform == 'LinkedIn':
            media_type = random.choices(MEDIA_TYPES, weights=[0.30, 0.35, 0.15, 0.20])[0]
        else:  # Reddit
            media_type = random.choices(MEDIA_TYPES, weights=[0.55, 0.25, 0.15, 0.05])[0]
            
        # Post text & hashtags
        corpus_data = TOPIC_CORPUS[category]
        template = random.choice(corpus_data['templates'])
        selected_hashtags = random.sample(corpus_data['hashtags'], k=random.randint(1, 3))
        
        # Add random trending cross-over tags occasionally
        if random.random() < 0.25:
            selected_hashtags.append(random.choice(['#Breaking', '#Trending', '#Viral', '#MustWatch', '#Innovation', '#Insights']))
            
        post_text = template.format(hashtag=" ".join(selected_hashtags))
        
        # Compute NLP attributes
        nlp_dict = extract_nlp_features(post_text)
        
        # Engagement simulation based on real multi-factor virality mechanics:
        # Base multiplier from followers (sublinear due to algorithm decay)
        reach_factor = np.log10(followers + 10) * 1.5
        
        # Media engagement bonus
        media_mult = {'Text': 1.0, 'Image': 1.4, 'Video': 2.1, 'Carousel': 1.7}[media_type]
        
        # Sentiment intensity driver (High excitement or high urgency drives virality)
        sentiment_mult = 1.0 + abs(nlp_dict['sentiment_polarity']) * 0.8 + nlp_dict['sentiment_subjectivity'] * 0.5
        
        # Time-of-day optimal window bonus (12-14 and 18-21)
        time_mult = 1.4 if (12 <= hour <= 14 or 18 <= hour <= 21) else 1.0
        
        # Conversational hook bonus (questions, exclamations, optimal length)
        hook_mult = 1.0
        if nlp_dict['question_count'] > 0:
            hook_mult += 0.35
        if nlp_dict['exclamation_count'] > 0:
            hook_mult += 0.25
        if 80 <= nlp_dict['char_count'] <= 220:
            hook_mult += 0.20
            
        # Platform-specific multiplier
        plat_mult = {'Twitter/X': 1.1, 'Instagram': 1.3, 'LinkedIn': 1.0, 'Reddit': 1.25}[platform]
        
        # Random algorithmic virality shock (Power law heavy tail)
        # Most posts get standard distribution, ~12% get viral algorithmic boost
        viral_shock = np.random.pareto(a=2.5) if random.random() < 0.14 else np.random.uniform(0.3, 1.4)
        
        latent_virality = reach_factor * media_mult * sentiment_mult * time_mult * hook_mult * plat_mult * viral_shock
        
        # Generate realistic likes, retweets/shares, comments
        likes = int(np.clip(latent_virality * np.random.uniform(15, 60), 0, 1500000))
        shares = int(np.clip(likes * np.random.uniform(0.08, 0.45), 0, 450000))
        comments = int(np.clip(likes * np.random.uniform(0.03, 0.25), 0, 180000))
        
        # Composite Engagement Score
        # Formula: Likes + (Shares * 3.0) + (Comments * 2.0)
        raw_engagement = likes + (shares * 3.0) + (comments * 2.0)
        
        # Engagement rate per thousand followers
        engagement_rate = (raw_engagement / max(100, followers)) * 1000
        
        data.append({
            'post_id': post_id,
            'platform': platform,
            'category': category,
            'post_text': post_text,
            'timestamp': post_time.strftime("%Y-%m-%d %H:%M:%S"),
            'hour_of_day': hour,
            'day_of_week': day_of_week,
            'author_followers': followers,
            'author_verified': author_verified,
            'media_type': media_type,
            'hashtag_count': len(selected_hashtags),
            'hashtags': " ".join(selected_hashtags),
            'char_count': nlp_dict['char_count'],
            'word_count': nlp_dict['word_count'],
            'sentiment_polarity': nlp_dict['sentiment_polarity'],
            'sentiment_subjectivity': nlp_dict['sentiment_subjectivity'],
            'question_count': nlp_dict['question_count'],
            'exclamation_count': nlp_dict['exclamation_count'],
            'has_url': int(nlp_dict['has_url']),
            'likes': likes,
            'shares_retweets': shares,
            'comments': comments,
            'raw_engagement': raw_engagement,
            'engagement_rate': round(engagement_rate, 2)
        })
        
    df = pd.DataFrame(data)
    
    # Define Ground-Truth Virality Tier based on engagement quantiles
    # Low: Bottom 50%
    # Moderate: 50% - 85%
    # Viral: Top 15% (Breakout viral phenomenon)
    q50 = df['raw_engagement'].quantile(0.50)
    q85 = df['raw_engagement'].quantile(0.85)
    
    def assign_tier(val):
        if val >= q85:
            return 'Viral'
        elif val >= q50:
            return 'Moderate'
        else:
            return 'Low'
            
    df['virality_tier'] = df['raw_engagement'].apply(assign_tier)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records saved to {output_path}")
    print("Class distribution:")
    print(df['virality_tier'].value_counts(normalize=True))
    return df


if __name__ == '__main__':
    generate_benchmark_dataset(5000, "data/social_media_trends.csv")
