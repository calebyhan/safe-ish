# ML-Powered Meme Coin Day Trading System - Complete Project Specification

## Project Overview

### Mission Statement
Build a two-stage algorithmic trading system for meme coins that uses machine learning to filter out scam tokens (rug pulls, pump-and-dumps) before applying trading strategies to the remaining "cleaner" subset.

### Core Philosophy
**Defense First, Offense Second** - Use ML as a protective shield to eliminate 85-90% of fraudulent tokens, then deploy multiple trading strategies on the approved subset.

### Expected Outcomes
- Filter accuracy: 85-90% scam detection rate
- Trading win rate: 55-65% on approved tokens
- Learning outcome: Comprehensive experience with ML, algorithmic trading, and crypto markets
- Capital: Starting with $500-2000 for educational purposes

---

## System Architecture

### High-Level Flow
```
1. Data Collection → 2. ML Filter → 3. Strategy Selection → 4. Execution → 5. Monitoring → 6. Analysis
```

### Two-Stage Design

**Stage 1: The Bouncer (ML Defense Layer)**
- Purpose: Identify and reject scam tokens
- Method: Machine learning classification
- Input: Token features from DexScreener API + on-chain data
- Output: Risk score (0-1) and recommendation (REJECT/CAUTION/PASS)

**Stage 2: The Trader (Strategy Layer)**
- Purpose: Generate profits on approved tokens
- Method: Multiple trading strategies
- Input: ML-approved tokens
- Output: Buy/sell signals with position sizing

---

## Technical Stack

### Languages & Frameworks
- **Primary Language:** Python 3.10+
- **ML Libraries:** scikit-learn, XGBoost, pandas, numpy
- **API/Web:** requests, aiohttp (async), FastAPI (optional API)
- **Visualization:** matplotlib, plotly, streamlit
- **Database:** SQLite for local storage, PostgreSQL for production
- **Testing:** pytest

### APIs & Data Sources
- **DexScreener API** (primary): Token data, pairs, volume, liquidity
- **Blockchain Explorers:** Solscan (Solana), Etherscan (Ethereum)
- **Social Data:** Twitter API, Telegram Bot API
- **Security Tools:** GoPlus Labs API, TokenSniffer API

### Infrastructure
- **Development:** Local machine, Jupyter notebooks
- **Production:** VPS (DigitalOcean/AWS) or local server
- **Monitoring:** Discord/Telegram bot for alerts

---

## Phase 1: Data Collection Pipeline

### 1.1 DexScreener Integration

**Endpoints to Use:**
```python
BASE_URL = "https://api.dexscreener.com"

# Primary endpoints
GET /token-boosts/latest/v1          # Rate limit: 60/min
GET /latest/dex/tokens/{chain}/{address}  # Rate limit: 300/min
GET /latest/dex/search?q={query}     # Rate limit: 300/min
GET /token-pairs/v1/{chain}/{address}     # Rate limit: 300/min
```

**Rate Limit Handling:**
```python
class RateLimiter:
    def __init__(self, requests_per_minute):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0
    
    async def wait_if_needed(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_request = time.time()
```

### 1.2 Feature Collection

**Core Feature Set (30+ features):**

```python
FEATURES = {
    # Basic Info
    'token_age_hours': float,
    'liquidity_usd': float,
    'market_cap': float,
    'pair_created_at': timestamp,
    
    # Volume Metrics (time-series critical)
    'volume_1min': float,
    'volume_5min': float,
    'volume_1hour': float,
    'volume_24hour': float,
    'volume_change_rate': float,  # Derived
    
    # Transaction Patterns
    'buy_count_1min': int,
    'buy_count_5min': int,
    'sell_count_1min': int,
    'sell_count_5min': int,
    'buy_sell_ratio': float,  # Derived
    'unique_buyers_5min': int,
    'unique_sellers_5min': int,
    
    # Holder Analysis
    'total_holders': int,
    'top1_holder_pct': float,
    'top5_holder_pct': float,
    'top10_holder_pct': float,
    'holder_concentration_gini': float,  # Derived
    
    # Price Dynamics
    'price_usd': float,
    'price_change_1min': float,
    'price_change_5min': float,
    'price_change_1hour': float,
    'price_change_24hour': float,
    'price_volatility_5min': float,  # Derived: std dev
    'all_time_high': float,
    'ath_ratio': float,  # current / ath
    
    # Liquidity Patterns
    'liquidity_locked': bool,
    'liquidity_lock_duration_days': float,
    'liquidity_change_rate_1hour': float,
    'liquidity_to_mcap_ratio': float,  # Derived
    
    # Smart Contract Features
    'mint_authority_revoked': bool,
    'has_blacklist_function': bool,
    'has_whitelist_function': bool,
    'contract_verified': bool,
    'has_proxy_pattern': bool,
    
    # Wash Trading Detection
    'circular_volume_pct': float,  # Same addresses buy/sell
    'repetitive_trade_count': int,  # Exact same amounts
    'zero_risk_trades': int,  # Buy/sell same amount same day
    
    # Social Metrics (if available)
    'twitter_mentions_24h': int,
    'telegram_member_count': int,
    'telegram_message_rate': float,
    
    # Temporal Features
    'hour_of_day': int,
    'day_of_week': int,
    'is_weekend': bool,
}
```

**Feature Engineering Functions:**
```python
def engineer_features(raw_features):
    """
    Create derived features from raw data
    """
    engineered = raw_features.copy()
    
    # Momentum indicators
    engineered['volume_acceleration'] = (
        raw_features['volume_5min'] / raw_features['volume_1min']
        if raw_features['volume_1min'] > 0 else 0
    )
    
    # Risk indicators
    engineered['whale_dominance'] = (
        raw_features['top10_holder_pct'] / raw_features['total_holders']
        if raw_features['total_holders'] > 0 else 1
    )
    
    # Safety score
    engineered['safety_features_count'] = sum([
        raw_features['liquidity_locked'],
        raw_features['mint_authority_revoked'],
        not raw_features['has_blacklist_function'],
        raw_features['contract_verified'],
    ])
    
    # Time-based risk
    engineered['launch_phase_risk'] = (
        1.0 if raw_features['token_age_hours'] < 24 else
        0.7 if raw_features['token_age_hours'] < 168 else  # 1 week
        0.3
    )
    
    return engineered
```

### 1.3 Data Collection Script

**Historical Data Collector:**
```python
class HistoricalDataCollector:
    """
    Collect data for ML training
    Goal: 1000+ tokens with known outcomes
    """
    
    def __init__(self):
        self.dex_api = DexScreenerAPI()
        self.db = Database('training_data.db')
    
    async def collect_training_data(self, days_back=90):
        """
        Collect historical tokens and track their outcomes
        """
        dataset = []
        
        for date in self.generate_date_range(days_back):
            # Get trending tokens from that date
            tokens = await self.get_historical_trending(date)
            
            for token in tokens:
                # Collect features at launch (T=0)
                features_t0 = await self.collect_features_snapshot(
                    token, 
                    time_offset_hours=0
                )
                
                # Check outcome 24 hours later
                outcome = await self.check_token_fate(
                    token,
                    hours_later=24
                )
                
                # Store in database
                self.db.insert({
                    **features_t0,
                    'outcome_label': outcome['label'],
                    'outcome_price_change': outcome['price_change'],
                    'outcome_liquidity_change': outcome['liquidity_change'],
                    'is_rug_pull': outcome['is_rug_pull'],
                    'is_pump_dump': outcome['is_pump_dump'],
                })
                
                dataset.append({
                    'features': features_t0,
                    'label': outcome['label']
                })
        
        return dataset
    
    def check_token_fate(self, token, hours_later=24):
        """
        Determine if token was scam or legitimate
        """
        initial_price = token.price_at_start
        final_price = token.price_after_hours(hours_later)
        initial_liquidity = token.liquidity_at_start
        final_liquidity = token.liquidity_after_hours(hours_later)
        
        price_change = (final_price / initial_price) - 1
        liquidity_change = (final_liquidity / initial_liquidity) - 1
        
        # Classification logic
        is_rug_pull = (
            price_change < -0.95 or  # 95% crash
            liquidity_change < -0.80 or  # 80% liquidity removed
            token.trading_stopped
        )
        
        is_pump_dump = (
            token.max_price_increase > 5.0 and  # 5x pump
            price_change < -0.70  # Then 70% crash
        )
        
        is_wash_trading = (
            token.circular_volume_pct > 0.50
        )
        
        is_legitimate = (
            not is_rug_pull and
            not is_pump_dump and
            not is_wash_trading and
            token.still_trading and
            liquidity_change > -0.20  # Max 20% liquidity decrease
        )
        
        # Label assignment
        if is_rug_pull:
            label = 0  # RUG_PULL
        elif is_pump_dump:
            label = 1  # PUMP_DUMP
        elif is_wash_trading:
            label = 2  # WASH_TRADING
        elif is_legitimate:
            label = 3  # LEGITIMATE
        else:
            label = 4  # UNKNOWN
        
        return {
            'label': label,
            'price_change': price_change,
            'liquidity_change': liquidity_change,
            'is_rug_pull': is_rug_pull,
            'is_pump_dump': is_pump_dump,
        }
```

---

## Phase 2: ML Model Development

### 2.1 Model Architecture

**Primary Model: Gradient Boosting Classifier**

Research shows this performs best for rug pull detection (AUC up to 0.891).

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

class RugPullDetector:
    """
    ML model for detecting scam tokens
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.label_encoder = None
    
    def train(self, X, y):
        """
        Train the rug pull detection model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Binary classification: Scam (0,1,2) vs Legitimate (3)
        y_train_binary = (y_train <= 2).astype(int)
        y_test_binary = (y_test <= 2).astype(int)
        
        # Try multiple models
        models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
        }
        
        best_score = 0
        best_model = None
        
        for name, model in models.items():
            # Train
            model.fit(X_train, y_train_binary)
            
            # Evaluate
            score = model.score(X_test, y_test_binary)
            print(f"{name} Accuracy: {score:.3f}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train_binary, cv=5)
            print(f"{name} CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
            
            if score > best_score:
                best_score = score
                best_model = model
        
        self.model = best_model
        self.feature_columns = X.columns.tolist()
        
        # Generate detailed report
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test_binary, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test_binary, y_pred))
        
        return self.model
    
    def predict_risk(self, features):
        """
        Predict rug pull risk for a token
        Returns: probability of being a scam (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Ensure features are in correct order
        X = features[self.feature_columns]
        
        # Get probability of being scam
        proba = self.model.predict_proba([X])[0]
        scam_probability = proba[1]  # Probability of class 1 (scam)
        
        return scam_probability
    
    def explain_prediction(self, features):
        """
        Use SHAP to explain why model made this prediction
        """
        import shap
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(features)
        
        # Return top contributing features
        feature_importance = dict(zip(
            self.feature_columns,
            shap_values[0]
        ))
        
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return sorted_features[:10]  # Top 10 features
```

### 2.2 Multi-Model Approach

For better coverage, train specialized models:

```python
class MultiModelDetector:
    """
    Ensemble of specialized detectors
    """
    
    def __init__(self):
        self.rug_pull_detector = RugPullDetector()
        self.wash_trading_detector = WashTradingDetector()
        self.pump_dump_detector = PumpDumpDetector()
    
    def train_all(self, dataset):
        """
        Train specialized models on different scam types
        """
        # Prepare datasets for each model
        rug_pull_data = self.prepare_rug_pull_dataset(dataset)
        wash_data = self.prepare_wash_trading_dataset(dataset)
        pump_data = self.prepare_pump_dump_dataset(dataset)
        
        # Train each
        self.rug_pull_detector.train(rug_pull_data)
        self.wash_trading_detector.train(wash_data)
        self.pump_dump_detector.train(pump_data)
    
    def comprehensive_risk_assessment(self, token_features):
        """
        Get risk scores from all models
        """
        scores = {
            'rug_pull_risk': self.rug_pull_detector.predict_risk(token_features),
            'wash_trading_risk': self.wash_trading_detector.predict_risk(token_features),
            'pump_dump_risk': self.pump_dump_detector.predict_risk(token_features),
        }
        
        # Weighted combination
        overall_risk = (
            scores['rug_pull_risk'] * 0.50 +  # Most dangerous
            scores['pump_dump_risk'] * 0.30 +
            scores['wash_trading_risk'] * 0.20
        )
        
        return {
            'overall_risk': overall_risk,
            'individual_scores': scores,
            'recommendation': self.categorize_risk(overall_risk)
        }
    
    def categorize_risk(self, risk_score):
        """
        Convert risk score to actionable recommendation
        """
        if risk_score > 0.80:
            return 'REJECT'  # Extreme risk
        elif risk_score > 0.50:
            return 'CAUTION'  # High risk
        elif risk_score > 0.30:
            return 'MONITOR'  # Medium risk
        else:
            return 'PASS'  # Low risk
```

### 2.3 Model Evaluation & Metrics

```python
class ModelEvaluator:
    """
    Comprehensive model evaluation
    """
    
    def evaluate(self, model, X_test, y_test):
        """
        Generate full evaluation report
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score
        )
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba)
        }
        
        # Cost-benefit analysis
        # False Negative (miss a scam) = lose money
        # False Positive (reject good token) = miss opportunity
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Assuming we lose $50 per scam and miss $20 per good token
        cost_fn = fn * 50  # Missed scams
        cost_fp = fp * 20  # Missed opportunities
        savings_tp = tp * 50  # Caught scams
        
        metrics['expected_value'] = savings_tp - cost_fn - cost_fp
        metrics['false_negatives'] = fn
        metrics['false_positives'] = fp
        
        return metrics
    
    def feature_importance_analysis(self, model, feature_names):
        """
        Identify most important features
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            feature_importance = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            
            print("\nTop 15 Most Important Features:")
            for feature, importance in feature_importance[:15]:
                print(f"{feature:30s}: {importance:.4f}")
            
            return feature_importance
```

---

## Phase 3: Trading Strategies

### 3.1 Strategy Base Class

```python
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    """
    
    def __init__(self, name):
        self.name = name
        self.min_ml_risk_threshold = 0.40  # Only trade tokens below this risk
    
    @abstractmethod
    def generate_signals(self, approved_tokens):
        """
        Generate buy/sell signals
        Returns: List of signal dictionaries
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal, portfolio):
        """
        Determine how much to invest
        """
        pass
    
    def validate_token(self, token):
        """
        Check if token meets strategy requirements
        """
        if token.ml_risk_score > self.min_ml_risk_threshold:
            return False
        
        if token.liquidity_usd < 10000:
            return False
        
        return True
```

### 3.2 Momentum Breakout Strategy

```python
class MomentumBreakoutStrategy(BaseStrategy):
    """
    Buy tokens breaking out with volume
    """
    
    def __init__(self):
        super().__init__("Momentum Breakout")
        self.volume_surge_threshold = 3.0  # 3x average
        self.breakout_threshold = 1.05  # 5% above resistance
    
    def generate_signals(self, approved_tokens):
        """
        Find breakout opportunities
        """
        signals = []
        
        for token in approved_tokens:
            if not self.validate_token(token):
                continue
            
            # Get price history
            candles = self.get_candles(token, interval='5m', limit=50)
            
            # Check for breakout
            if self.is_breaking_out(candles):
                # Check volume confirmation
                if self.has_volume_surge(candles):
                    # Calculate targets
                    support_level = self.find_support(candles)
                    
                    signal = {
                        'strategy': self.name,
                        'token': token,
                        'action': 'BUY',
                        'entry_price': token.current_price,
                        'stop_loss': support_level * 0.95,
                        'take_profit_1': token.current_price * 1.5,
                        'take_profit_2': token.current_price * 2.5,
                        'confidence': self.calculate_confidence(candles),
                        'reasoning': self.explain_signal(candles)
                    }
                    
                    signals.append(signal)
        
        return signals
    
    def is_breaking_out(self, candles):
        """
        Check if price broke above resistance
        """
        # Find recent resistance (highest high in last 20 candles)
        recent_high = candles['high'][-20:].max()
        current_price = candles['close'][-1]
        
        # Breaking out if current > resistance by threshold
        return current_price > (recent_high * self.breakout_threshold)
    
    def has_volume_surge(self, candles):
        """
        Check for volume confirmation
        """
        current_volume = candles['volume'][-1]
        avg_volume = candles['volume'][:-1].mean()
        
        return current_volume > (avg_volume * self.volume_surge_threshold)
    
    def find_support(self, candles):
        """
        Identify support level for stop loss
        """
        # Find recent lows
        recent_lows = candles['low'][-20:]
        
        # Support is near recent swing low
        support = recent_lows.min()
        
        return support
    
    def calculate_confidence(self, candles):
        """
        Confidence score 0-1 based on signal strength
        """
        volume_ratio = candles['volume'][-1] / candles['volume'][:-1].mean()
        price_momentum = (candles['close'][-1] / candles['close'][-10] - 1)
        
        # Normalize to 0-1
        confidence = min(
            0.5 + (volume_ratio / 10) + (price_momentum * 2),
            1.0
        )
        
        return confidence
    
    def calculate_position_size(self, signal, portfolio):
        """
        Size based on confidence and risk
        """
        base_size = portfolio.max_position_size
        
        # Adjust for confidence
        size = base_size * signal['confidence']
        
        # Adjust for ML risk (lower risk = larger position)
        risk_multiplier = 1 - signal['token'].ml_risk_score
        size *= risk_multiplier
        
        return size
```

### 3.3 Dip Buying Strategy

```python
class DipBuyingStrategy(BaseStrategy):
    """
    Buy dips on established, legitimate tokens
    """
    
    def __init__(self):
        super().__init__("Dip Buyer")
        self.min_token_age_days = 30
        self.dip_range = (-40, -20)  # Buy 20-40% dips
    
    def generate_signals(self, approved_tokens):
        """
        Find dip opportunities on established tokens
        """
        signals = []
        
        for token in approved_tokens:
            # Only established tokens
            if token.age_days < self.min_token_age_days:
                continue
            
            if not self.validate_token(token):
                continue
            
            # Check if dipping
            price_change_24h = token.price_change_pct_24h
            
            if self.dip_range[0] < price_change_24h < self.dip_range[1]:
                # Is this healthy correction or death spiral?
                health = self.assess_dip_health(token)
                
                if health['is_healthy']:
                    signal = {
                        'strategy': self.name,
                        'token': token,
                        'action': 'BUY',
                        'entry_price': token.current_price,
                        'stop_loss': token.current_price * 0.85,
                        'take_profit_1': token.price_24h_ago * 0.95,  # Back to 95% of previous
                        'take_profit_2': token.price_24h_ago * 1.05,  # Recover and profit
                        'confidence': health['confidence'],
                        'reasoning': f"Dip buying: {abs(price_change_24h):.1f}% correction on established token"
                    }
                    
                    signals.append(signal)
        
        return signals
    
    def assess_dip_health(self, token):
        """
        Distinguish healthy correction from collapse
        """
        # Check fundamentals
        liquidity_stable = (
            token.liquidity_usd / token.liquidity_usd_24h_ago > 0.9
        )
        
        volume_spike = (
            token.volume_24h > token.avg_volume_7d * 1.5
        )
        
        holders_stable = (
            token.holders_change_pct_24h > -5  # Max 5% holder loss
        )
        
        # Healthy if fundamentals intact
        is_healthy = liquidity_stable and holders_stable
        
        # Higher confidence if volume spike (capitulation?)
        confidence = 0.7 if (is_healthy and volume_spike) else 0.5
        
        return {
            'is_healthy': is_healthy,
            'confidence': confidence,
            'reasons': {
                'liquidity_stable': liquidity_stable,
                'volume_spike': volume_spike,
                'holders_stable': holders_stable
            }
        }
    
    def calculate_position_size(self, signal, portfolio):
        """
        Larger positions for established dips
        """
        base_size = portfolio.max_position_size
        
        # Dip buying can be more aggressive on established tokens
        size = base_size * 1.2 * signal['confidence']
        
        return min(size, portfolio.max_position_size * 1.5)
```

### 3.4 Social Momentum Strategy

```python
class SocialMomentumStrategy(BaseStrategy):
    """
    Trade tokens gaining social traction
    """
    
    def __init__(self):
        super().__init__("Social Momentum")
        self.twitter_api = TwitterAPI()
        self.telegram_api = TelegramAPI()
        self.min_social_score = 70
    
    def generate_signals(self, approved_tokens):
        """
        Find tokens with strong social momentum
        """
        signals = []
        
        for token in approved_tokens:
            if not self.validate_token(token):
                continue
            
            # Calculate social score
            social_score = self.calculate_social_score(token)
            
            if social_score >= self.min_social_score:
                # Check if momentum is accelerating
                trend = self.get_trend_direction(token)
                
                if trend == 'ACCELERATING':
                    signal = {
                        'strategy': self.name,
                        'token': token,
                        'action': 'BUY',
                        'entry_price': token.current_price,
                        'stop_loss': token.current_price * 0.85,
                        'take_profit_1': token.current_price * 1.8,
                        'take_profit_2': token.current_price * 3.0,
                        'confidence': social_score / 100,
                        'social_score': social_score,
                        'reasoning': f"Strong social momentum: {social_score}/100"
                    }
                    
                    signals.append(signal)
        
        return signals
    
    def calculate_social_score(self, token):
        """
        Aggregate social metrics into single score
        """
        try:
            # Gather metrics
            twitter_mentions = self.twitter_api.count_mentions(
                token.symbol, 
                hours=24
            )
            
            twitter_sentiment = self.twitter_api.get_sentiment(
                token.symbol
            )  # -1 to 1
            
            telegram_members = 0
            telegram_activity = 0
            if token.telegram_url:
                telegram_members = self.telegram_api.get_member_count(
                    token.telegram_url
                )
                telegram_activity = self.telegram_api.get_message_rate(
                    token.telegram_url
                )  # messages per hour
            
            # Weighted scoring
            score = (
                min(twitter_mentions / 100, 30) +  # Max 30 points
                ((twitter_sentiment + 1) / 2) * 20 +  # 0-20 points
                min(telegram_members / 1000, 25) +  # Max 25 points
                min(telegram_activity * 2, 25)  # Max 25 points
            )
            
            return min(score, 100)
            
        except Exception as e:
            print(f"Error calculating social score: {e}")
            return 0
    
    def get_trend_direction(self, token):
        """
        Determine if social momentum is accelerating
        """
        mentions_now = self.twitter_api.count_mentions(token.symbol, hours=6)
        mentions_prev = self.twitter_api.count_mentions(
            token.symbol, 
            hours_start=6, 
            hours_end=12
        )
        
        if mentions_now > mentions_prev * 1.5:
            return 'ACCELERATING'
        elif mentions_now > mentions_prev:
            return 'GROWING'
        else:
            return 'DECLINING'
    
    def calculate_position_size(self, signal, portfolio):
        """
        Size based on social score strength
        """
        base_size = portfolio.max_position_size
        
        # Higher social score = larger position
        social_multiplier = signal['social_score'] / 100
        
        return base_size * social_multiplier
```

### 3.5 Smart Money Following Strategy

```python
class SmartMoneyStrategy(BaseStrategy):
    """
    Copy trades from consistently profitable wallets
    """
    
    def __init__(self):
        super().__init__("Smart Money")
        self.smart_wallets = []
        self.min_wallet_trades = 20
        self.min_wallet_winrate = 0.65
    
    def identify_smart_wallets(self):
        """
        Find wallets with proven track records
        """
        # Scan top performers
        all_wallets = self.scan_top_traders(days=30)
        
        smart_wallets = []
        
        for wallet in all_wallets:
            stats = self.analyze_wallet_performance(wallet)
            
            if (stats['win_rate'] > self.min_wallet_winrate and
                stats['total_trades'] > self.min_wallet_trades and
                stats['avg_profit_multiplier'] > 2.0):
                
                smart_wallets.append({
                    'address': wallet,
                    'win_rate': stats['win_rate'],
                    'total_trades': stats['total_trades'],
                    'avg_profit': stats['avg_profit_multiplier']
                })
        
        self.smart_wallets = smart_wallets
        print(f"Identified {len(smart_wallets)} smart wallets")
        
        return smart_wallets
    
    def generate_signals(self, approved_tokens):
        """
        Monitor smart wallet activity
        """
        signals = []
        
        for wallet in self.smart_wallets:
            # Check recent buys (last hour)
            recent_buys = self.get_wallet_recent_buys(
                wallet['address'],
                hours=1
            )
            
            for buy in recent_buys:
                # Is this an approved token?
                if buy.token in approved_tokens:
                    # Smart wallet bought an ML-approved token!
                    signal = {
                        'strategy': self.name,
                        'token': buy.token,
                        'action': 'BUY',
                        'entry_price': buy.token.current_price,
                        'stop_loss': buy.token.current_price * 0.85,
                        'take_profit_1': buy.token.current_price * 2.0,
                        'take_profit_2': buy.token.current_price * 3.5,
                        'confidence': 0.85,  # High confidence
                        'smart_wallet': wallet['address'],
                        'wallet_winrate': wallet['win_rate'],
                        'reasoning': f"Smart wallet (WR: {wallet['win_rate']:.1%}) bought this token"
                    }
                    
                    signals.append(signal)
        
        return signals
    
    def calculate_position_size(self, signal, portfolio):
        """
        Match smart wallet's conviction
        """
        base_size = portfolio.max_position_size
        
        # Higher wallet win rate = more confidence
        wallet_confidence = signal['wallet_winrate']
        
        return base_size * 1.3 * wallet_confidence  # Aggressive on smart money
```

---

## Phase 4: Execution & Portfolio Management

### 4.1 Portfolio Manager

```python
class Portfolio:
    """
    Manage capital, positions, and risk
    """
    
    def __init__(self, initial_capital, max_position_size=100, max_positions=5):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        
        self.open_positions = []
        self.closed_positions = []
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def can_open_position(self):
        """Check if we can open new position"""
        return len(self.open_positions) < self.max_positions
    
    def open_position(self, signal):
        """Open a new position"""
        if not self.can_open_position():
            print("Max positions reached")
            return False
        
        # Calculate position size
        position_size = min(
            signal['position_size'],
            self.current_capital * 0.20  # Max 20% per position
        )
        
        position = Position(
            token=signal['token'],
            strategy=signal['strategy'],
            entry_price=signal['entry_price'],
            stop_loss=signal['stop_loss'],
            take_profit_1=signal.get('take_profit_1'),
            take_profit_2=signal.get('take_profit_2'),
            size_usd=position_size,
            opened_at=datetime.now()
        )
        
        self.open_positions.append(position)
        self.current_capital -= position_size
        self.total_trades += 1
        
        print(f"✅ Opened position: {position.token.symbol}")
        print(f"   Strategy: {position.strategy}")
        print(f"   Size: ${position_size:.2f}")
        print(f"   Entry: ${signal['entry_price']:.6f}")
        
        return True
    
    def close_position(self, position, reason, current_price=None):
        """Close an existing position"""
        if current_price is None:
            current_price = position.token.current_price
        
        # Calculate P&L
        pnl_pct = (current_price / position.entry_price - 1) * 100
        pnl_usd = position.size_usd * (pnl_pct / 100)
        
        # Update capital
        self.current_capital += position.size_usd + pnl_usd
        
        # Track statistics
        if pnl_usd > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Move to closed
        position.closed_at = datetime.now()
        position.exit_price = current_price
        position.pnl_usd = pnl_usd
        position.pnl_pct = pnl_pct
        position.close_reason = reason
        
        self.closed_positions.append(position)
        self.open_positions.remove(position)
        
        print(f"🔄 Closed position: {position.token.symbol}")
        print(f"   Reason: {reason}")
        print(f"   P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)")
        print(f"   Hold time: {position.hold_time_hours:.1f}h")
        
        return pnl_usd
    
    def get_statistics(self):
        """Get portfolio statistics"""
        win_rate = (
            self.winning_trades / self.total_trades 
            if self.total_trades > 0 else 0
        )
        
        total_pnl = sum(p.pnl_usd for p in self.closed_positions)
        roi = (self.current_capital / self.initial_capital - 1) * 100
        
        avg_win = np.mean([
            p.pnl_usd for p in self.closed_positions if p.pnl_usd > 0
        ]) if self.winning_trades > 0 else 0
        
        avg_loss = np.mean([
            p.pnl_usd for p in self.closed_positions if p.pnl_usd <= 0
        ]) if self.losing_trades > 0 else 0
        
        return {
            'current_capital': self.current_capital,
            'total_pnl': total_pnl,
            'roi_pct': roi,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'open_positions': len(self.open_positions)
        }

class Position:
    """Represents a single trading position"""
    
    def __init__(self, token, strategy, entry_price, stop_loss, 
                 take_profit_1, take_profit_2, size_usd, opened_at):
        self.token = token
        self.strategy = strategy
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit_1 = take_profit_1
        self.take_profit_2 = take_profit_2
        self.size_usd = size_usd
        self.opened_at = opened_at
        
        self.closed_at = None
        self.exit_price = None
        self.pnl_usd = None
        self.pnl_pct = None
        self.close_reason = None
    
    @property
    def hold_time_hours(self):
        """Calculate hold time"""
        end = self.closed_at if self.closed_at else datetime.now()
        return (end - self.opened_at).total_seconds() / 3600
    
    @property
    def age_hours(self):
        """Time since position opened"""
        return (datetime.now() - self.opened_at).total_seconds() / 3600
```

### 4.2 Position Monitor

```python
class PositionMonitor:
    """
    Monitor open positions and trigger closes
    """
    
    def __init__(self, portfolio, ml_filter):
        self.portfolio = portfolio
        self.ml_filter = ml_filter
    
    async def monitor_positions(self):
        """
        Check all open positions for exit conditions
        """
        for position in self.portfolio.open_positions:
            current_price = await self.get_current_price(position.token)
            
            # Check exit conditions
            should_close, reason = self.check_exit_conditions(
                position, 
                current_price
            )
            
            if should_close:
                self.portfolio.close_position(
                    position, 
                    reason, 
                    current_price
                )
    
    def check_exit_conditions(self, position, current_price):
        """
        Determine if position should be closed
        """
        # 1. Stop loss hit
        if current_price <= position.stop_loss:
            return True, "Stop loss triggered"
        
        # 2. Take profit 1 hit
        if position.take_profit_1 and current_price >= position.take_profit_1:
            return True, "Take profit 1 hit"
        
        # 3. Take profit 2 hit
        if position.take_profit_2 and current_price >= position.take_profit_2:
            return True, "Take profit 2 hit"
        
        # 4. Time-based exit (don't hold overnight)
        if position.age_hours > 12:
            return True, "Time limit exceeded (12h)"
        
        # 5. Trailing stop (if in profit)
        pnl_pct = (current_price / position.entry_price - 1) * 100
        if pnl_pct > 20:  # If up 20%+
            # Trail stop to lock in profit
            trailing_stop = current_price * 0.90  # 10% trailing
            if current_price < trailing_stop:
                return True, "Trailing stop hit"
        
        # 6. ML risk increased (emergency exit)
        new_risk_assessment = self.ml_filter.triage_token(
            position.token.address,
            position.token.chain_id
        )
        
        if new_risk_assessment['risk_score'] > 0.70:
            return True, "⚠️ ML risk increased - emergency exit!"
        
        # 7. Liquidity crisis
        current_liquidity = position.token.liquidity_usd
        if current_liquidity < position.token.initial_liquidity * 0.70:
            return True, "Liquidity dropped 30%+"
        
        return False, None
```

---

## Phase 5: Main Trading Bot

### 5.1 Orchestrator

```python
class MemeC oinTradingBot:
    """
    Main orchestrator - ties everything together
    """
    
    def __init__(self, initial_capital=1000):
        # Stage 1: ML Filter
        self.ml_filter = MultiModelDetector()
        
        # Stage 2: Strategies
        self.strategies = [
            MomentumBreakoutStrategy(),
            DipBuyingStrategy(),
            SocialMomentumStrategy(),
            SmartMoneyStrategy(),
        ]
        
        # Portfolio
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            max_position_size=100,
            max_positions=5
        )
        
        # Monitor
        self.monitor = PositionMonitor(self.portfolio, self.ml_filter)
        
        # Data
        self.dex_api = DexScreenerAPI()
        self.db = Database()
        
        # Config
        self.scan_interval_seconds = 60
        self.max_risk_score = 0.40
    
    async def run(self):
        """
        Main trading loop
        """
        print("🤖 Starting Meme Coin Trading Bot...")
        print(f"💰 Initial Capital: ${self.portfolio.initial_capital}")
        print(f"🎯 Max Position Size: ${self.portfolio.max_position_size}")
        print(f"📊 Active Strategies: {len(self.strategies)}")
        print("="*60)
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                print(f"\n🔄 Iteration {iteration} - {datetime.now()}")
                
                # Step 1: Get trending tokens
                trending_tokens = await self.fetch_trending_tokens()
                print(f"📈 Found {len(trending_tokens)} trending tokens")
                
                # Step 2: ML Filtering
                approved_tokens = await self.filter_tokens(trending_tokens)
                print(f"✅ Approved {len(approved_tokens)} tokens after ML filter")
                
                # Step 3: Generate signals from strategies
                all_signals = []
                for strategy in self.strategies:
                    signals = strategy.generate_signals(approved_tokens)
                    if signals:
                        print(f"   {strategy.name}: {len(signals)} signals")
                        all_signals.extend(signals)
                
                # Step 4: Rank and execute best signals
                if all_signals:
                    ranked_signals = self.rank_signals(all_signals)
                    await self.execute_signals(ranked_signals)
                
                # Step 5: Monitor open positions
                await self.monitor.monitor_positions()
                
                # Step 6: Print status
                self.print_status()
                
                # Wait before next iteration
                await asyncio.sleep(self.scan_interval_seconds)
                
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)
    
    async def fetch_trending_tokens(self):
        """
        Get trending tokens from DexScreener
        """
        # Combine multiple sources
        trending = []
        
        # Latest boosts
        boosts = await self.dex_api.get_boosted_tokens()
        trending.extend(boosts)
        
        # Search popular keywords
        for keyword in ['pepe', 'doge', 'shib', 'moon']:
            results = await self.dex_api.search(keyword)
            trending.extend(results[:10])
        
        # Deduplicate
        unique_tokens = {t.address: t for t in trending}
        
        return list(unique_tokens.values())
    
    async def filter_tokens(self, tokens):
        """
        Apply ML filter to remove scams
        """
        approved = []
        rejected = []
        
        for token in tokens:
            # Collect features
            features = await self.collect_token_features(token)
            
            # ML assessment
            assessment = self.ml_filter.comprehensive_risk_assessment(features)
            
            # Store risk score on token
            token.ml_risk_score = assessment['overall_risk']
            token.ml_recommendation = assessment['recommendation']
            
            # Filter decision
            if assessment['recommendation'] == 'PASS':
                approved.append(token)
                self.log_token_result(token, 'APPROVED', assessment)
            elif assessment['recommendation'] == 'MONITOR':
                # Could add to watch list
                pass
            else:
                rejected.append(token)
                self.log_token_result(token, 'REJECTED', assessment)
        
        # Log filtering stats
        self.db.log_filter_results(
            timestamp=datetime.now(),
            total_scanned=len(tokens),
            approved=len(approved),
            rejected=len(rejected)
        )
        
        return approved
    
    def rank_signals(self, signals):
        """
        Rank signals by quality
        """
        # Score each signal
        for signal in signals:
            score = self.calculate_signal_score(signal)
            signal['quality_score'] = score
        
        # Sort by score
        ranked = sorted(
            signals, 
            key=lambda x: x['quality_score'], 
            reverse=True
        )
        
        return ranked
    
    def calculate_signal_score(self, signal):
        """
        Unified signal scoring
        """
        score = 0
        
        # Base confidence from strategy
        score += signal['confidence'] * 40
        
        # ML risk (lower is better)
        ml_bonus = (1 - signal['token'].ml_risk_score) * 30
        score += ml_bonus
        
        # Token fundamentals
        if signal['token'].liquidity_usd > 50000:
            score += 10
        if signal['token'].age_days > 7:
            score += 10
        if signal['token'].liquidity_locked:
            score += 10
        
        return score
    
    async def execute_signals(self, ranked_signals):
        """
        Execute top signals
        """
        for signal in ranked_signals:
            # Can we open more positions?
            if not self.portfolio.can_open_position():
                print("⚠️ Max positions reached")
                break
            
            # Calculate position size
            signal['position_size'] = signal['strategy_obj'].calculate_position_size(
                signal, 
                self.portfolio
            )
            
            # Execute
            success = self.portfolio.open_position(signal)
            
            if success:
                # Log to database
                self.db.log_trade(signal, 'OPEN')
                
                # Send notification
                await self.send_notification(
                    f"🎯 Opened {signal['strategy']} position: "
                    f"{signal['token'].symbol} @ ${signal['entry_price']:.6f}"
                )
    
    def print_status(self):
        """
        Print current status
        """
        stats = self.portfolio.get_statistics()
        
        print("\n" + "="*60)
        print("📊 PORTFOLIO STATUS")
        print("="*60)
        print(f"💰 Capital: ${stats['current_capital']:.2f}")
        print(f"📈 Total P&L: ${stats['total_pnl']:+.2f} ({stats['roi_pct']:+.2f}%)")
        print(f"🎯 Win Rate: {stats['win_rate']:.1%} ({stats['winning_trades']}W / {stats['losing_trades']}L)")
        print(f"💵 Avg Win: ${stats['avg_win']:.2f} | Avg Loss: ${stats['avg_loss']:.2f}")
        print(f"📊 Profit Factor: {stats['profit_factor']:.2f}")
        print(f"🔓 Open Positions: {stats['open_positions']}/{self.portfolio.max_positions}")
        
        if self.portfolio.open_positions:
            print("\nOpen Positions:")
            for pos in self.portfolio.open_positions:
                current_price = pos.token.current_price
                pnl = (current_price / pos.entry_price - 1) * 100
                print(f"  {pos.token.symbol}: {pnl:+.2f}% ({pos.age_hours:.1f}h old)")
        
        print("="*60 + "\n")
```

---

## Phase 6: Deployment & Operations

### 6.1 Configuration

```yaml
# config.yaml

trading:
  initial_capital: 1000
  max_position_size: 100
  max_positions: 5
  scan_interval_seconds: 60
  
ml_filter:
  max_risk_score: 0.40
  model_path: "models/rug_detector.pkl"
  retrain_interval_days: 7
  min_training_samples: 500

strategies:
  momentum:
    enabled: true
    volume_threshold: 3.0
    breakout_threshold: 1.05
  
  dip_buyer:
    enabled: true
    min_token_age_days: 30
    dip_range: [-40, -20]
  
  social:
    enabled: true
    min_social_score: 70
  
  smart_money:
    enabled: true
    min_wallet_winrate: 0.65

risk_management:
  max_loss_per_trade_pct: 15
  max_drawdown_pct: 30
  stop_trading_on_drawdown: true
  time_based_exit_hours: 12

chains:
  - solana
  - ethereum
  - base

notifications:
  discord_webhook: "YOUR_WEBHOOK_URL"
  telegram_bot_token: "YOUR_BOT_TOKEN"
  telegram_chat_id: "YOUR_CHAT_ID"

database:
  type: "sqlite"
  path: "data/trading.db"
```

### 6.2 Database Schema

```sql
-- trades table
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    token_address TEXT NOT NULL,
    token_symbol TEXT,
    chain_id TEXT,
    strategy TEXT,
    action TEXT, -- 'OPEN' or 'CLOSE'
    price REAL,
    size_usd REAL,
    ml_risk_score REAL,
    stop_loss REAL,
    take_profit_1 REAL,
    take_profit_2 REAL,
    confidence REAL,
    close_reason TEXT,
    pnl_usd REAL,
    pnl_pct REAL,
    hold_time_hours REAL
);

-- ml_assessments table
CREATE TABLE ml_assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    token_address TEXT NOT NULL,
    token_symbol TEXT,
    chain_id TEXT,
    risk_score REAL,
    recommendation TEXT,
    rug_pull_risk REAL,
    wash_trading_risk REAL,
    pump_dump_risk REAL,
    outcome_label INTEGER, -- Actual outcome after 24h
    outcome_verified BOOLEAN DEFAULT 0
);

-- filter_stats table  
CREATE TABLE filter_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_scanned INTEGER,
    approved INTEGER,
    rejected INTEGER,
    rejection_rate REAL
);

-- portfolio_snapshots table
CREATE TABLE portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    capital REAL,
    total_pnl REAL,
    roi_pct REAL,
    total_trades INTEGER,
    win_rate REAL,
    open_positions INTEGER
);
```

### 6.3 Monitoring & Alerts

```python
class NotificationManager:
    """
    Send alerts via Discord/Telegram
    """
    
    def __init__(self, config):
        self.discord_webhook = config['discord_webhook']
        self.telegram_token = config['telegram_bot_token']
        self.telegram_chat = config['telegram_chat_id']
    
    async def send_trade_alert(self, trade_info):
        """
        Alert on trade execution
        """
        message = f"""
🎯 **Trade Alert**

Token: {trade_info['symbol']}
Action: {trade_info['action']}
Strategy: {trade_info['strategy']}
Price: ${trade_info['price']:.6f}
Size: ${trade_info['size']:.2f}
ML Risk: {trade_info['ml_risk']:.2f}
Confidence: {trade_info['confidence']:.0%}
        """
        
        await self.send_discord(message)
        await self.send_telegram(message)
    
    async def send_position_closed(self, position_info):
        """
        Alert on position close
        """
        emoji = "🟢" if position_info['pnl'] > 0 else "🔴"
        
        message = f"""
{emoji} **Position Closed**

Token: {position_info['symbol']}
Strategy: {position_info['strategy']}
Reason: {position_info['reason']}
Hold Time: {position_info['hold_hours']:.1f}h
P&L: ${position_info['pnl']:+.2f} ({position_info['pnl_pct']:+.2f}%)
        """
        
        await self.send_discord(message)
        await self.send_telegram(message)
    
    async def send_daily_summary(self, stats):
        """
        Daily performance summary
        """
        message = f"""
📊 **Daily Summary**

Capital: ${stats['capital']:.2f}
P&L: ${stats['pnl']:+.2f} ({stats['roi']:+.2f}%)
Trades: {stats['total_trades']} ({stats['win_rate']:.1%} WR)
Best Trade: {stats['best_trade']}
Worst Trade: {stats['worst_trade']}
        """
        
        await self.send_discord(message)
```

### 6.4 Continuous Learning

```python
class ModelRetrainer:
    """
    Periodically retrain ML models with new data
    """
    
    def __init__(self, db, ml_filter):
        self.db = db
        self.ml_filter = ml_filter
    
    async def retrain_models(self):
        """
        Retrain with latest verified outcomes
        """
        print("🔄 Retraining ML models...")
        
        # Get recent assessments with verified outcomes
        training_data = self.db.get_verified_assessments(days=30)
        
        if len(training_data) < 500:
            print("⚠️ Not enough training data yet")
            return
        
        # Prepare features and labels
        X = training_data[FEATURE_COLUMNS]
        y = training_data['outcome_label']
        
        # Retrain models
        self.ml_filter.rug_pull_detector.train(X, y)
        
        # Save updated models
        self.save_models()
        
        # Evaluate on test set
        metrics = self.evaluate_model_performance(X, y)
        
        print(f"✅ Models retrained")
        print(f"   New Accuracy: {metrics['accuracy']:.3f}")
        print(f"   New Precision: {metrics['precision']:.3f}")
        print(f"   New Recall: {metrics['recall']:.3f}")
    
    async def verify_past_predictions(self):
        """
        Check outcomes of past predictions
        """
        # Get unverified assessments older than 24h
        unverified = self.db.get_unverified_assessments(
            hours_old_min=24
        )
        
        for assessment in unverified:
            # Check actual outcome
            outcome = await self.check_token_outcome(
                assessment['token_address'],
                assessment['chain_id']
            )
            
            # Update database
            self.db.update_assessment_outcome(
                assessment['id'],
                outcome_label=outcome['label'],
                outcome_verified=True
            )
```

---

## Phase 7: Testing & Backtesting

### 7.1 Paper Trading Mode

```python
class PaperTradingBot(MemeCoinTradingBot):
    """
    Simulate trading without real money
    """
    
    def __init__(self, initial_capital=1000):
        super().__init__(initial_capital)
        self.paper_mode = True
        self.simulated_slippage = 0.01  # 1% slippage
    
    async def execute_signals(self, ranked_signals):
        """
        Simulate execution with slippage
        """
        for signal in ranked_signals:
            if not self.portfolio.can_open_position():
                break
            
            # Simulate slippage
            executed_price = signal['entry_price'] * (1 + self.simulated_slippage)
            signal['entry_price'] = executed_price
            
            # Calculate position size
            signal['position_size'] = signal['strategy_obj'].calculate_position_size(
                signal,
                self.portfolio
            )
            
            # "Execute" (just track it)
            self.portfolio.open_position(signal)
            
            print(f"📝 [PAPER] Opened position: {signal['token'].symbol}")
```

### 7.2 Backtesting Framework

```python
class Backtester:
    """
    Test strategies on historical data
    """
    
    def __init__(self, ml_filter, strategies):
        self.ml_filter = ml_filter
        self.strategies = strategies
    
    def backtest(self, historical_data, initial_capital=1000):
        """
        Run backtest on historical data
        """
        portfolio = Portfolio(initial_capital)
        
        # Group data by timestamp
        timestamps = historical_data['timestamp'].unique()
        timestamps.sort()
        
        for ts in timestamps:
            # Get tokens at this timestamp
            tokens_at_ts = historical_data[
                historical_data['timestamp'] == ts
            ]
            
            # Apply ML filter
            approved = []
            for _, token in tokens_at_ts.iterrows():
                risk = self.ml_filter.predict_risk(token[FEATURE_COLUMNS])
                if risk < 0.40:
                    approved.append(token)
            
            # Generate signals
            signals = []
            for strategy in self.strategies:
                strategy_signals = strategy.generate_signals(approved)
                signals.extend(strategy_signals)
            
            # Execute top signals
            for signal in signals[:3]:
                if portfolio.can_open_position():
                    portfolio.open_position(signal)
            
            # Check exits (use actual future prices from data)
            for position in portfolio.open_positions:
                future_price = self.get_future_price(
                    position.token,
                    ts,
                    historical_data
                )
                
                should_close, reason = self.check_exit(
                    position,
                    future_price
                )
                
                if should_close:
                    portfolio.close_position(position, reason, future_price)
        
        # Return results
        return portfolio.get_statistics()
```

---

## Project Timeline

### Week 1-2: Foundation
- Set up development environment
- Implement DexScreener API wrapper
- Build data collection pipeline
- Collect 500-1000 historical tokens

### Week 3: ML Development  
- Feature engineering
- Train initial ML models
- Evaluate model performance
- Implement multi-model approach

### Week 4: Strategy Implementation
- Code all 4 trading strategies
- Unit test each strategy
- Paper trading simulations

### Week 5: Integration
- Build main orchestrator
- Portfolio management
- Position monitoring
- Database setup

### Week 6: Testing & Deployment
- Backtesting
- Paper trading with live data
- Deploy to VPS
- Monitor performance

### Week 7+: Iteration
- Analyze results
- Retrain models
- Optimize strategies
- Scale up capital (if successful)

---

## Success Metrics

### ML Filter Performance
- **Target Accuracy:** 85-90%
- **False Negative Rate:** <10% (missing scams)
- **False Positive Rate:** <20% (rejecting good tokens)

### Trading Performance
- **Win Rate:** 55-65%
- **Average Win:** 1.8x - 2.5x
- **Average Loss:** -10% to -15%
- **Monthly ROI:** 10-20% (stretch goal)
- **Max Drawdown:** <30%

### Operational Metrics
- **Uptime:** >99%
- **API Error Rate:** <1%
- **Model Staleness:** <7 days
- **Decision Latency:** <5 seconds

---

## Risk Management Rules

### Hard Stops
1. **Circuit Breaker:** Stop trading if down 30% from starting capital
2. **Daily Loss Limit:** Stop if lose 10% in single day
3. **Per-Trade Risk:** Never risk more than 15% on single trade
4. **Position Concentration:** Max 20% of capital per position
5. **Overnight Holdings:** Preferably close all positions before 12 hours

### Monitoring
- Check positions every 60 seconds
- Verify ML risk every 15 minutes on open positions
- Retrain models weekly
- Review performance daily

---

## Next Steps for LLM Implementation

This document provides a comprehensive blueprint. To implement:

1. **Start with Phase 1:** Build data collection first
2. **Validate data quality:** Ensure features are accurate
3. **Train ML models:** Use collected historical data
4. **Test extensively:** Paper trade for 2+ weeks
5. **Deploy carefully:** Start with minimum capital
6. **Iterate rapidly:** Analyze and improve weekly

The system is modular - each phase can be built and tested independently before integration.

---

## Additional Resources Needed

### APIs & Keys
- DexScreener API (no key required)
- Twitter API (for social strategy)
- Telegram Bot API (for notifications)
- GoPlus Labs API (for security checks)

### Infrastructure
- VPS: 2 CPU, 4GB RAM minimum
- Database: SQLite (dev) or PostgreSQL (prod)
- Python 3.10+ environment

### Estimated Costs
- Development: $0 (use free tiers)
- Production VPS: $10-20/month
- API costs: $0-50/month
- Initial trading capital: $500-2000

---

## Final Notes

This is an **educational project** with real money at risk. Key principles:

1. **Start small** - Use money you can afford to lose
2. **Learn first** - Focus on learning over profits
3. **Iterate quickly** - Improve based on data
4. **Stay disciplined** - Follow the rules you set
5. **Have fun** - It's a learning experience!

The ML filter is the most valuable component - even if trading doesn't profit, you'll build a useful scam detector and gain practical ML experience.

Good luck! 🚀
