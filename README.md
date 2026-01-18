# Safe-ish: ML-Powered Meme Coin Trading System

A two-stage algorithmic trading system that uses machine learning to filter scam tokens before applying trading strategies.

## Philosophy

**Defense First, Offense Second** - Use ML as a protective shield to eliminate fraudulent tokens (85-90% target), then deploy trading strategies on the approved subset (55-65% win rate target).

## Project Status

### ✅ Week 1 Complete: Foundation
- [x] Project structure and configuration
- [x] DexScreener API integration with rate limiting
- [x] Feature collection system (30+ features)
- [x] Database schema and manager
- [x] Tested and validated with live data

### ✅ Week 2 Complete: ML Development
- [x] Historical data collector with labeling system
- [x] RugPullDetector model (binary classifier)
- [x] MultiModelDetector (ensemble for specific scam types)
- [x] Training pipeline with derived features
- [x] Comprehensive evaluation suite
- [x] CLI tools for data collection and training

### ✅ Week 3 Complete: Trading Strategies
- [x] Base strategy framework (signals, positions, exits)
- [x] Momentum breakout strategy
- [x] Dip buying strategy
- [x] Portfolio manager with P&L tracking
- [x] Position monitor with trailing stops
- [x] Risk manager with circuit breakers
- [x] Main trading bot orchestrator
- [x] All tests passing

### 🔄 Next: Week 4 - Integration & Testing
- [ ] End-to-end testing with live API data
- [ ] Backtesting framework
- [ ] Performance dashboard (Streamlit)
- [ ] Monitoring & alerts

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit config/config.yaml as needed
# Default settings are configured for safe paper trading
```

### 3. Test the System

```bash
# Test API connection and feature collection
python test_features.py
```

Expected output:
```
✓ Feature collection successful!
✓ Total ML features: 30
✓ All tests passed!
```

## Project Structure

```
safe-ish/
├── config/
│   └── config.yaml              # Trading and ML configuration
├── data/
│   ├── raw/                     # Raw data collection
│   ├── processed/               # Processed datasets
│   └── models/                  # Trained ML models
├── src/
│   ├── data_collection/
│   │   ├── dex_api.py          # DexScreener API wrapper
│   │   ├── features.py         # Feature collection system
│   │   └── collector.py        # Historical data collector (TODO)
│   ├── ml/
│   │   ├── models.py           # ML models (TODO)
│   │   ├── training.py         # Training pipeline (TODO)
│   │   └── evaluation.py       # Model evaluation (TODO)
│   ├── strategies/             # Trading strategies (TODO)
│   ├── execution/              # Portfolio management (TODO)
│   ├── bot/                    # Main trading bot (TODO)
│   └── utils/
│       ├── database.py         # SQLite database manager
│       ├── logger.py           # Logging system (TODO)
│       └── notifications.py    # Alerts system (TODO)
├── scripts/                    # CLI scripts (TODO)
├── tests/                      # Unit tests (TODO)
└── notebooks/                  # Jupyter analysis (TODO)
```

## Features Implemented

### DexScreener API Integration
- Async API wrapper with rate limiting (300 req/min)
- Token search and pair data retrieval
- Robust error handling
- Support for Solana, Ethereum, and Base chains

### Feature Collection System
- **30+ features** including:
  - Basic: token age, price, liquidity, market cap
  - Volume: 5min, 1hour, 24hour metrics
  - Transactions: buy/sell counts and ratios
  - Price dynamics: price changes across timeframes
  - Derived: volume acceleration, liquidity ratios, risk scores

### Database System
- SQLite with comprehensive schema
- Tables for trades, ML assessments, filter stats, portfolio snapshots
- Optimized indexes for fast queries
- Transaction safety with context managers

## Configuration

### Trading Settings
```yaml
trading:
  initial_capital: 1000
  max_position_size: 100
  max_positions: 5
  scan_interval_seconds: 60
```

### ML Filter
```yaml
ml_filter:
  max_risk_score: 0.40
  model_path: "data/models/rug_detector.pkl"
  retrain_interval_days: 7
```

### Risk Management
```yaml
risk_management:
  max_loss_per_trade_pct: 15
  max_drawdown_pct: 30
  stop_trading_on_drawdown: true
  time_based_exit_hours: 12
```

## Example Usage

### Collecting Token Features

```python
import asyncio
from src.data_collection.dex_api import DexScreenerAPI
from src.data_collection.features import TokenFeatures

async def main():
    async with DexScreenerAPI() as api:
        # Search for a token
        results = await api.search_pairs("BONK")
        pair = results['pairs'][0]

        # Parse data
        parsed = api.parse_token_data(pair)

        # Create features
        features = TokenFeatures(
            token_address=parsed['token_address'],
            token_symbol=parsed['token_symbol'],
            # ... (see test_features.py for full example)
        )

        # Get ML features
        ml_features = features.to_ml_features()
        print(f"Total features: {len(ml_features)}")

asyncio.run(main())
```

### Training ML Model

```bash
# Step 1: Collect training data (500+ tokens)
python scripts/collect_training_data.py --mode collect --num-tokens 500

# Step 2: Wait 24 hours, then label outcomes
python scripts/collect_training_data.py --mode label --label-hours 24

# Step 3: Train model
python scripts/train_model.py --mode binary --evaluate --report

# Step 4: Test model
python test_ml_model.py
```

### Using Trained Model

```python
from src.ml.models import RugPullDetector

# Load model
model = RugPullDetector()
model.load('data/models/rug_detector.pkl')

# Predict risk
features = {
    'token_age_hours': 12,
    'liquidity_usd': 5000,
    # ... all features
}

risk_score = model.predict_risk(features)
print(f"Risk: {risk_score:.3f}")  # 0-1, higher = more risky
print(f"Action: {model._score_to_recommendation(risk_score)}")
```

### Database Operations

```python
from src.utils.database import Database

db = Database("data/trading.db")

# Log an ML assessment
db.log_ml_assessment({
    'token_address': '0x...',
    'token_symbol': 'TOKEN',
    'chain_id': 'solana',
    'risk_score': 0.35,
    'recommendation': 'PASS'
})

# Get statistics
stats = db.get_portfolio_stats(days=7)
print(f"Win rate: {stats['win_rate']:.1%}")
```

## Implementation Roadmap

### Week 1: ✅ Foundation (COMPLETE)
- Project setup and configuration
- DexScreener API integration
- Feature collection system
- Database implementation

### Week 2: ✅ ML Development (IMPLEMENTATION COMPLETE)
- Historical data collector with labeling system
- RugPullDetector model (Gradient Boosting)
- MultiModelDetector ensemble
- Training pipeline with derived features
- Comprehensive evaluation suite
- CLI tools (collect_training_data.py, train_model.py)

### Week 3: ✅ Trading Strategies (COMPLETE)
- Base strategy framework with signals and positions
- Momentum breakout strategy (volume + momentum)
- Dip buying strategy (dip + recovery)
- Portfolio manager with P&L tracking
- Position monitor with trailing stops
- Risk manager with circuit breakers
- Main trading bot orchestrator

### Week 4: Integration & Testing
- End-to-end testing with live API
- Backtesting framework
- Performance dashboard (Streamlit)
- Monitoring & alerts

### Week 5+: Deployment
- Production deployment
- Live trading (paper first, then minimal capital)
- Performance analysis
- Continuous improvement

## Safety & Risk Management

### Hard Stops (Circuit Breakers)
1. Stop if down 30% from starting capital
2. Stop if lose 10% in a single day
3. Max 15% risk per trade
4. Max 20% capital per position
5. Close all positions after 12 hours

### Paper Trading First
- **MANDATORY:** Run paper trading for minimum 2 weeks
- Validate ML filter accuracy
- Test strategies without real money
- Fix bugs before live deployment

### Progressive Capital Deployment
1. Paper trading ($0)
2. Minimal capital ($100-200)
3. Small capital ($500) if profitable
4. Scale gradually with proven results

## Educational Goals

Even if trading breaks even, this project provides:
- Production ML system experience
- Async Python and API integration skills
- Algorithmic trading knowledge
- Risk management expertise
- Complete ML pipeline: data → model → deployment

The ML filter alone (scam detector) is valuable and could be used independently.

## Resources

- **Specification:** See `docs/meme_coin_trader_spec.md` for full technical spec
- **Quick Start:** See `docs/quick_start_guide.md` for implementation guide
- **Implementation Plan:** See `.claude/plans/` for detailed implementation plan

## Warning

⚠️ **This is an educational project.** Cryptocurrency trading is extremely risky. Only use capital you can afford to lose completely. Start with paper trading and minimal amounts.

## License

MIT License - See LICENSE file for details

## Contributing

This is a personal learning project, but suggestions and feedback are welcome via issues.

---

**Status:** Week 3 Complete | Next: Integration & Testing
**Last Updated:** 2026-01-15