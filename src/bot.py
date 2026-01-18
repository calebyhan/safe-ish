"""Main Trading Bot Orchestrator"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import yaml
from pathlib import Path

from .data_collection.dex_api import DexScreenerAPI
from .data_collection.features import TokenFeatures
from .ml.models import RugPullDetector
from .strategies.base import BaseStrategy, Signal
from .strategies.momentum import MomentumBreakoutStrategy
from .strategies.dip_buying import DipBuyingStrategy
from .trading.portfolio import PortfolioManager
from .trading.monitor import PositionMonitor, MonitorConfig
from .trading.risk import RiskManager, RiskConfig
from .utils.database import Database
from .utils.logger import setup_logging, get_logger


class TradingBot:
    """
    Main trading bot orchestrator

    Coordinates:
    - Token scanning via DexScreener API
    - ML filtering for scam detection
    - Strategy signal generation
    - Position management
    - Risk management
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        paper_trading: bool = True
    ):
        """
        Initialize trading bot

        Args:
            config_path: Path to configuration file
            paper_trading: Whether to run in paper trading mode
        """
        self.config = self._load_config(config_path)
        self.paper_trading = paper_trading

        # Initialize logging
        log_config = self.config.get('logging', {})
        self.logger = setup_logging(
            log_dir=log_config.get('log_dir', 'logs'),
            console_level=log_config.get('console_level', 'INFO'),
            file_level=log_config.get('file_level', 'DEBUG')
        )

        # Initialize components
        self.api: Optional[DexScreenerAPI] = None
        self.db = Database(self.config.get('database', {}).get('path', 'data/trading.db'))
        self.ml_model: Optional[RugPullDetector] = None

        # Trading components
        self.portfolio = PortfolioManager(
            initial_capital=self.config.get('trading', {}).get('initial_capital', 1000),
            max_positions=self.config.get('trading', {}).get('max_positions', 5),
            max_position_size_pct=self.config.get('trading', {}).get('max_position_size', 100) / 1000,
            paper_trading=paper_trading
        )

        # Initialize strategies
        self.strategies: Dict[str, BaseStrategy] = {}
        self._init_strategies()

        # Risk manager
        risk_config = RiskConfig(
            max_drawdown_pct=self.config.get('risk_management', {}).get('max_drawdown_pct', 30),
            max_daily_loss_pct=self.config.get('risk_management', {}).get('max_daily_loss_pct', 10),
            max_loss_per_trade_pct=self.config.get('risk_management', {}).get('max_loss_per_trade_pct', 15),
            max_hold_time_hours=self.config.get('risk_management', {}).get('time_based_exit_hours', 12)
        )
        self.risk_manager = RiskManager(self.portfolio, risk_config)

        # Position monitor
        monitor_config = MonitorConfig(
            update_interval_seconds=self.config.get('trading', {}).get('scan_interval_seconds', 60),
            max_hold_time_hours=self.config.get('risk_management', {}).get('time_based_exit_hours', 12)
        )
        self.monitor = PositionMonitor(
            self.portfolio,
            self.strategies,
            monitor_config,
            price_fetcher=self._fetch_token_price
        )

        # State
        self.is_running = False
        self.tokens_scanned = 0
        self.signals_generated = 0
        self.trades_executed = 0

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return {}

    def _init_strategies(self):
        """Initialize trading strategies"""
        # Momentum Breakout Strategy
        momentum_config = self.config.get('strategies', {}).get('momentum_breakout', {})
        self.strategies['momentum_breakout'] = MomentumBreakoutStrategy(
            volume_spike_multiplier=momentum_config.get('volume_spike_multiplier', 2.0),
            min_price_momentum_pct=momentum_config.get('min_price_momentum_pct', 5.0),
            min_buy_sell_ratio=momentum_config.get('min_buy_sell_ratio', 1.5),
            max_ml_risk_score=self.config.get('ml_filter', {}).get('max_risk_score', 0.40),
            time_exit_hours=momentum_config.get('time_exit_hours', 4.0)
        )

        # Dip Buying Strategy
        dip_config = self.config.get('strategies', {}).get('dip_buying', {})
        self.strategies['dip_buying'] = DipBuyingStrategy(
            min_dip_pct=dip_config.get('min_dip_pct', 20.0),
            max_dip_pct=dip_config.get('max_dip_pct', 40.0),
            min_recovery_momentum_pct=dip_config.get('min_recovery_pct', 2.0),
            max_ml_risk_score=self.config.get('ml_filter', {}).get('max_risk_score', 0.40),
            time_exit_hours=dip_config.get('time_exit_hours', 6.0)
        )

    async def _load_ml_model(self):
        """Load the ML model"""
        model_path = self.config.get('ml_filter', {}).get('model_path', 'data/models/rug_detector.pkl')

        try:
            self.ml_model = RugPullDetector()
            self.ml_model.load(model_path)
            self.logger.info(f"ML model loaded from {model_path}")
        except Exception as e:
            self.logger.warning(f"Could not load ML model: {e}")
            self.logger.warning("Running without ML filter (higher risk)")
            self.ml_model = None

    async def start(self):
        """Start the trading bot"""
        mode = "PAPER TRADING" if self.paper_trading else "LIVE TRADING"
        self.logger.info("=" * 60)
        self.logger.info(f"{mode} BOT STARTING")
        self.logger.info("=" * 60)

        # Load ML model
        await self._load_ml_model()

        # Initialize API
        self.api = DexScreenerAPI()
        await self.api.__aenter__()

        self.is_running = True

        # Start background tasks
        tasks = [
            asyncio.create_task(self._scan_loop()),
            asyncio.create_task(self.monitor.start())
        ]

        self.logger.info(f"Bot started at {datetime.utcnow()}")
        self.logger.info(f"Initial capital: ${self.portfolio.initial_capital:.2f}")
        self.logger.info(f"Strategies: {', '.join(self.strategies.keys())}")
        self.logger.info(f"ML Filter: {'Enabled' if self.ml_model else 'Disabled'}")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self):
        """Stop the trading bot"""
        self.logger.info("Stopping bot...")
        self.is_running = False
        self.monitor.stop()

        if self.api:
            await self.api.__aexit__(None, None, None)

        # Log final summary
        stats = self.portfolio.get_statistics()
        self.logger.performance_snapshot(
            capital=stats['portfolio_value'],
            pnl=stats['total_pnl'],
            roi=stats['roi_pct'],
            trades=stats['total_trades'],
            win_rate=stats['win_rate']
        )

        self.logger.info("Bot stopped.")

    async def _scan_loop(self):
        """Main scanning loop for finding trading opportunities"""
        scan_interval = self.config.get('trading', {}).get('scan_interval_seconds', 60)
        search_queries = self.config.get('scanning', {}).get('queries', ['PEPE', 'BONK', 'WIF', 'DOGE'])

        while self.is_running:
            try:
                # Check if we can trade
                can_trade, reason = self.risk_manager.check_can_trade()
                if not can_trade:
                    self.logger.warning(f"Trading paused: {reason}")
                    await asyncio.sleep(scan_interval)
                    continue

                # Scan for opportunities
                await self._scan_tokens(search_queries)

                await asyncio.sleep(scan_interval)

            except Exception as e:
                self.logger.error(f"Error in scan loop: {e}")
                await asyncio.sleep(10)

    async def _scan_tokens(self, queries: List[str]):
        """
        Scan tokens for trading opportunities

        Args:
            queries: Search queries to find tokens
        """
        for query in queries:
            if not self.is_running:
                break

            try:
                results = await self.api.search_pairs(query)

                for pair in results.get('pairs', [])[:10]:  # Check top 10 per query
                    self.tokens_scanned += 1

                    # Parse token data
                    parsed = self.api.parse_token_data(pair)

                    # Apply ML filter
                    ml_risk = await self._assess_ml_risk(parsed)
                    if ml_risk > self.config.get('ml_filter', {}).get('max_risk_score', 0.40):
                        continue

                    # Add ML risk to token data
                    parsed['ml_risk_score'] = ml_risk

                    # Check all strategies
                    for strategy in self.strategies.values():
                        signal = strategy.analyze(parsed)
                        if signal:
                            await self._process_signal(signal, ml_risk)

            except Exception as e:
                self.logger.error(f"Error scanning for '{query}': {e}")

    async def _assess_ml_risk(self, token_data: Dict) -> float:
        """
        Assess ML risk score for a token

        Args:
            token_data: Token market data

        Returns:
            Risk score 0-1 (higher = riskier)
        """
        if not self.ml_model:
            return 0.5  # Default moderate risk without model

        try:
            # Create features dict
            features = {
                'token_age_hours': token_data.get('token_age_hours', 0),
                'liquidity_usd': token_data.get('liquidity_usd', 0),
                'market_cap': token_data.get('market_cap', 0),
                'price_usd': token_data.get('price_usd', 0),
                'volume_5min': token_data.get('volume_5m', 0),
                'volume_1hour': token_data.get('volume_1h', 0),
                'volume_24hour': token_data.get('volume_24h', 0),
                'buy_count_5min': token_data.get('txns_5m_buys', 0),
                'sell_count_5min': token_data.get('txns_5m_sells', 0),
                'buy_count_1hour': token_data.get('txns_1h_buys', 0),
                'sell_count_1hour': token_data.get('txns_1h_sells', 0),
                'price_change_5min': token_data.get('price_change_5m', 0),
                'price_change_1hour': token_data.get('price_change_1h', 0),
                'price_change_24hour': token_data.get('price_change_24h', 0),
            }

            # Add derived features
            features = self._add_derived_features(features)

            # Get prediction
            risk_score = self.ml_model.predict_risk(features)
            return risk_score

        except Exception as e:
            self.logger.error(f"ML risk assessment error: {e}")
            return 0.5

    def _add_derived_features(self, features: Dict) -> Dict:
        """Add derived features for ML model"""
        # Buy/sell ratios
        sell_5m = features.get('sell_count_5min', 1) or 1
        buy_5m = features.get('buy_count_5min', 0)
        features['buy_sell_ratio_5min'] = buy_5m / sell_5m

        sell_1h = features.get('sell_count_1hour', 1) or 1
        buy_1h = features.get('buy_count_1hour', 0)
        features['buy_sell_ratio_1hour'] = buy_1h / sell_1h

        # Liquidity to market cap ratio
        mcap = features.get('market_cap', 1) or 1
        features['liquidity_to_mcap_ratio'] = features.get('liquidity_usd', 0) / mcap

        # Volume acceleration
        vol_5m = features.get('volume_5min', 0)
        vol_1h = features.get('volume_1hour', 1) or 1
        rate_5m = vol_5m / 5
        rate_1h = vol_1h / 60
        features['volume_acceleration'] = rate_5m / rate_1h if rate_1h > 0 else 0

        # Launch phase risk
        age = features.get('token_age_hours', 0)
        features['launch_phase_risk'] = max(0, 1 - (age / 24)) if age < 24 else 0

        # Safety score
        safety = 0
        if age > 168:
            safety += 1
        if features.get('liquidity_usd', 0) > 10000:
            safety += 1
        if features.get('market_cap', 0) > 100000:
            safety += 1
        if features.get('liquidity_to_mcap_ratio', 0) > 0.1:
            safety += 1
        features['safety_score'] = safety

        # Volume to liquidity ratio
        liq = features.get('liquidity_usd', 1) or 1
        features['volume_to_liquidity_ratio'] = features.get('volume_24hour', 0) / liq

        return features

    async def _process_signal(self, signal: Signal, ml_risk: float):
        """
        Process a trading signal

        Args:
            signal: Trading signal
            ml_risk: ML risk score
        """
        self.signals_generated += 1

        # Validate with risk manager
        signal_data = {
            'position_size_pct': signal.position_size_pct,
            'stop_loss_pct': (signal.entry_price - signal.stop_loss) / signal.entry_price,
            'chain_id': signal.chain_id,
            'ml_risk_score': ml_risk
        }

        is_valid, reason = self.risk_manager.validate_signal(signal_data)
        if not is_valid:
            self.logger.signal_rejected(signal.token_symbol, reason)
            return

        # Open position
        position = self.portfolio.open_position(signal, ml_risk)
        if position:
            self.trades_executed += 1

            # Log to database
            self.db.log_trade({
                'token_address': signal.token_address,
                'token_symbol': signal.token_symbol,
                'chain_id': signal.chain_id,
                'strategy': signal.strategy_name,
                'action': 'OPEN',
                'price': signal.entry_price,
                'size_usd': position.size_usd,
                'ml_risk_score': ml_risk,
                'stop_loss': signal.stop_loss,
                'take_profit_1': signal.take_profit_1,
                'take_profit_2': signal.take_profit_2,
                'confidence': signal.confidence
            })

            # Log ML assessment
            self.db.log_ml_assessment({
                'token_address': signal.token_address,
                'token_symbol': signal.token_symbol,
                'chain_id': signal.chain_id,
                'risk_score': ml_risk,
                'recommendation': 'PASS' if ml_risk < 0.40 else 'CAUTION'
            })

    async def _fetch_token_price(self, token_address: str, chain_id: str) -> Optional[Dict]:
        """
        Fetch current price for a token

        Args:
            token_address: Token address
            chain_id: Blockchain ID

        Returns:
            Token data dict or None
        """
        try:
            results = await self.api.search_pairs(token_address[:8])
            for pair in results.get('pairs', []):
                if (pair.get('baseToken', {}).get('address', '') == token_address and
                    pair.get('chainId', '') == chain_id):
                    return self.api.parse_token_data(pair)
        except Exception:
            pass
        return None

    def get_status(self) -> Dict:
        """Get bot status"""
        return {
            'is_running': self.is_running,
            'paper_trading': self.paper_trading,
            'tokens_scanned': self.tokens_scanned,
            'signals_generated': self.signals_generated,
            'trades_executed': self.trades_executed,
            'portfolio': self.portfolio.get_statistics(),
            'risk': self.risk_manager.get_risk_metrics(),
            'strategies': {name: s.get_statistics() for name, s in self.strategies.items()}
        }

    def print_status(self):
        """Print comprehensive status"""
        print("\n" + "=" * 60)
        print("TRADING BOT STATUS")
        print("=" * 60)

        status = self.get_status()
        print(f"\n{'[PAPER TRADING]' if self.paper_trading else '[LIVE TRADING]'}")
        print(f"Running: {'Yes' if status['is_running'] else 'No'}")
        print(f"ML Filter: {'Enabled' if self.ml_model else 'Disabled'}")

        print(f"\nActivity:")
        print(f"  Tokens Scanned: {status['tokens_scanned']}")
        print(f"  Signals Generated: {status['signals_generated']}")
        print(f"  Trades Executed: {status['trades_executed']}")

        self.portfolio.print_summary()
        self.risk_manager.print_status()

        print("\nStrategies:")
        for name, stats in status['strategies'].items():
            print(f"  {name}:")
            print(f"    Signals: {stats['signals_generated']}")
            print(f"    Active: {stats['active_positions']}")


async def run_bot(config_path: str = "config/config.yaml", paper_trading: bool = True):
    """Run the trading bot"""
    bot = TradingBot(config_path=config_path, paper_trading=paper_trading)

    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\nReceived shutdown signal...")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(run_bot())
