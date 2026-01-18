"""Logging utilities for the trading bot"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class TradingLogger:
    """
    Centralized logging for the trading system

    Provides:
    - File and console logging
    - Different log levels for different components
    - Structured log formatting
    - Automatic log rotation
    """

    def __init__(
        self,
        name: str = "safe-ish",
        log_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        """
        Initialize logger

        Args:
            name: Logger name
            log_dir: Directory for log files
            console_level: Minimum level for console output
            file_level: Minimum level for file output
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            '%(levelname)-8s | %(name)s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (daily rotation)
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def debug(self, msg: str, **kwargs):
        """Log debug message"""
        self.logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs):
        """Log info message"""
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        """Log warning message"""
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs):
        """Log error message"""
        self.logger.error(msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        """Log critical message"""
        self.logger.critical(msg, **kwargs)

    def trade_opened(self, symbol: str, strategy: str, size: float, price: float, **kwargs):
        """Log trade opening"""
        self.info(
            f"TRADE OPENED | {symbol} | {strategy} | Size: ${size:.2f} | Entry: ${price:.6f}",
            extra=kwargs
        )

    def trade_closed(self, symbol: str, pnl: float, pnl_pct: float, reason: str, **kwargs):
        """Log trade closing"""
        emoji = "🟢" if pnl > 0 else "🔴"
        self.info(
            f"TRADE CLOSED {emoji} | {symbol} | P&L: ${pnl:.2f} ({pnl_pct:+.1f}%) | {reason}",
            extra=kwargs
        )

    def signal_generated(self, symbol: str, strategy: str, confidence: float, **kwargs):
        """Log signal generation"""
        self.info(
            f"SIGNAL | {symbol} | {strategy} | Confidence: {confidence:.2%}",
            extra=kwargs
        )

    def signal_rejected(self, symbol: str, reason: str, **kwargs):
        """Log signal rejection"""
        self.warning(
            f"SIGNAL REJECTED | {symbol} | {reason}",
            extra=kwargs
        )

    def ml_assessment(self, symbol: str, risk_score: float, recommendation: str, **kwargs):
        """Log ML risk assessment"""
        self.debug(
            f"ML FILTER | {symbol} | Risk: {risk_score:.3f} | {recommendation}",
            extra=kwargs
        )

    def circuit_breaker(self, state: str, reason: str, **kwargs):
        """Log circuit breaker events"""
        self.critical(
            f"CIRCUIT BREAKER | State: {state} | Reason: {reason}",
            extra=kwargs
        )

    def performance_snapshot(self, capital: float, pnl: float, roi: float, trades: int, win_rate: float, **kwargs):
        """Log performance snapshot"""
        self.info(
            f"PERFORMANCE | Capital: ${capital:.2f} | P&L: ${pnl:+.2f} ({roi:+.1f}%) | "
            f"Trades: {trades} | Win Rate: {win_rate:.1%}",
            extra=kwargs
        )


class ComponentLogger:
    """Logger for specific components with prefixed names"""

    def __init__(self, component_name: str, base_logger: Optional[TradingLogger] = None):
        """
        Initialize component logger

        Args:
            component_name: Name of the component (e.g., 'DexAPI', 'Strategy.Momentum')
            base_logger: Base TradingLogger instance (creates new if None)
        """
        if base_logger is None:
            base_logger = TradingLogger()

        self.base_logger = base_logger
        self.component_name = component_name
        self.logger = logging.getLogger(f"{base_logger.name}.{component_name}")

    def debug(self, msg: str, **kwargs):
        """Log debug message with component prefix"""
        self.logger.debug(f"[{self.component_name}] {msg}", **kwargs)

    def info(self, msg: str, **kwargs):
        """Log info message with component prefix"""
        self.logger.info(f"[{self.component_name}] {msg}", **kwargs)

    def warning(self, msg: str, **kwargs):
        """Log warning message with component prefix"""
        self.logger.warning(f"[{self.component_name}] {msg}", **kwargs)

    def error(self, msg: str, **kwargs):
        """Log error message with component prefix"""
        self.logger.error(f"[{self.component_name}] {msg}", **kwargs)

    def critical(self, msg: str, **kwargs):
        """Log critical message with component prefix"""
        self.logger.critical(f"[{self.component_name}] {msg}", **kwargs)


# Global logger instance
_global_logger: Optional[TradingLogger] = None


def get_logger(component_name: Optional[str] = None) -> TradingLogger | ComponentLogger:
    """
    Get logger instance

    Args:
        component_name: Optional component name for ComponentLogger

    Returns:
        TradingLogger or ComponentLogger instance
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = TradingLogger()

    if component_name:
        return ComponentLogger(component_name, _global_logger)

    return _global_logger


def setup_logging(
    log_dir: str = "logs",
    console_level: str = "INFO",
    file_level: str = "DEBUG"
) -> TradingLogger:
    """
    Setup global logging configuration

    Args:
        log_dir: Directory for log files
        console_level: Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_level: File log level

    Returns:
        TradingLogger instance
    """
    global _global_logger

    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    console_level_int = level_map.get(console_level.upper(), logging.INFO)
    file_level_int = level_map.get(file_level.upper(), logging.DEBUG)

    _global_logger = TradingLogger(
        log_dir=log_dir,
        console_level=console_level_int,
        file_level=file_level_int
    )

    return _global_logger


# Example usage
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()

    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test trading-specific methods
    logger.trade_opened("BONK", "momentum_breakout", 100.0, 0.000012)
    logger.trade_closed("BONK", 15.5, 15.5, "TAKE_PROFIT_1")
    logger.signal_generated("PEPE", "dip_buying", 0.85)
    logger.ml_assessment("WIF", 0.25, "PASS")
    logger.performance_snapshot(1050.0, 50.0, 5.0, 10, 0.60)

    # Test component logger
    api_logger = get_logger("DexAPI")
    api_logger.info("Fetching token data...")
    api_logger.debug("Rate limit check passed")

    strategy_logger = get_logger("Strategy.Momentum")
    strategy_logger.info("Analyzing token for entry signal")
    strategy_logger.warning("Low liquidity detected")
