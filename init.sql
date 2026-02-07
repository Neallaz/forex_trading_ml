-- Create tables for storing trading results
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    total_return DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    win_rate DECIMAL(10, 4),
    profit_factor DECIMAL(10, 4),
    total_trades INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    backtest_id INTEGER REFERENCES backtest_results(id),
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    symbol VARCHAR(10) NOT NULL,
    direction VARCHAR(4) CHECK (direction IN ('BUY', 'SELL')),
    entry_price DECIMAL(20, 10),
    exit_price DECIMAL(20, 10),
    size DECIMAL(20, 10),
    pnl DECIMAL(20, 10),
    pnl_percent DECIMAL(10, 4),
    stop_loss DECIMAL(20, 10),
    take_profit DECIMAL(20, 10)
);

CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    accuracy DECIMAL(10, 4),
    precision DECIMAL(10, 4),
    recall DECIMAL(10, 4),
    f1_score DECIMAL(10, 4),
    auc DECIMAL(10, 4),
    training_time INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(20, 10),
    high DECIMAL(20, 10),
    low DECIMAL(20, 10),
    close DECIMAL(20, 10),
    volume DECIMAL(20, 4),
    UNIQUE(symbol, timestamp)
);

CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
CREATE INDEX idx_trades_backtest_id ON trades(backtest_id);
CREATE INDEX idx_backtest_results_created_at ON backtest_results(created_at);