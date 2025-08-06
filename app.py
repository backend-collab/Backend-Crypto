from datetime import datetime, timezone, timedelta
from functools import wraps
from collections import OrderedDict
import ccxt
import pandas as pd
import mplfinance as mpf
import io
import os
import requests
import time
import logging
from flask import Flask, jsonify, request, make_response
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
VALID_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w', '1M']
alert_history = []
current_user = 'backend-collab'  # Default user for tracking

def get_utc_time():
    """Get current UTC time in YYYY-MM-DD HH:MM:SS format"""
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

def log_request(user, endpoint):
    """Log request dengan user dan timestamp"""
    utc_time = get_utc_time()
    logger.info(f"Request at {utc_time} by {user} to {endpoint}")

def initialize_exchange():
    """Initialize exchange with proper error handling"""
    try:
        exchange = ccxt.binanceus()
        exchange.load_markets()
        return exchange
    except ccxt.NetworkError as e:
        logger.error(f"Network error: {str(e)}")
        raise Exception("Tidak dapat terhubung ke exchange. Silakan coba lagi.")
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error: {str(e)}")
        raise Exception("Error pada exchange. Mohon periksa status exchange.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise Exception("Terjadi kesalahan yang tidak diharapkan.")

class TimedCache(OrderedDict):
    """Cache implementation with time-based expiration"""
    def __init__(self, max_age_seconds=300, max_size=100):
        self.max_age = timedelta(seconds=max_age_seconds)
        self.max_size = max_size
        super().__init__()
        
    def __setitem__(self, key, value):
        if len(self) >= self.max_size:
            self.popitem(last=False)  # Remove oldest
        super().__setitem__(key, (datetime.now(), value))
        
    def __getitem__(self, key):
        timestamp, value = super().__getitem__(key)
        if datetime.now() - timestamp > self.max_age:
            del self[key]
            raise KeyError
        return value

# Initialize cache
cache_data = TimedCache()

class RateLimiter:
    """Rate limiter implementation"""
    def __init__(self, max_requests=1200, per_seconds=60):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.requests = []
        
    def can_make_request(self):
        now = datetime.now()
        self.requests = [req for req in self.requests 
                        if req > now - timedelta(seconds=self.per_seconds)]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

rate_limiter = RateLimiter()

def rate_limit_decorator(func):
    """Decorator for rate limiting"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not rate_limiter.can_make_request():
            raise Exception("Rate limit exceeded. Please try again later.")
        return func(*args, **kwargs)
    return wrapper

def retry_on_error(max_retries=3, delay=1):
    """Decorator for retry logic"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                    retries += 1
                    if retries == max_retries:
                        raise
                    logger.warning(f"Attempt {retries} failed. Retrying in {delay} seconds...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry_on_error(max_retries=3)
def fetch_ohlcv_safe(exchange, symbol, timeframe, limit):
    """Fetch OHLCV data with retry logic"""
    return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

def validate_symbol(symbol_input):
    """Validate and normalize trading symbol"""
    try:
        if not symbol_input:
            raise ValueError("Symbol tidak boleh kosong")
            
        symbol = symbol_input.upper().replace('-', '/')
        
        valid_base_currencies = ['BTC', 'ETH', 'USDT', 'USD', 'BUSD']
        
        parts = symbol.split('/')
        if len(parts) != 2:
            raise ValueError("Format symbol tidak valid. Contoh: BTC/USDT")
            
        base, quote = parts
        
        if quote not in valid_base_currencies:
            raise ValueError(f"Quote currency tidak valid. Harus salah satu dari: {', '.join(valid_base_currencies)}")
            
        return symbol
    except Exception as e:
        logger.error(f"Symbol validation error: {str(e)}")
        raise ValueError(f"Symbol validation error: {str(e)}")

def generate_chart(df, symbol, timeframe):
    """Generate candlestick chart"""
    try:
        df_chart = df.copy()
        df_chart.index = pd.to_datetime(df_chart['timestamp'], unit='ms')
        df_chart = df_chart[['open', 'high', 'low', 'close', 'volume']]
        df_chart = df_chart.astype(float)

        title = f"Chart {symbol} ({timeframe})"
        style = mpf.make_mpf_style(base_mpf_style='yahoo', gridstyle='-.')

        moving_averages = None
        if len(df_chart) >= 200:
            moving_averages = (50, 200)
        elif len(df_chart) >= 50:
            moving_averages = (20, 50)

        buffer = io.BytesIO()
        mpf.plot(
            df_chart.tail(100),
            type='candle',
            style=style,
            title=title,
            ylabel='Price (USD)',
            volume=True,
            mav=moving_averages,
            savefig=dict(fname=buffer, format='png', bbox_inches='tight'),
            returnfig=False)

        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        return io.BytesIO()

def get_onchain_data(symbol):
    """Get on-chain data for various cryptocurrencies"""
    try:
        if 'BTC' in symbol:
            btc_data = {}
            try:
                response = requests.get("https://api.blockchain.info/stats", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    btc_data.update({
                        "network_hash_rate": data.get('hash_rate', 0),
                        "difficulty": data.get('difficulty', 0),
                        "total_bitcoins": data.get('totalbc', 0) / 100000000,
                        "unconfirmed_count": data.get('n_btc_mined', 0),
                        "mempool_size": data.get('mempool_size', 0)
                    })
            except Exception as e:
                logger.error(f"Error fetching BTC data: {str(e)}")

            return btc_data if btc_data else {"error": "Gagal mengambil data Bitcoin"}

        elif 'ETH' in symbol:
            etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
            if not etherscan_api_key:
                return {"error": "ETHERSCAN_API_KEY tidak ditemukan"}

            eth_data = {}
            try:
                supply_url = f"https://api.etherscan.io/api?module=stats&action=ethsupply&apikey={etherscan_api_key}"
                response = requests.get(supply_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == '1':
                        eth_data['total_supply'] = int(data['result']) / 10**18
            except Exception as e:
                logger.error(f"Error fetching ETH supply: {str(e)}")

            return eth_data if eth_data else {"error": "Gagal mengambil data Ethereum"}

        else:
            try:
                coin_id = symbol.split('/')[0].lower()
                if coin_id == 'btc': coin_id = 'bitcoin'
                elif coin_id == 'eth': coin_id = 'ethereum'

                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "market_cap": data.get('market_data', {}).get('market_cap', {}).get('usd', 0),
                        "total_volume": data.get('market_data', {}).get('total_volume', {}).get('usd', 0),
                        "circulating_supply": data.get('market_data', {}).get('circulating_supply', 0),
                        "max_supply": data.get('market_data', {}).get('max_supply', 0)
                    }
                else:
                    return {"error": f"Data tidak tersedia untuk {symbol}"}

            except Exception as e:
                logger.error(f"Error fetching coin data: {str(e)}")
                return {"error": f"Gagal mengambil data untuk {symbol}"}

    except Exception as e:
        logger.error(f"Error in get_onchain_data: {str(e)}")
        return {"error": f"Gagal mengambil on-chain data: {str(e)}"}

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    try:
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()

        rsi = RSIIndicator(close=df['close'], window=14)
        df['RSI'] = rsi.rsi()

        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()

        macd = MACD(close=df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()

        return df
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        raise

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    levels = {
        '0.0': low,
        '0.236': low + 0.236 * diff,
        '0.382': low + 0.382 * diff,
        '0.5': low + 0.5 * diff,
        '0.618': low + 0.618 * diff,
        '0.786': low + 0.786 * diff,
        '1.0': high
    }
    return {k: round(v, 8) for k, v in levels.items()}

def calculate_pivot_points(high, low, close):
    """Calculate pivot points"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    
    return {
        'pivot': round(pivot, 8),
        'r1': round(r1, 8),
        'r2': round(r2, 8),
        's1': round(s1, 8),
        's2': round(s2, 8)
    }

@app.route('/api/analyze', methods=['GET'])
@rate_limit_decorator
def analyze_crypto():
    """Main analysis endpoint"""
    log_request(current_user, '/api/analyze')
    
    try:
        symbol = request.args.get('symbol')
        timeframe = request.args.get('timeframe', '1d')

        validated_symbol = validate_symbol(symbol)
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(f"Timeframe tidak valid. Harus salah satu dari: {', '.join(VALID_TIMEFRAMES)}")

        exchange = initialize_exchange()
        ohlcv = fetch_ohlcv_safe(exchange, validated_symbol, timeframe, 250)
        
        if not ohlcv or len(ohlcv) < 2:
            raise ValueError("Insufficient data")

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Calculate indicators
        df = calculate_technical_indicators(df)
        latest_data = df.iloc[-1]

        # Calculate Fibonacci levels
        period_high = df['high'].max()
        period_low = df['low'].min()
        fibonacci_levels = calculate_fibonacci_levels(period_high, period_low)

        # Calculate Pivot Points
        prev_day = df.iloc[-2]
        pivot_points = calculate_pivot_points(prev_day['high'],
                                           prev_day['low'],
                                           prev_day['close'])

        # Get on-chain data
        onchain_data = get_onchain_data(validated_symbol)

        response_data = {
            "symbol": validated_symbol,
            "timestamp": get_utc_time(),
            "price": latest_data['close'],
            "technical_indicators": {
                "sma_20": round(latest_data['SMA_20'], 8),
                "sma_50": round(latest_data['SMA_50'], 8),
                "sma_200": round(latest_data['SMA_200'], 8),
                "rsi": round(latest_data['RSI'], 2),
                "macd": round(latest_data['MACD'], 8),
                "macd_signal": round(latest_data['MACD_signal'], 8),
                "bb_upper": round(latest_data['BB_upper'], 8),
                "bb_lower": round(latest_data['BB_lower'], 8)
            },
            "fibonacci_levels": fibonacci_levels,
            "pivot_points": pivot_points,
            "onchain_data": onchain_data
        }

        return jsonify(response_data)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except ccxt.NetworkError as ne:
        return jsonify({"error": "Network error"}), 503
    except Exception as e:
        logger.error(f"Unexpected error in analyze_crypto: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/realtime/<path:symbol>')
@rate_limit_decorator
def get_realtime_data(symbol):
    """Endpoint for real-time data"""
    try:
        validated_symbol = validate_symbol(symbol)
        exchange = initialize_exchange()
        ticker = exchange.fetch_ticker(validated_symbol)

        try:
            order_book = exchange.fetch_order_book(validated_symbol, limit=10)
            bid_price = order_book['bids'][0][0] if order_book.get('bids') else None
            ask_price = order_book['asks'][0][0] if order_book.get('asks') else None
        except Exception as e:
            logger.error(f"Error fetching order book: {str(e)}")
            bid_price = None
            ask_price = None

        return jsonify({
            "symbol": validated_symbol,
            "timestamp": get_utc_time(),
            "price": ticker.get('last', 0),
            "change_24h": ticker.get('percentage', 0),
            "volume_24h": ticker.get('quoteVolume', 0),
            "bid": bid_price,
            "ask": ask_price
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in get_realtime_data: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/fibonacci/<path:symbol>')
@rate_limit_decorator
def get_fibonacci_levels(symbol):
    """Endpoint for Fibonacci levels"""
    try:
        validated_symbol = validate_symbol(symbol)
        exchange = initialize_exchange()
        ohlcv = fetch_ohlcv_safe(exchange, validated_symbol, '1d', 50)

        if not ohlcv or len(ohlcv) < 10:
            return jsonify({"error": "Insufficient data"}), 400

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        period_high = df['high'].max()
        period_low = df['low'].min()
        current_price = df.iloc[-1]['close']

        fib_levels = calculate_fibonacci_levels(period_high, period_low)
        price_distances = {level: abs(current_price - price)
                         for level, price in fib_levels.items()}
        nearest_level = min(price_distances, key=price_distances.get)

        return jsonify({
            "symbol": validated_symbol,
            "timestamp": get_utc_time(),
            "current_price": current_price,
            "fibonacci_levels": fib_levels,
            "nearest_level": nearest_level,
            "nearest_price": fib_levels[nearest_level]
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in get_fibonacci_levels: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/')
def home():
    """Home page"""
    response = make_response("""
    <h1>🚀 Advanced Crypto Trading API</h1>
    <h2>Fitur Terbaru:</h2>
    <ul>
        <li>✅ Real-time Price & Volume Analysis</li>
        <li>✅ Technical Indicators (SMA, RSI, MACD, Bollinger Bands)</li>
        <li>✅ Fibonacci Retracement Levels</li>
        <li>✅ Pivot Points & Support/Resistance</li>
        <li>✅ On-Chain Data Integration</li>
    </ul>
    <h2>Endpoints:</h2>
    <ul>
        <li><code>/api/analyze?symbol=BTC/USDT&timeframe=1d</code> - Analisis lengkap</li>
        <li><code>/api/realtime/BTC/USDT</code> - Data real-time</li>
        <li><code>/api/fibonacci/BTC/USDT</code> - Level Fibonacci</li>
    </ul>
    <h3>Test Links:</h3>
    <ul>
        <li><a href="/api/analyze?symbol=BTC/USDT&timeframe=1d">Test Analyze BTC/USDT</a></li>
        <li><a href="/api/realtime/BTC/USDT">Test Realtime BTC/USDT</a></li>
        <li><a href="/api/fibonacci/BTC/USDT">Test Fibonacci BTC/USDT</a></li>
    </ul>
    """)
    
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

@app.after_request
def add_header(response):
    """Add security and cache headers to all responses"""
    if request.path.startswith('/api/'):
        response.cache_control.max_age = 300  # 5 minutes
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    return response

if __name__ == "__main__":
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
