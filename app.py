from flask import Flask, request, jsonify, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import ccxt
import pandas as pd
import pandas_ta as ta
import os
import requests
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import threading
import json
import mplfinance as mpf
import io
import logging
import matplotlib
from cors_handler import add_cors_headers
from functools import wraps
from collections import OrderedDict

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
# Di bagian atas app.py, setelah import
if os.environ.get('HF_SPACE'):
    # Konfigurasi khusus untuk HF Space
    matplotlib.use('Agg')
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables untuk caching dan real-time updates
cache_data = {}
alert_history = []
current_user = 'backend-collab'  # Untuk tracking user

def log_request(user, endpoint):
    """Log request dengan user dan timestamp"""
    utc_time = get_utc_time()
    logger.info(f"Request at {utc_time} by {user} to {endpoint}")

def get_utc_time():
    """Get current UTC time in YYYY-MM-DD HH:MM:SS format"""
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

def initialize_exchange():
    try:
        exchange = ccxt.binanceus()
        # Test connection
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

def retry_on_error(max_retries=3, delay=1):
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
    """
    Fetch OHLCV data dengan retry logic
    """
    return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)


class TimedCache(OrderedDict):
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

# Ganti cache global
cache_data = TimedCache()

class RateLimiter:
    def __init__(self, max_requests=1200, per_seconds=60):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.requests = []
        
    def can_make_request(self):
        now = datetime.now()
        # Remove old requests
        self.requests = [req for req in self.requests 
                        if req > now - timedelta(seconds=self.per_seconds)]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

rate_limiter = RateLimiter()

def rate_limit_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not rate_limiter.can_make_request():
            raise Exception("Rate limit exceeded. Please try again later.")
        return func(*args, **kwargs)
    return wrapper

def generate_chart(df, symbol, timeframe):
    """
    Membuat gambar grafik candlestick menggunakan mplfinance dan menyimpannya di memori.
    """
    try:
        # Pastikan kolom yang dibutuhkan ada dan tipenya benar
        df_chart = df.copy()
        df_chart.index = pd.to_datetime(df_chart['timestamp'], unit='ms')
        df_chart = df_chart[['open', 'high', 'low', 'close', 'volume']]
        df_chart = df_chart.astype(float)

        # Konfigurasi plot
        title = f"Chart {symbol} ({timeframe})"
        style = mpf.make_mpf_style(base_mpf_style='yahoo', gridstyle='-.')

        # Tambahkan Moving Averages ke plot (hanya jika data cukup)
        moving_averages = None
        if len(df_chart) >= 200:
            moving_averages = (50, 200)
        elif len(df_chart) >= 50:
            moving_averages = (20, 50)

        # Buat plot
        buffer = io.BytesIO()
        mpf.plot(
            df_chart.tail(100),  # Ambil 100 candle terakhir untuk digambar
            type='candle',
            style=style,
            title=title,
            ylabel='Price (USD)',
            volume=True,
            mav=moving_averages,
            savefig=dict(fname=buffer, format='png', bbox_inches='tight'),
            returnfig=False)

        buffer.seek(0)  # Pindahkan pointer ke awal buffer
        return buffer
    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        # Return empty buffer if chart generation fails
        buffer = io.BytesIO()
        return buffer


def validate_symbol(symbol_input):
    """
    Validasi dan normalisasi simbol trading
    """
    try:
        if not symbol_input:
            raise ValueError("Symbol tidak boleh kosong")
            
        symbol = symbol_input.upper().replace('-', '/')
        
        # Daftar base currency yang valid di Binance US
        valid_base_currencies = ['BTC', 'ETH', 'USDT', 'USD', 'BUSD']
        
        # Cek format symbol (e.g., BTC/USDT)
        parts = symbol.split('/')
        if len(parts) != 2:
            raise ValueError("Format symbol tidak valid. Contoh: BTC/USDT")
            
        base, quote = parts
        
        # Validasi quote currency
        if quote not in valid_base_currencies:
            raise ValueError(f"Quote currency tidak valid. Harus salah satu dari: {', '.join(valid_base_currencies)}")
            
        return symbol
    except Exception as e:
        logger.error(f"Symbol validation error: {str(e)}")
        raise ValueError(f"Symbol validation error: {str(e)}")

VALID_TIMEFRAMES = [
    '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d',
    '3d', '1w', '1M'
]


def calculate_fibonacci_levels(high, low):
    """Hitung level Fibonacci retracement"""
    diff = high - low
    levels = {
        'level_0': high,
        'level_23.6': high - (diff * 0.236),
        'level_38.2': high - (diff * 0.382),
        'level_50': high - (diff * 0.5),
        'level_61.8': high - (diff * 0.618),
        'level_78.6': high - (diff * 0.786),
        'level_100': low
    }
    return levels


def calculate_pivot_points(high, low, close):
    """Hitung pivot points dan support/resistance levels"""
    pivot = (high + low + close) / 3

    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)

    return {
        'pivot': pivot,
        'resistance_1': r1,
        'resistance_2': r2,
        'resistance_3': r3,
        'support_1': s1,
        'support_2': s2,
        'support_3': s3
    }


def get_onchain_data(symbol):
    """Ambil data on-chain dari API blockchain explorer"""
    try:
        # Untuk Bitcoin
        if 'BTC' in symbol:
            # Menggunakan multiple endpoints untuk data Bitcoin yang lebih lengkap
            btc_data = {}

            # 1. Data dari Blockchain.info
            try:
                url = "https://api.blockchain.info/stats"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    btc_data.update({
                        "network_hash_rate":
                        data.get('hash_rate', 0),
                        "difficulty":
                        data.get('difficulty', 0),
                        "total_bitcoins":
                        data.get('totalbc', 0) / 100000000,
                        "unconfirmed_count":
                        data.get('n_btc_mined', 0),
                        "mempool_size":
                        data.get('mempool_size', 0)
                    })
            except:
                pass

            # 2. Data dari Mempool.space (untuk informasi mempool yang lebih akurat)
            try:
                mempool_url = "https://mempool.space/api/mempool"
                mempool_response = requests.get(mempool_url, timeout=10)
                if mempool_response.status_code == 200:
                    mempool_data = mempool_response.json()
                    btc_data.update({
                        "mempool_transactions":
                        mempool_data.get('count', 0),
                        "mempool_size_bytes":
                        mempool_data.get('vsize', 0),
                        "mempool_fees":
                        mempool_data.get('total_fee', 0)
                    })
            except:
                pass

            # 3. Data network dari Mempool.space
            try:
                network_url = "https://mempool.space/api/v1/difficulty-adjustment"
                network_response = requests.get(network_url, timeout=10)
                if network_response.status_code == 200:
                    network_data = network_response.json()
                    btc_data.update({
                        "difficulty_change":
                        network_data.get('difficultyChange', 0),
                        "estimated_retarget_date":
                        network_data.get('estimatedRetargetDate', 0),
                        "blocks_until_retarget":
                        network_data.get('remainingBlocks', 0)
                    })
            except:
                pass

            return btc_data if btc_data else {
                "error": "Gagal mengambil data Bitcoin"
            }

        # Untuk Ethereum menggunakan Etherscan API
        elif 'ETH' in symbol:
            etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
            if not etherscan_api_key:
                return {
                    "error": "ETHERSCAN_API_KEY tidak ditemukan di secrets"
                }

            eth_data = {}

            # 1. ETH Total Supply
            try:
                supply_url = f"https://api.etherscan.io/api?module=stats&action=ethsupply&apikey={etherscan_api_key}"
                supply_response = requests.get(supply_url, timeout=10)
                if supply_response.status_code == 200:
                    supply_data = supply_response.json()
                    if supply_data['status'] == '1':
                        eth_data['total_supply'] = int(
                            supply_data['result']) / 10**18
            except:
                pass

            # 2. Gas Price
            try:
                gas_url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={etherscan_api_key}"
                gas_response = requests.get(gas_url, timeout=10)
                if gas_response.status_code == 200:
                    gas_data = gas_response.json()
                    if gas_data['status'] == '1':
                        eth_data.update({
                            'safe_gas_price':
                            gas_data['result']['SafeGasPrice'],
                            'standard_gas_price':
                            gas_data['result']['StandardGasPrice'],
                            'fast_gas_price':
                            gas_data['result']['FastGasPrice']
                        })
            except:
                pass

            # 3. Latest Block Number
            try:
                block_url = f"https://api.etherscan.io/api?module=proxy&action=eth_blockNumber&apikey={etherscan_api_key}"
                block_response = requests.get(block_url, timeout=10)
                if block_response.status_code == 200:
                    block_data = block_response.json()
                    if 'result' in block_data:
                        eth_data['latest_block'] = int(block_data['result'],
                                                       16)
            except:
                pass

            # 4. Node Count dari Ethernodes.org
            try:
                nodes_url = "https://www.ethernodes.org/api/nodes"
                nodes_response = requests.get(nodes_url, timeout=10)
                if nodes_response.status_code == 200:
                    nodes_data = nodes_response.json()
                    eth_data['total_nodes'] = nodes_data.get('total', 0)
            except:
                pass

            return eth_data if eth_data else {
                "error": "Gagal mengambil data Ethereum"
            }

        # Untuk cryptocurrency lainnya, gunakan CoinGecko API untuk data yang tersedia
        else:
            try:
                # Ambil market data dari CoinGecko
                coin_id = symbol.split('/')[0].lower()  # Ambil base currency
                if coin_id == 'btc': coin_id = 'bitcoin'
                elif coin_id == 'eth': coin_id = 'ethereum'

                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "market_cap":
                        data.get('market_data', {}).get('market_cap',
                                                        {}).get('usd', 0),
                        "total_volume":
                        data.get('market_data', {}).get('total_volume',
                                                        {}).get('usd', 0),
                        "circulating_supply":
                        data.get('market_data',
                                 {}).get('circulating_supply', 0),
                        "max_supply":
                        data.get('market_data', {}).get('max_supply', 0),
                        "developer_score":
                        data.get('developer_data', {}).get('stars', 0),
                        "community_score":
                        data.get('community_data',
                                 {}).get('twitter_followers', 0)
                    }
                else:
                    return {"error": f"Data tidak tersedia untuk {symbol}"}

            except Exception as e:
                return {
                    "error": f"Gagal mengambil data untuk {symbol}: {str(e)}"
                }

    except Exception as e:
        return {"error": f"Gagal mengambil on-chain data: {str(e)}"}


def detect_candlestick_patterns(df):
    """Deteksi pola candlestick penting"""
    patterns = []

    try:
        # Pastikan data cukup untuk analisis pattern
        if len(df) < 3:
            return patterns

        # Analisis manual untuk pola sederhana
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        # Doji pattern - open hampir sama dengan close
        body_size = abs(latest['close'] - latest['open'])
        total_range = latest['high'] - latest['low']

        if total_range > 0 and body_size / total_range < 0.1:
            patterns.append("Doji - Indecision pattern")

        # Hammer pattern - small body, long lower shadow
        lower_shadow = latest['open'] - latest['low'] if latest[
            'open'] < latest['close'] else latest['close'] - latest['low']
        upper_shadow = latest['high'] - max(latest['open'], latest['close'])

        if total_range > 0 and lower_shadow > 2 * body_size and upper_shadow < body_size:
            patterns.append("Hammer - Bullish reversal")

        # Engulfing pattern - current candle body engulfs previous
        if len(df) > 1:
            curr_body_high = max(latest['open'], latest['close'])
            curr_body_low = min(latest['open'], latest['close'])
            prev_body_high = max(prev['open'], prev['close'])
            prev_body_low = min(prev['open'], prev['close'])

            # Bullish engulfing
            if (latest['close'] > latest['open']
                    and prev['close'] < prev['open']
                    and curr_body_low < prev_body_low
                    and curr_body_high > prev_body_high):
                patterns.append("Bullish Engulfing - Strong bullish signal")

            # Bearish engulfing
            elif (latest['close'] < latest['open']
                  and prev['close'] > prev['open']
                  and curr_body_low < prev_body_low
                  and curr_body_high > prev_body_high):
                patterns.append("Bearish Engulfing - Strong bearish signal")

    except Exception as e:
        print(f"DEBUG: Error in pattern detection: {e}")

    return patterns


def check_macd_crossover(df):
    """Cek crossover MACD dan generate alert"""
    if len(df) < 2:
        return None

    current_macd = df.iloc[-1].get('MACD_12_26_9', 0)
    current_signal = df.iloc[-1].get('MACDs_12_26_9', 0)
    prev_macd = df.iloc[-2].get('MACD_12_26_9', 0)
    prev_signal = df.iloc[-2].get('MACDs_12_26_9', 0)

    # Bullish crossover: MACD crosses above signal
    if prev_macd <= prev_signal and current_macd > current_signal:
        return {
            "type": "MACD_BULLISH_CROSSOVER",
            "message": "🟢 MACD Bullish Crossover - Sinyal Beli Potensial",
            "timestamp": datetime.now().isoformat()
        }

    # Bearish crossover: MACD crosses below signal
    elif prev_macd >= prev_signal and current_macd < current_signal:
        return {
            "type": "MACD_BEARISH_CROSSOVER",
            "message": "🔴 MACD Bearish Crossover - Sinyal Jual Potensial",
            "timestamp": datetime.now().isoformat()
        }

    return None


def get_realtime_volume_analysis(symbol, timeframe='1m'):
    """Analisis volume real-time"""
    try:
        exchange = ccxt.binanceus() # <-- DIGANTI
        # Ambil data volume 24h dan bandingkan dengan average
        ticker = exchange.fetch_ticker(symbol)
        volume_24h = ticker.get('quoteVolume', 0)

        # Ambil data historis untuk perbandingan
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=7)
        if ohlcv:
            df_vol = pd.DataFrame(ohlcv,
                                  columns=[
                                      'timestamp', 'open', 'high', 'low',
                                      'close', 'volume'
                                  ])
            avg_volume = df_vol['volume'].mean()
            volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1

            return {
                "current_24h_volume":
                volume_24h,
                "average_7d_volume":
                avg_volume,
                "volume_ratio":
                round(volume_ratio, 2),
                "volume_status":
                "High" if volume_ratio > 1.5 else
                "Normal" if volume_ratio > 0.7 else "Low"
            }
    except Exception as e:
        return {"error": f"Gagal mengambil analisis volume: {str(e)}"}

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": "operational",
        "timestamp": get_utc_time(),
        "user": current_user,
        "version": "1.0.0"
    })

# Tambahkan decorator ke route yang perlu CORS
@app.route('/api/analyze', methods=['GET'])
@add_cors_headers
@rate_limit_decorator
def analyze_crypto():
    # ... kode yang sudah ada ...
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe', '1d')

    try:
        # Validasi input
        validated_symbol = validate_symbol(symbol)
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(f"Timeframe tidak valid. Harus salah satu dari: {', '.join(VALID_TIMEFRAMES)}")

        # Initialize exchange dengan error handling
        exchange = initialize_exchange()
        
        # Fetch data dengan retry logic
        ohlcv = fetch_ohlcv_safe(exchange, validated_symbol, timeframe, 250)
        if not ohlcv:
            raise ValueError("Tidak dapat mengambil data dari exchange")

    try:
        exchange = ccxt.binanceus() # <-- DIGANTI

        # --- 1. AMBIL DATA TEKNIKAL (OHLCV) ---
        ohlcv = exchange.fetch_ohlcv(validated_symbol, timeframe, limit=250)
        if not ohlcv or len(ohlcv) < 200:
            return jsonify(
                {"error": f"Data teknikal tidak cukup untuk {timeframe}"}), 404

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Hitung semua indikator
        df.ta.rsi(append=True)
        df.ta.macd(append=True)
        df.ta.bbands(length=20, append=True)
        df.ta.stoch(append=True)
        df.ta.adx(append=True)
        df.ta.ichimoku(append=True)
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)

        latest_data = df.iloc[-1]

        # --- 2. FIBONACCI LEVELS ---
        period_high = df['high'].tail(50).max()
        period_low = df['low'].tail(50).min()
        fibonacci_levels = calculate_fibonacci_levels(period_high, period_low)

        # --- 3. PIVOT POINTS ---
        prev_day = df.iloc[-2]  # Data hari sebelumnya
        pivot_points = calculate_pivot_points(prev_day['high'],
                                              prev_day['low'],
                                              prev_day['close'])

        # --- 4. REAL-TIME VOLUME ANALYSIS ---
        volume_analysis = get_realtime_volume_analysis(validated_symbol)

        # --- 5. ORDER BOOK DATA ---
        order_book_data = {
            "bid_volume": None,
            "ask_volume": None,
            "ratio": None
        }
        try:
            order_book = exchange.fetch_order_book(validated_symbol, limit=100)
            bids = order_book['bids'][:20]  # Top 20 bids
            asks = order_book['asks'][:20]  # Top 20 asks

            bid_volume = sum([price * amount for price, amount in bids])
            ask_volume = sum([price * amount for price, amount in asks])
            ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')

            order_book_data = {
                "bid_volume":
                round(bid_volume, 2),
                "ask_volume":
                round(ask_volume, 2),
                "ratio":
                round(ratio, 2),
                "market_pressure":
                "Bullish"
                if ratio > 1.2 else "Bearish" if ratio < 0.8 else "Neutral"
            }
        except Exception as e:
            print(f"DEBUG: Gagal mengambil order book: {e}")

        # --- 6. FEAR & GREED INDEX ---
        fear_greed_data = {"value": None, "classification": "N/A"}
        try:
            fng_response = requests.get("https://api.alternative.me/fng/",
                                        timeout=10)
            fng_response.raise_for_status()
            fng_json = fng_response.json()
            fear_greed_data = {
                "value": fng_json['data'][0]['value'],
                "classification": fng_json['data'][0]['value_classification']
            }
        except Exception as e:
            print(f"DEBUG: Gagal mengambil Fear & Greed Index: {e}")

        # --- 7. ON-CHAIN DATA ---
        onchain_data = get_onchain_data(validated_symbol)

        # --- 8. CANDLESTICK PATTERNS ---
        try:
            candlestick_patterns = detect_candlestick_patterns(df.copy())
        except Exception as e:
            print(f"DEBUG: Error detecting candlestick patterns: {e}")
            candlestick_patterns = []

        # --- 9. MACD CROSSOVER ALERT ---
        try:
            macd_alert = check_macd_crossover(df)
            if macd_alert:
                alert_history.append(macd_alert)
                # Keep only last 50 alerts
                if len(alert_history) > 50:
                    alert_history.pop(0)
        except Exception as e:
            print(f"DEBUG: Error checking MACD crossover: {e}")
            macd_alert = None

        # --- 10. TECHNICAL ANALYSIS ---
        def get_indicator_value(indicator_name):
            if indicator_name in latest_data and pd.notna(
                    latest_data[indicator_name]):
                return round(latest_data[indicator_name], 2)
            return None

        price = latest_data['close']
        rsi_val = get_indicator_value('RSI_14')
        macd_line = get_indicator_value('MACD_12_26_9')
        signal_line = get_indicator_value('MACDs_12_26_9')
        sma50 = get_indicator_value('SMA_50')
        sma200 = get_indicator_value('SMA_200')

        # Sinyal trading
        rsi_signal = "Netral"
        if rsi_val:
            if rsi_val > 70: rsi_signal = "Overbought - Pertimbangkan Jual"
            elif rsi_val < 30: rsi_signal = "Oversold - Pertimbangkan Beli"

        trend_signal = "Netral / Sideways"
        if price and sma50 and sma200:
            if price > sma50 and sma50 > sma200: trend_signal = "Uptrend Kuat"
            elif price < sma50 and sma50 < sma200:
                trend_signal = "Downtrend Kuat"

        result = {
            "symbol":
            validated_symbol,
            "timeframe":
            timeframe,
            "close_price":
            price,
            "technical_indicators": {
                "rsi": rsi_val,
                "macd_line": macd_line,
                "macd_signal": signal_line,
                "sma50": sma50,
                "sma200": sma200,
                "bb_upper": get_indicator_value('BBU_20_2.0'),
                "bb_middle": get_indicator_value('BBM_20_2.0'),
                "bb_lower": get_indicator_value('BBL_20_2.0'),
            },
            "fibonacci_levels":
            fibonacci_levels,
            "pivot_points":
            pivot_points,
            "signals": {
                "trend_signal": trend_signal,
                "rsi_signal": rsi_signal,
                "macd_crossover": macd_alert,
                "candlestick_patterns": candlestick_patterns
            },
            "market_sentiment": {
                "order_book": order_book_data,
                "volume_analysis": volume_analysis,
                "fear_and_greed": fear_greed_data
            },
            "onchain_data":
            onchain_data,
            "alerts": {
                "latest_macd_alert": macd_alert,
                "recent_alerts": alert_history[-5:] if alert_history else []
            },
            "timestamp":
            pd.to_datetime(latest_data['timestamp'], unit='ms').isoformat(),
            "last_updated":
            datetime.now().isoformat()
        }

        # Cache data untuk auto-update
        cache_data[validated_symbol] = result

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in analyze_crypto: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/chart/<path:symbol>/<timeframe>')
@app.route('/api/chart/<symbol>/<timeframe>')
def get_chart(symbol, timeframe):
    """
    Endpoint yang dioptimalkan untuk menghasilkan dan mengirim gambar grafik.
    """
    validated_symbol = symbol.upper().replace('-', '/')
    logger.info(
        f"Menerima permintaan grafik untuk {validated_symbol} ({timeframe})")

    try:
        exchange = ccxt.binanceus() # <-- DIGANTI
        # --- OPTIMISASI: Ambil hanya data yang cukup untuk plot (150 candle) ---
        # Ini jauh lebih ringan daripada mengambil 250 candle.
        ohlcv = exchange.fetch_ohlcv(validated_symbol, timeframe, limit=150)

        if not ohlcv or len(ohlcv) < 50:  # Butuh minimal 50 untuk SMA 50
            logger.warning("Data tidak cukup untuk membuat grafik.")
            return jsonify({"error":
                            "Data tidak cukup untuk membuat grafik"}), 404

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        logger.info("Memulai pembuatan gambar grafik...")
        chart_buffer = generate_chart(df, validated_symbol, timeframe)
        logger.info("Grafik berhasil dibuat, mengirim file gambar...")

        return send_file(chart_buffer, mimetype='image/png')

    except Exception as e:
        # --- PENANGANAN ERROR YANG LEBIH BAIK ---
        # Jika terjadi error, kita catat dan kirim pesan JSON, bukan membuat server crash.
        logger.error(
            f"Error saat membuat grafik untuk {validated_symbol}: {e}",
            exc_info=True)
        return jsonify({"error": f"Gagal membuat grafik: {str(e)}"}), 500


@app.route('/api/alerts/<path:symbol>')
def get_alerts(symbol):
    """Endpoint khusus untuk mendapatkan alert terbaru"""
    try:
        validated_symbol = validate_symbol(symbol)
        recent_alerts = [
            alert for alert in alert_history if validated_symbol in str(alert)
        ]
        return jsonify({
            "symbol": validated_symbol,
            "alerts": recent_alerts[-10:],  # 10 alert terbaru
            "total_alerts": len(recent_alerts)
        })
    except Exception as e:
        return jsonify({"error": f"Error getting alerts: {str(e)}"}), 500


@app.route('/api/realtime/<path:symbol>')
def get_realtime_data(symbol):
    """Endpoint untuk data real-time singkat"""
    try:
        validated_symbol = validate_symbol(symbol)
        exchange = ccxt.binanceus() # <-- DIGANTI
        ticker = exchange.fetch_ticker(validated_symbol)

        # Safely get order book
        try:
            order_book = exchange.fetch_order_book(validated_symbol, limit=10)
            bid_price = order_book['bids'][0][0] if order_book.get(
                'bids') else None
            ask_price = order_book['asks'][0][0] if order_book.get(
                'asks') else None
        except:
            bid_price = None
            ask_price = None

        return jsonify({
            "symbol": validated_symbol,
            "price": ticker.get('last', 0),
            "change_24h": ticker.get('percentage', 0),
            "volume_24h": ticker.get('quoteVolume', 0),
            "bid": bid_price,
            "ask": ask_price,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error":
                        f"Error getting realtime data: {str(e)}"}), 500


@app.route('/api/fibonacci/<path:symbol>')
def get_fibonacci_only(symbol):
    """Endpoint khusus untuk level Fibonacci"""
    try:
        validated_symbol = validate_symbol(symbol)
        exchange = ccxt.binanceus() # <-- DIGANTI
        ohlcv = exchange.fetch_ohlcv(validated_symbol, '1d', limit=50)

        if not ohlcv or len(ohlcv) < 10:
            return jsonify(
                {"error": "Insufficient data for Fibonacci calculation"}), 400

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        period_high = df['high'].max()
        period_low = df['low'].min()
        current_price = df.iloc[-1]['close']

        fib_levels = calculate_fibonacci_levels(period_high, period_low)

        # Tentukan level terdekat
        price_distances = {
            level: abs(current_price - price)
            for level, price in fib_levels.items()
        }
        nearest_level = min(price_distances, key=price_distances.get)

        return jsonify({
            "symbol": validated_symbol,
            "current_price": current_price,
            "fibonacci_levels": fib_levels,
            "nearest_level": nearest_level,
            "nearest_price": fib_levels[nearest_level],
            "period_high": period_high,
            "period_low": period_low
        })
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except ccxt.NetworkError as ne:
        return jsonify({"error": "Network error"}), 503
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/')
def home():
    return """
    <h1>🚀 Advanced Crypto Trading API</h1>
    <h2>Fitur Terbaru:</h2>
    <ul>
        <li>✅ Real-time Volume & Order Book Analysis</li>
        <li>✅ Fibonacci Retracement Levels</li>
        <li>✅ Pivot Points & Support/Resistance</li>
        <li>✅ Auto-Update Technical Indicators</li>
        <li>✅ On-Chain Data Integration</li>
        <li>✅ MACD Crossover & Candlestick Pattern Alerts</li>
    </ul>
    <h2>Endpoints:</h2>
    <ul>
        <li><code>/api/analyze?symbol=BTC/USDT&timeframe=1d</code> - Analisis lengkap</li>
        <li><code>/api/alerts/BTC/USDT</code> - Alert terbaru</li>
        <li><code>/api/realtime/BTC/USDT</code> - Data real-time</li>
        <li><code>/api/fibonacci/BTC/USDT</code> - Level Fibonacci</li>
    </ul>
    <h3>Test Links:</h3>
    <ul>
        <li><a href="/api/analyze?symbol=BTC/USDT&timeframe=1d">Test Analyze BTC/USDT</a></li>
        <li><a href="/api/realtime/BTC/USDT">Test Realtime BTC/USDT</a></li>
        <li><a href="/api/fibonacci/BTC/USDT">Test Fibonacci BTC/USDT</a></li>
        <li><a href="/api/chart/BTC/USDT/1d">Test Chart BTC/USDT</a></li>
    </ul>
    """
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    return response            
    
    if not ohlcv or len(ohlcv) < 2:
    return jsonify({"error": "Insufficient data"}), 400

@app.after_request
def add_header(response):
    if request.path.startswith('/api/'):
        response.cache_control.max_age = 300  # 5 minutes
    return response
# Ganti nama file "your_main_file.py" dengan nama file python Anda, misal "app.py"
# Blok ini tidak lagi diperlukan jika Anda menggunakan Gunicorn melalui Dockerfile,
# tetapi tidak ada salahnya untuk membiarkannya untuk testing lokal.
# Pastikan portnya berbeda dari yang digunakan Gunicorn untuk menghindari konflik.
debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
app.run(host='0.0.0.0', port=5000, debug=debug_mode)
