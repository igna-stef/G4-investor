import logging
import os
import uuid
import requests
from fastapi import FastAPI, HTTPException, Request, Response
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, EmailStr
import yfinance as yf  # Asegúrate de tener instalada la librería yfinance

from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv # NEW
import plotly.graph_objects as go # NEW
from plotly.subplots import make_subplots # NEW
from datetime import datetime, timedelta # NEW
import pandas as pd # NEW
import json # NEW

load_dotenv() # NEW 


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

SESSIONS = {}

# ==================================================
# MODELOS Pydantic
# ==================================================

class MessageModel(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[MessageModel]
    config: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    llm_response: str
    retrieved_docs: List[Dict[str, Any]]

# ==================================================
# TOOL para llamar al microservicio de Chroma
# ==================================================

class ChromaQuerySchema(BaseModel):
    query: str

# def _call_chroma_api_tool(query: str) -> Dict[str, Any]:
#     """
#     Llama a la API de Chroma (p.ej. http://localhost:8002/chatbot)
#     y retorna un diccionario con 'llm_response' y 'retrieved_docs'.
#     """
#     try:
#         api_url = os.getenv("API_URL", "http://host.docker.internal:8002/chatbot")
#         payload = {"question": query}
#         resp = requests.post(api_url, json=payload, timeout=10)

#         logger.info(f"Código de respuesta: {resp.status_code}")
#         logger.info(f"Contenido de respuesta: {resp.text}")

#         resp.raise_for_status()
#         data = resp.json()
        
#         sources = [
#             doc["metadata"]["source"]
#             for doc in data.get("retrieved_docs", [])
#             if "metadata" in doc and "source" in doc["metadata"]
#         ]

#         response = data.get("llm_response", "")

#         return f"{response} \n\nSources: {sources}"

#     except Exception as e:
#         logger.error(f"Error llamando a la API de Chroma: {e}", exc_info=True)
#         return {
#             "llm_response": f"Ocurrió un error consultando a Chroma: {str(e)}",
#             "retrieved_docs": []
#         }


def _call_chroma_api_tool(query: str) -> str:
    """
    Llama a la API de Chroma, la cual consulta una base de datos vectorial que contiene
    más de 1000 documentos PDF sobre temas financieros (documentos fechados desde 2015 hasta 2021/2022).
    
    Retorna un string que contiene el mensaje de respuesta y, al final, una única línea "Sources: ..." 
    que consolida todas las fuentes utilizadas. Si no se encuentra información en el corpus, se debe
    indicar: "The answer is not found within the corpus documents, but I have found this in other sources:"
    y generar una respuesta completa a partir de otras fuentes (incluyendo "Internet Search").
    
    Se implementa un mecanismo de reintentos (hasta 3 intentos) en caso de timeouts o errores transitorios.
    En caso de error definitivo, se devuelve un mensaje genérico sin detalles técnicos.
    """
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            api_url = os.getenv("API_URL", "http://api:80/chatbot")
            payload = {"question": query}
            resp = requests.post(api_url, json=payload, timeout=20)
            
            logger.info(f"Código de respuesta (Intento {attempt+1}): {resp.status_code}")
            logger.info(f"Contenido de respuesta: {resp.text}")
            
            resp.raise_for_status()
            data = resp.json()
            
            sources = [
                doc["metadata"]["source"]
                for doc in data.get("retrieved_docs", [])
                if "metadata" in doc and "source" in doc["metadata"]
            ]
            # Si no se encontraron fuentes en el corpus, se usará "Internet Search"
            if not sources:
                sources = ["Internet Search"]

            response = data.get("llm_response", "").strip()
            return f"{response}\n\nSources: {', '.join(sorted(set(sources)))}"
        except requests.exceptions.ReadTimeout as e:
            logger.error(f"Read timeout en el intento {attempt+1} para la query: {query}", exc_info=True)
            if attempt == max_attempts - 1:
                return "An error occurred while querying Chroma, please check the connection.\n\nSources: Chroma API"
            continue
        except Exception as e:
            logger.error("Error llamando a la API de Chroma", exc_info=True)
            return "An error occurred while querying Chroma, please check the connection.\n\nSources: Chroma API"


chroma_tool = StructuredTool.from_function(
    func=_call_chroma_api_tool,
    name="call_chroma_api",
    description=(
        "Queries the Chroma API using a given 'query' and retrieves relevant vector-based information. "
        "Returns two key outputs:\n"
        "- 'llm_response': A generated response based on the retrieved data.\n"
        "- 'retrieved_docs': A collection of the most relevant documents fetched from the Chroma database. "
        "This tool is useful for retrieving contextual financial data, past stock trends, or other relevant financial insights."
    ),
    args_schema=ChromaQuerySchema,
    return_direct=True
)

# ==================================================
# TOOL para consultar el precio de una acción en Yahoo Finance
# ==================================================

class StockPriceQuery(BaseModel):
    ticker: str

def _get_stock_price(ticker: str) -> Dict[str, Any]:
    """
    Consulta el precio actual de una acción usando Yahoo Finance.
    Retorna un diccionario con 'llm_response' y 'retrieved_docs'.
    Se intenta obtener el precio utilizando diferentes campos disponibles.
    """
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("regularMarketPrice") or stock.info.get("previousClose")
        if price is None and hasattr(stock, "fast_info"):
            price = stock.fast_info.get("lastPrice")
        if price is None:
            raise ValueError("Precio no disponible.")
            
        return f"The current price of {ticker.upper()} is ${price:.2f}.\n\nSources: Yahoo Finance"
    except Exception as e:
        logger.error(f"Error obteniendo el precio de {ticker}: {e}")
        return f"Error obtaining stock price for {ticker}: {str(e)}"


stock_price_tool = StructuredTool.from_function(
    func=_get_stock_price,
    name="get_stock_price",
    description=(
        "Retrieves the current stock price of a given company using Yahoo Finance. "
        "Takes a stock ticker (e.g., 'AAPL' for Apple, 'TSLA' for Tesla) as input and returns the latest available market price. "
        "This tool is useful for checking real-time stock values and assessing market movements."
    ),
    args_schema=StockPriceQuery,
    return_direct=True
)

# ==================================================
# TOOL para consultar información financiera actual en Yahoo Finance
# ==================================================

class FinancialInfoQuery(BaseModel):
    ticker: str

def _get_financial_info(ticker: str) -> Dict[str, Any]:
    """
    Consulta información financiera actual de una empresa usando Yahoo Finance.
    Retorna un diccionario con 'llm_response' y 'retrieved_docs'.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get("marketCap", "N/A")
        trailing_pe = info.get("trailingPE", "N/A")
        forward_pe = info.get("forwardPE", "N/A")
        dividend_yield = info.get("dividendYield", "N/A")
        response_text = (
            f"Financial information for {ticker.upper()}: "
            f"Market Cap: {market_cap}, Trailing P/E: {trailing_pe}, "
            f"Forward P/E: {forward_pe}, Dividend Yield: {dividend_yield}."
        )
    
        return f"{response_text}\n\nSources: Yahoo Finance"
    except Exception as e:
        logger.error(f"Error obteniendo información financiera de {ticker}: {e}", exc_info=True)
        return f"Error al obtener información financiera de {ticker}: {str(e)}"


financial_info_tool = StructuredTool.from_function(
    func=_get_financial_info,
    name="get_financial_info",
    description=(
        "Retrieves detailed financial information for a given stock using Yahoo Finance. "
        "Takes a stock ticker (e.g., 'AAPL' for Apple, 'TSLA' for Tesla) as input and returns key financial metrics. "
        "The retrieved data may include market capitalization, P/E ratio, revenue, earnings, and other relevant indicators. "
        "This tool is useful for fundamental analysis and evaluating a company's financial health."
    ),
    args_schema=FinancialInfoQuery,
    return_direct=True
)

# --------------------------- NEW ---------------------------

# ==================================================
# TOOL para análisis de sentimiento de mercado
# ==================================================

class MarketSentimentQuery(BaseModel):
    ticker: str

def _get_market_sentiment(ticker: str) -> Dict[str, Any]:
    """
    Analiza el sentimiento de mercado usando datos de Yahoo Finance.
    Incluye recomendaciones de analistas y métricas de sentimiento.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Recopilar datos de sentimiento
        sentiment_data = {
            "recommendationMean": info.get("recommendationMean", "N/A"),
            "recommendationKey": info.get("recommendationKey", "N/A"),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions", "N/A"),
            "targetHighPrice": info.get("targetHighPrice", "N/A"),
            "targetLowPrice": info.get("targetLowPrice", "N/A"),
            "targetMeanPrice": info.get("targetMeanPrice", "N/A")
        }
        
        response_text = (
            f"Market sentiment analysis for {ticker.upper()}:\n"
            f"- Analyst Rating: {sentiment_data['recommendationMean']}/5 ({sentiment_data['recommendationKey']})\n"
            f"- Number of Analysts: {sentiment_data['numberOfAnalystOpinions']}\n"
            f"- Price Targets: Low ${sentiment_data['targetLowPrice']}, "
            f"Mean ${sentiment_data['targetMeanPrice']}, "
            f"High ${sentiment_data['targetHighPrice']}"
        )
        
        return f"{response_text}\n\nSources: Yahoo Finance"
    except Exception as e:
        logger.error(f"Error obtaining market sentiment for {ticker}: {e}")
        return f"Error obtaining market sentiment for {ticker}: {str(e)}"

market_sentiment_tool = StructuredTool.from_function(
    func=_get_market_sentiment,
    name="get_market_sentiment",
    description=(
        "Analyzes market sentiment for a given stock using Yahoo Finance data. "
        "Provides analyst recommendations, price targets, and overall market sentiment. "
        "This tool is useful for understanding market expectations and professional opinions."
    ),
    args_schema=MarketSentimentQuery,
    return_direct=True
)


# --------------------------- NEW ---------------------------

# ==================================================
# TOOL para análisis técnico y gráficos
# ==================================================

class TechnicalAnalysisQuery(BaseModel):
    ticker: str
    period: str = "1mo"  # default a 1 mes

def _get_technical_analysis(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """
    Realiza análisis técnico básico y devuelve métricas clave.
    Usa un período más largo para calcular SMAs correctamente.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Obtener datos históricos para un año para calcular SMAs correctamente
        hist_long = stock.history(period="1y")
        
        # Obtener datos del período solicitado para el precio actual y RSI
        hist_short = stock.history(period="1mo")
        
        # Calcular métricas técnicas básicas
        current_price = hist_short['Close'][-1]
        
        # Calcular SMAs usando el conjunto de datos más largo
        sma_50 = hist_long['Close'].rolling(window=50).mean().iloc[-1]
        sma_200 = hist_long['Close'].rolling(window=200).mean().iloc[-1]
        
        # Calcular RSI usando los datos más recientes
        rsi = calculate_rsi(hist_short['Close'])
        
        # Calcular tendencias
        sma_50_trend = "above" if current_price > sma_50 else "below"
        sma_200_trend = "above" if current_price > sma_200 else "below"
        
        # Determinar tendencia general
        if current_price > sma_50 and current_price > sma_200:
            trend = "bullish"
        elif current_price < sma_50 and current_price < sma_200:
            trend = "bearish"
        else:
            trend = "mixed"
        
        # Interpretar RSI
        rsi_interpretation = (
            "overbought" if rsi > 70 
            else "oversold" if rsi < 30 
            else "neutral"
        )
        
        response_text = (
            f"Technical Analysis for {ticker.upper()}:\n\n"
            f"Current Price: ${current_price:.2f}\n"
            f"50-day SMA: ${sma_50:.2f} (price is {sma_50_trend})\n"
            f"200-day SMA: ${sma_200:.2f} (price is {sma_200_trend})\n"
            f"RSI (14): {rsi:.2f} ({rsi_interpretation})\n\n"
            f"Overall Trend: {trend.capitalize()}\n"
            f"Analysis Summary:\n"
            f"• The stock is currently trading {sma_50_trend} its 50-day SMA and {sma_200_trend} its 200-day SMA\n"
            f"• RSI indicates the stock is {rsi_interpretation}\n"
            f"• The overall technical trend appears to be {trend}"
        )
        
        return f"{response_text}\n\nSources: Yahoo Finance"
    except Exception as e:
        logger.error(f"Error performing technical analysis for {ticker}: {e}")
        return f"Error performing technical analysis for {ticker}: {str(e)}"

def calculate_rsi(prices, periods=14):
    """
    Calcula el RSI (Relative Strength Index)
    """
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        return rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return 50  # valor neutral por defecto en caso de error

technical_analysis_tool = StructuredTool.from_function(
    func=_get_technical_analysis,
    name="get_technical_analysis",
    description=(
        "Performs technical analysis on a given stock, including SMA calculations and RSI. "
        "Takes a stock ticker and time period as input. "
        "Returns key technical indicators and their interpretations including:\n"
        "- Current price\n"
        "- 50-day and 200-day Simple Moving Averages (SMA)\n"
        "- Relative Strength Index (RSI)\n"
        "- Overall trend analysis and interpretation\n"
        "The analysis includes price position relative to SMAs and RSI interpretation."
    ),
    args_schema=TechnicalAnalysisQuery,
    return_direct=True
)


class ChartQuery(BaseModel):
    ticker: str
    period: str = "6mo"  # default 6 months
    interval: str = "1d"  # default daily data
    include_volume: bool = True
    include_indicators: bool = True


def _generate_chart_analysis(ticker: str, hist: pd.DataFrame, current_price: float, 
                           sma50: float, sma200: float) -> str:
    """
    Genera un análisis narrativo del gráfico basado en los datos históricos y métricas.
    """
    try:
        # Calcular métricas clave
        start_price = hist['Close'].iloc[0]
        price_change = current_price - start_price
        price_change_pct = (price_change / start_price) * 100
        
        # Obtener información adicional de Yahoo Finance
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Determinar tendencia
        trend = "upward" if price_change > 0 else "downward"
        strength = (
            "strong" if abs(price_change_pct) > 20 
            else "moderate" if abs(price_change_pct) > 10 
            else "slight"
        )
        
        # Analizar volumen
        avg_volume = hist['Volume'].mean()
        recent_volume = hist['Volume'].tail(5).mean()
        volume_trend = "increasing" if recent_volume > avg_volume else "decreasing"
        
        # Analizar SMAs
        sma_support = (
            "supported by both the 50-day and 200-day moving averages" 
            if current_price > sma50 and current_price > sma200
            else "trading below key moving averages" 
            if current_price < sma50 and current_price < sma200
            else "showing mixed signals relative to moving averages"
        )
        
        # Obtener información fundamental relevante
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        business_summary = info.get('longBusinessSummary', '').split('.')[0]  # Primera oración
        
        # Construir análisis narrativo
        analysis_parts = [
            f"The chart shows {ticker}'s {strength} {trend} trend over the displayed period.",
            f"The stock has {'gained' if price_change > 0 else 'lost'} {abs(price_change_pct):.1f}% in value,",
            f"{sma_support}.",
            f"Trading volume has been {volume_trend}.",
        ]
        
        # Añadir contexto fundamental si está disponible
        if sector and industry:
            analysis_parts.append(
                f"As a {sector} company in the {industry} industry, {ticker} {business_summary.lower()}."
            )
        
        # Añadir información sobre eventos recientes si están disponibles
        recent_events = []
        if info.get('revenueGrowth'):
            growth = info['revenueGrowth'] * 100
            recent_events.append(f"revenue growth of {growth:.1f}%")
        if info.get('earningsGrowth'):
            growth = info['earningsGrowth'] * 100
            recent_events.append(f"earnings growth of {growth:.1f}%")
        
        if recent_events:
            analysis_parts.append(
                f"Recent performance shows {' and '.join(recent_events)}."
            )
        
        return " ".join(analysis_parts)
    
    except Exception as e:
        logger.error(f"Error generating chart analysis: {e}")
        return "Analysis not available due to insufficient data."

def _format_analysis_text(ticker: str, current_price: float, price_change: float, 
                        price_change_pct: float, period: str, interval: str,
                        sma50: float = None, sma200: float = None) -> str:
    """
    Helper function to format analysis text consistently.
    Now includes more detailed technical analysis.
    """
    
    # Determinar tendencias si los SMAs están disponibles
    trend = "Not available"
    if sma50 is not None and sma200 is not None and not pd.isna(sma50) and not pd.isna(sma200):
        if current_price > sma50 and current_price > sma200:
            trend = "Bullish"
        elif current_price < sma50 and current_price < sma200:
            trend = "Bearish"
        else:
            trend = "Mixed"
    
    analysis = [
        f"Chart Analysis for {ticker}:",
        f"Current Price: ${current_price:.2f}",
        f"Price Change: ${price_change:.2f} ({price_change_pct:.2f}%)",
        f"Period: {period}",
        f"Interval: {interval}",
        f"Overall Trend: {trend}"
    ]
    
    # Add SMAs with trend indication
    if sma50 is not None and not pd.isna(sma50):
        trend_50 = "above" if current_price > sma50 else "below"
        analysis.append(f"50-day SMA: ${sma50:.2f} (price is {trend_50})")
    
    if sma200 is not None and not pd.isna(sma200):
        trend_200 = "above" if current_price > sma200 else "below"
        analysis.append(f"200-day SMA: ${sma200:.2f} (price is {trend_200})")
    
    # Add technical analysis summary if available
    if sma50 is not None and sma200 is not None and not pd.isna(sma50) and not pd.isna(sma200):
        analysis.append("\nTechnical Analysis Summary:")
        analysis.append(f"• The stock is trading {trend_50} its 50-day SMA and {trend_200} its 200-day SMA")
        analysis.append(f"• The overall technical trend appears to be {trend.lower()}")
    
    return "\n".join(analysis)


# --------------------------- NEW ---------------------------

# ==================================================
# TOOL para generar grafico de la funcion
# ==================================================

def _generate_stock_chart(ticker: str, period: str = "6mo", interval: str = "1d", 
                         include_volume: bool = True, include_indicators: bool = True) -> Dict[str, Any]:
    """
    Generates an interactive stock chart with technical indicators and analysis.
    """
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data available for {ticker}")

        # Calculate metrics
        current_price = df['Close'].iloc[-1]
        price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
        price_change_pct = (price_change / df['Close'].iloc[0]) * 100

        # Calculate SMAs
        sma50 = df['Close'].rolling(window=50).mean().iloc[-1]
        sma200 = df['Close'].rolling(window=200).mean().iloc[-1]

        # Create figure
        fig = make_subplots(
            rows=2 if include_volume else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3] if include_volume else [1],
            subplot_titles=[f'{ticker} Price', 'Volume'] if include_volume else [f'{ticker} Price']
        )

        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add volume
        if include_volume:
            colors = ['red' if row['Open'] > row['Close'] else 'green' 
                     for i, row in df.iterrows()]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors
                ),
                row=2, col=1
            )

        # Add technical indicators
        if include_indicators:
            # Add 50-day SMA
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'].rolling(window=50).mean(),
                    name='50-day SMA',
                    line=dict(color='orange', width=1.5)
                ),
                row=1, col=1
            )
            
            # Add 200-day SMA
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'].rolling(window=200).mean(),
                    name='200-day SMA',
                    line=dict(color='red', width=1.5)
                ),
                row=1, col=1
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{ticker} Stock Analysis",
                x=0.5,
                xanchor='center'
            ),
            yaxis_title="Price (USD)",
            yaxis2_title="Volume" if include_volume else None,
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            template="plotly_white"
        )

        # Generate technical analysis text
        technical_text = _format_analysis_text(
            ticker=ticker,
            current_price=current_price,
            price_change=price_change,
            price_change_pct=price_change_pct,
            period=period,
            interval=interval,
            sma50=sma50,
            sma200=sma200
        )

        # Generate narrative analysis
        narrative_analysis = _generate_chart_analysis(
            ticker=ticker,
            hist=df,
            current_price=current_price,
            sma50=sma50,
            sma200=sma200
        )

        # Combine technical and narrative analysis
        combined_analysis = f"{technical_text}\n\nChart Analysis:\n{narrative_analysis}"

        return {
            "success": True,
            "text": combined_analysis,
            "figure": fig.to_json(),
            "source": "Yahoo Finance"
        }

    except Exception as e:
        logger.error(f"Error generating chart for {ticker}: {e}")
        return {
            "success": False,
            "text": f"Error generating chart for {ticker}: {str(e)}",
            "figure": None,
            "source": "Error"
        }

 
chart_tool = StructuredTool.from_function(
    func=_generate_stock_chart,
    name="generate_stock_chart",
    description=(
        "Generates interactive stock charts with technical indicators. "
        "Can display candlestick patterns, volume, moving averages. "
        "Parameters include ticker symbol, time period, and interval. "
        "Returns both a visualization and a text analysis summary."
    ),
    args_schema=ChartQuery,
    return_direct=True
)

# --------------------------- NEW ---------------------------

# ==================================================
# TOOL para generar un analisis completo utilizando varias funciones
# ==================================================

class ComprehensiveAnalysisQuery(BaseModel):
    ticker: str
    period: str = "6mo"  # default a 6 meses
    interval: str = "1d"  # default a datos diarios
    include_volume: bool = True
    include_indicators: bool = True

def _get_comprehensive_analysis(ticker: str, period: str = "6mo", interval: str = "1d",
                              include_volume: bool = True, include_indicators: bool = True) -> Dict[str, Any]:
    """
    Realiza un análisis completo incluyendo gráfico y todos los indicadores en una sola llamada eficiente.
    """
    try:
        # Una sola instancia de Ticker para todas las operaciones
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Obtener datos históricos una sola vez
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            raise ValueError(f"No historical data available for {ticker}")
        
        # 1. Precio actual y datos de mercado
        current_price = info.get("regularMarketPrice") or info.get("previousClose")
        if current_price is None and hasattr(stock, "fast_info"):
            current_price = stock.fast_info.get("lastPrice")
        
        price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[0]
        price_change_pct = (price_change / hist['Close'].iloc[0]) * 100
        
        # 2. Métricas fundamentales
        market_cap = info.get("marketCap", "N/A")
        trailing_pe = info.get("trailingPE", "N/A")
        forward_pe = info.get("forwardPE", "N/A")
        dividend_yield = info.get("dividendYield", "N/A")
        
        # 3. Datos de sentimiento
        sentiment_data = {
            "recommendationMean": info.get("recommendationMean", "N/A"),
            "recommendationKey": info.get("recommendationKey", "N/A"),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions", "N/A"),
            "targetHighPrice": info.get("targetHighPrice", "N/A"),
            "targetLowPrice": info.get("targetLowPrice", "N/A"),
            "targetMeanPrice": info.get("targetMeanPrice", "N/A")
        }
        
        # 4. Análisis técnico
        sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        rsi = calculate_rsi(hist['Close'])
        
        # Determinar tendencias
        sma_50_trend = "above" if current_price > sma_50 else "below"
        sma_200_trend = "above" if current_price > sma_200 else "below"
        
        if current_price > sma_50 and current_price > sma_200:
            trend = "bullish"
        elif current_price < sma_50 and current_price < sma_200:
            trend = "bearish"
        else:
            trend = "mixed"
        
        # Interpretar RSI
        rsi_interpretation = (
            "overbought" if rsi > 70 
            else "oversold" if rsi < 30 
            else "neutral"
        )
        
        # 5. Generar gráfico
        fig = make_subplots(
            rows=2 if include_volume else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3] if include_volume else [1],
            subplot_titles=[f'{ticker} Price', 'Volume'] if include_volume else [f'{ticker} Price']
        )

        # Añadir candlestick
        fig.add_trace(
            go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Añadir volumen
        if include_volume:
            colors = ['red' if row['Open'] > row['Close'] else 'green' 
                     for i, row in hist.iterrows()]
            fig.add_trace(
                go.Bar(
                    x=hist.index,
                    y=hist['Volume'],
                    name='Volume',
                    marker_color=colors
                ),
                row=2, col=1
            )

        # Añadir indicadores técnicos
        if include_indicators:
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist['Close'].rolling(window=50).mean(),
                    name='50-day SMA',
                    line=dict(color='orange', width=1.5)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist['Close'].rolling(window=200).mean(),
                    name='200-day SMA',
                    line=dict(color='red', width=1.5)
                ),
                row=1, col=1
            )

        # Actualizar layout
        fig.update_layout(
            title=dict(
                text=f"{ticker} Stock Analysis",
                x=0.5,
                xanchor='center'
            ),
            yaxis_title="Price (USD)",
            yaxis2_title="Volume" if include_volume else None,
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            template="plotly_white"
        )
        
        # Construir respuesta textual
        response = [
            f"Comprehensive Analysis for {ticker.upper()}:\n",
            
            "Current Market Data:",
            f"• Price: ${current_price:.2f}",
            f"• Price Change: ${price_change:.2f} ({price_change_pct:.2f}%)",
            f"• Market Cap: {market_cap:,}",
            f"• P/E Ratio (Trailing): {trailing_pe}",
            f"• P/E Ratio (Forward): {forward_pe}",
            f"• Dividend Yield: {dividend_yield:.2%}" if isinstance(dividend_yield, (float, int)) else f"• Dividend Yield: {dividend_yield}",
            
            "\nMarket Sentiment:",
            f"• Analyst Rating: {sentiment_data['recommendationMean']}/5 ({sentiment_data['recommendationKey']})",
            f"• Number of Analysts: {sentiment_data['numberOfAnalystOpinions']}",
            f"• Price Targets: Low ${sentiment_data['targetLowPrice']}, Mean ${sentiment_data['targetMeanPrice']}, High ${sentiment_data['targetHighPrice']}",
            
            "\nTechnical Analysis:",
            f"• 50-day SMA: ${sma_50:.2f} (price is {sma_50_trend})",
            f"• 200-day SMA: ${sma_200:.2f} (price is {sma_200_trend})",
            f"• RSI (14): {rsi:.2f} ({rsi_interpretation})",
            f"• Overall Trend: {trend.capitalize()}",
            
            "\nTechnical Summary:",
            f"• The stock is trading {sma_50_trend} its 50-day SMA and {sma_200_trend} its 200-day SMA",
            f"• RSI indicates the stock is {rsi_interpretation}",
            f"• The overall technical trend appears to be {trend}"
        ]

        return {
            "success": True,
            "text": "\n".join(response) + "\n\nSources: Yahoo Finance",
            "figure": fig.to_json()
        }
        
    except Exception as e:
        logger.error(f"Error performing comprehensive analysis for {ticker}: {e}")
        return {
            "success": False,
            "text": f"Error performing comprehensive analysis for {ticker}: {str(e)}",
            "figure": None
        }

comprehensive_analysis_tool = StructuredTool.from_function(
    func=_get_comprehensive_analysis,
    name="get_comprehensive_analysis",
    description=(
        "Performs a complete stock analysis including:\n"
        "- Current price and market data\n"
        "- Key financial metrics and ratios\n"
        "- Market sentiment and analyst recommendations\n"
        "- Technical analysis with SMA and RSI\n"
        "- Interactive price chart with volume and indicators\n"
        "Takes a stock ticker and optional parameters for customization.\n"
        "Returns both comprehensive analysis text and interactive visualization."
    ),
    args_schema=ComprehensiveAnalysisQuery,
    return_direct=True
)


# ==================================================
# CONFIGURACIÓN DEL LLM Y AGENTE
# ==================================================

llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.1,
    openai_api_key= os.getenv("OPENAI_API_KEY"),
)

llm = llm.bind_tools([
    chart_tool,
    chroma_tool,
    stock_price_tool, 
    technical_analysis_tool,
    financial_info_tool,
    market_sentiment_tool,
    comprehensive_analysis_tool
    ], 
    tool_choice="auto")

# llm = llm.bind_tools([chroma_tool, stock_price_tool, financial_info_tool], tool_choice="auto")

PROMPT_SYSTEM = """
You are a virtual financial assistant specializing in answering questions about finance, investments, and stock markets in English.
You are known for being a funny and sarcastic assistant when answering questions outside your expertise.
You can access the following tools to improve the accuracy of your responses:

1.	Chroma Vector Database: This tool provides historical data and contextual insights extracted from over 1000 PDF documents on financial topics (documents dated from 2015 to 2021/2022). When using this tool:
    - If the relevant answer is found, include the document sources (e.g., PDF names) in your response.
    - If no matching information is found, explicitly state: 
        "The answer is not found within the corpus documents, but I have found this in other sources:" 
        and then generate a comprehensive answer based on additional data (include "Internet Search" as a source).
2.	get_stock_price: To retrieve the current stock price from Yahoo Finance.
3.	get_financial_info: To obtain detailed financial information about a company using Yahoo Finance.

Response Instructions:
- Provide clear, precise, and structured responses.
- Use a professional and educational tone, adapting to the user’s level of knowledge.
- If the question involves financial analysis, offer relevant context before responding.
- If a tool provides information, integrate the data naturally into your response.
- If you do not have sufficient information, be transparent rather than speculating.
- If the user's request is non-financial (e.g., "Who is Cristiano Ronaldo?"), respond with a sarcastic remark and do not include a "Sources:" line.
- IMPORTANT: Your final answer must include only one single line at the end that begins with "Sources: " listing all the sources used (e.g., Yahoo Finance, Chroma API, Internet Search). If intermediate tool outputs include "Sources:" lines, ignore them and merge all source information into one final "Sources:" line with unique values.


Examples of Expected Questions and Responses:

Example 1: Stock Price Inquiry

User: What is the current price of AAPL?
Action: Use get_stock_price("AAPL")
Expected Response:

“The current stock price of Apple Inc. (AAPL) is $189.52 USD (last update: February 6, 2025). Keep in mind that stock prices can change rapidly due to market volatility.”

Example 2: Company Financial Information

User: What is Tesla’s market capitalization?
Action: Use get_financial_info("TSLA")
Expected Response:

“Tesla Inc. (TSLA) has a market capitalization of $850 billion USD according to the most recent data. This value represents the total valuation of the company in the stock market and is a key indicator of its size and relevance in the industry.”

Example 3: Financial Analysis with Context

User: Is it a good time to invest in Microsoft?
Action: Use get_stock_price("MSFT") and get_financial_info("MSFT"), and make a comprehensive analysis of the company usis the data retrieved from that two tools.
Expected Response:
“Microsoft Corp. (MSFT) is currently trading at $402.75 USD. Its recent performance shows a 15% increase over the past six months, driven by a rise in cloud service revenues.
From a fundamental perspective, Microsoft has a market capitalization of $3.1 trillion USD, with a P/E ratio of 32, indicating a high valuation compared to the tech sector.
If you are looking for a long-term investment, the tech sector remains strong, but consider the risks of overvaluation and market volatility. Would you like us to analyze a specific time period or compare Microsoft with other companies?”*


Example 4: Non-Financial Query

User: Who is Cristiano Ronaldo?

Expected Response:
"I only provide financial advice. Would you like to know his net worth instead?"
(No "Sources:" line is added in this case.)

Additional Considerations:
- If the user requests technical analysis, mention relevant indicators such as RSI, moving averages, or trading volume.
- If the user is looking for investment recommendations, emphasize that you do not provide direct financial advice but can offer data and trends to assist their decision-making.
- If the user inquires about general trends, mention macroeconomic events and their impact on markets.

Ideal Response Format:
- Accurate data obtained from the available tools.
- Concise and clear explanation for users with no advanced knowledge.
- Financial context if the question requires it.
- Closing with a suggestion or question to encourage user engagement.

1. get_stock_price: To retrieve the current stock price from Yahoo Finance.
2. get_financial_info: To obtain detailed financial information about a company using Yahoo Finance.
3. get_technical_analysis: To perform technical analysis including SMA and RSI calculations
4. generate_stock_chart: To create visual price charts
5. get_comprehensive_analysis: To generate a deep stock analysis

Tool Usage Guidelines:
- When a user asks for "technical analysis", use get_technical_analysis
- When a user asks to "show", "display", or "see" a chart/graph, use generate_stock_chart
- Do not combine get_technical_analysis and generate_stock_chart unless explicitly requested

Examples:

User: "Can you show me a technical analysis of NVDA?"
Action: Use get_technical_analysis("NVDA")
Response: [Technical metrics and indicators]

User: "Show me a chart for NVDA"
Action: Use generate_stock_chart("NVDA")
Response: [Visual price chart]

User: "Can you show me both technical analysis and chart for NVDA?"
Action: Use both get_technical_analysis("NVDA") and generate_stock_chart("NVDA")
Response: [Both metrics and visual chart]

Remember to keep responses clear and focused on the specific tool that best matches the user's request.

IMPORTANT - Tool Selection Rules:

1. Use get_technical_analysis when:
   - User asks for "technical analysis"
   - User mentions "technical indicators"
   - User requests "indicators" or "metrics"
   Even if these requests include the word "show"

2. Use generate_stock_chart ONLY when:
   - User specifically requests a visual chart/graph
   - User explicitly asks to see price movements
   - The request is specifically about visualizing the price

3. Use Comprehensive Stock Analysis Tool (get_comprehensive_analysis) when:
• User request a complete market overview including:
  - Current price and market metrics
  - Financial ratios and fundamentals
  - Market sentiment and analyst views
  - Technical indicators (SMA, RSI)
  - Interactive price visualization

When to Use get_comprehensive_analysis:
• For any stock-specific question
• When complete market analysis is needed
• When visual price representation is requested
• To get real-time market insights

Tool Parameters for get_comprehensive_analysis:
• ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
• period: Time period (default "6mo")
• interval: Data interval (default "1d")

Examples:

Use get_technical_analysis for:
- "Show me a technical analysis of NVDA"
- "Can you show me technical indicators for AAPL"
- "Give me technical analysis of TSLA"
- "Show technical metrics for MSFT"

Use generate_stock_chart ONLY for:
- "Show me NVDA's price chart"
- "Display the stock chart for AAPL"
- "Can I see a price graph of TSLA"
- "Show me how MSFT stock has moved"

Remember:
- The presence of "technical analysis" or "technical indicators" ALWAYS overrides any chart request
- The word "show" alone should not trigger a chart - look at the full context
"""

# Crear el agente con las tools
graph_builder = create_react_agent(
    llm,
    tools=[
        chart_tool,
        chroma_tool,
        stock_price_tool, 
        technical_analysis_tool,
        financial_info_tool,
        market_sentiment_tool,
        comprehensive_analysis_tool
    ],
    state_modifier=PROMPT_SYSTEM,
    checkpointer=MemorySaver()
)

# ==================================================
# FUNCIONES AUXILIARES
# ==================================================

def extract_ticker(message: str) -> str:
    """
    Extrae y valida el símbolo bursátil dinámicamente usando Yahoo Finance
    """
    # Normalizar mensaje
    message_upper = message.upper()
    
    # Palabras a ignorar
    ignore_words = {
        'SHOW', 'ME', 'A', 'THE', 'CHART', 'GRAPH', 'PLOT', 'OF', 'FOR', 
        'PRICE', 'STOCK', 'STOCKS', 'VISUALIZATION', 'VIEW', 'ANALYSIS',
        'SENTIMENT', 'TECHNICAL', 'MARKET', 'FINANCIAL', 'INFO', 'INFORMATION',
        'CAN', 'YOU', 'PLEASE', 'TELL', 'WHAT', 'IS', 'ARE', 'ABOUT'
    }
    
    # Primero, intentar encontrar palabras que parezcan tickers (2-5 letras, todas mayúsculas)
    potential_tickers = [
        word for word in message_upper.split()
        if word.isalpha() and 2 <= len(word) <= 5 and word not in ignore_words
    ]
    
    logger.debug(f"Potential tickers found: {potential_tickers}")
    
    # Validar cada potencial ticker
    for ticker in potential_tickers:
        try:
            stock = yf.Ticker(ticker)
            # Intentar obtener el precio actual como validación
            info = stock.info
            if info and ('regularMarketPrice' in info or 'currentPrice' in info):
                logger.debug(f"Valid ticker found: {ticker}")
                return ticker
        except Exception as e:
            logger.debug(f"Failed to validate ticker {ticker}: {str(e)}")
            continue
    
    # Si no se encontró ningún ticker válido, buscar en el mensaje completo
    # patrones comunes como "$MSFT" o "(AAPL)"
    import re
    pattern = r'[\$$$]?([A-Z]{2,5})[$$\s]?'
    matches = re.findall(pattern, message_upper)
    
    for match in matches:
        if match not in ignore_words:
            try:
                stock = yf.Ticker(match)
                info = stock.info
                if info and ('regularMarketPrice' in info or 'currentPrice' in info):
                    logger.debug(f"Valid ticker found from pattern: {match}")
                    return match
            except Exception as e:
                logger.debug(f"Failed to validate ticker from pattern {match}: {str(e)}")
                continue
    
    return None

# ==================================================
# ENDPOINTS
# ==================================================

@app.get("/")
async def root():
    """
    Endpoint de prueba para verificar que el server corre bien.
    """
    return {"message": "¡Bienvenido! Tu backend de FastAPI está en marcha."}

# @app.post("/chat")
# async def chat_endpoint(request: ChatRequest):
#     messages_list = [msg.dict() for msg in request.messages]
    
#     if not messages_list:
#         raise HTTPException(status_code=400, detail="No user messages found.")
    
#     try:
#         result = graph_builder.invoke(
#             {"messages": messages_list},
#             config={"configurable": {"thread_id": str(uuid.uuid4())}}
#         )
#         final_message = result["messages"][-1].content
#         return {"response": final_message}
#     except Exception as e:
#         logger.error(f"Error invoking the agent: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    messages_list = [msg.dict() for msg in request.messages]

    if not messages_list:
        raise HTTPException(status_code=400, detail="No user messages found.")

    try:
        last_message = messages_list[-1]["content"]
        logger.info(f"Mensaje recibido: {last_message}")

        # Dejar que el LLM decida qué herramienta usar basado en el prompt
        response = graph_builder.invoke(
            {"messages": messages_list},
            config={"configurable": {"thread_id": str(uuid.uuid4())}}
        )

        response_parts = []
        has_plot = False
        plot_data = None
        
        # Si el LLM generó un gráfico, extraer los datos
        response_content = response["messages"][-1].content
        has_plot = "figure" in response_content 
        # if isinstance(response_content, dict) else False
        if has_plot:
            ticker = extract_ticker(last_message)
            chart_data = _generate_stock_chart(ticker)
            if chart_data["success"]:
                has_plot = True
                plot_data = chart_data["figure"]
                response_parts.append(chart_data["text"])
                
                # Combinar todas las respuestas
                combined_response = "\n\n".join(response_parts)

                return {
                    "response": combined_response,
                    "has_plot": has_plot,
                    "plot_data": plot_data
                }
        else:
            return {
                "response": response["messages"][-1].content,
                "has_plot": False,
                "plot_data": None
            }
        # plot_data = response_content.get("figure", None) if has_plot else None

    except Exception as e:
        logger.error(f"Error en chat_endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )

#     except Exception as e:
#         logger.error(f"Error invoking the agent: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/whatsapp")
async def whatsapp_webhook(request: Request) -> Response:
    """
    Endpoint para manejar mensajes entrantes de WhatsApp vía Twilio.
    """
    form_data = await request.form()
    from_number = form_data.get("From")
    user_message = form_data.get("Body")

    if not from_number or not user_message:
        resp = MessagingResponse()
        resp.message("No recibí tu número o tu mensaje. Intenta de nuevo.")
        return Response(content=str(resp), media_type="application/xml")

    if from_number not in SESSIONS:
        SESSIONS[from_number] = {"messages": []}

    SESSIONS[from_number]["messages"].append({"role": "user", "content": user_message})

    response = graph_builder.invoke(
        {"messages": SESSIONS[from_number]["messages"]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}}
    )

    llm_msg = response["messages"][-1].content
    SESSIONS[from_number]["messages"].append({"role": "assistant", "content": llm_msg})

    twilio_resp = MessagingResponse()
    twilio_resp.message(llm_msg)
    return Response(content=str(twilio_resp), media_type="application/xml")