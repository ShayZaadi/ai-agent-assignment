"""
Shared tools and persistence layer.
Used by agent.py (assignment 1) and agents_v2.py (assignment 2).
"""

import json
import os
import re
import requests

HISTORY_FILE = "history.json"

CURRENCY_NAMES = {
    "USD": "US Dollar",
    "EUR": "Euro",
    "GBP": "British Pound",
    "JPY": "Japanese Yen",
    "CHF": "Swiss Franc",
    "CAD": "Canadian Dollar",
    "AUD": "Australian Dollar",
    "CNY": "Chinese Yuan",
    "SEK": "Swedish Krona",
    "NOK": "Norwegian Krone",
    "DKK": "Danish Krone",
    "TRY": "Turkish Lira",
    "AED": "UAE Dirham",
    "JOD": "Jordanian Dinar"
}


# persistence

def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            print("Welcome back! Loaded previous conversation history.")
            return history
        except (json.JSONDecodeError, IOError):
            print("Warning: history file corrupted - starting fresh.")
            return []
    print("Hello! Starting a new conversation.")
    return []


def save_history(history: list) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Warning: could not save history - {e}")


def reset_history() -> list:
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    print("History cleared. Starting a new conversation.")
    return []


# weather

def getWeather(city: str) -> str:
    """Fetch current weather for a city from wttr.in."""
    try:
        url = f"https://wttr.in/{requests.utils.quote(city)}?format=j1"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()

        current = data["current_condition"][0]
        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        humidity = current["humidity"]
        desc = current["weatherDesc"][0]["value"]
        wind_kmph = current["windspeedKmph"]
        area_name = data.get("nearest_area", [{}])[0].get("areaName", [{}])[0].get("value", city)

        return (
            f"Weather in {area_name}: {desc}. "
            f"Temperature: {temp_c}°C (feels like {feels_like}°C). "
            f"Humidity: {humidity}%. Wind: {wind_kmph} km/h."
        )
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the weather service."
    except requests.exceptions.Timeout:
        return f"Error: Weather service timed out for '{city}'."
    except requests.exceptions.HTTPError as e:
        if "500" in str(e) or "404" in str(e):
            return f"Could not find weather data for '{city}'. Please check the city name."
        return f"Error fetching weather for '{city}': {e}"
    except Exception:
        return f"Could not find weather data for '{city}'. Please check the city name."


# math

def calculateMath(expression: str) -> str:
    """Evaluate a math expression deterministically using sandboxed eval."""
    cleaned = re.sub(r"[^\d\s\+\-\*\/\(\)\.\,\%\^]", "", expression)
    cleaned = cleaned.replace("^", "**").replace(",", ".")
    cleaned = re.sub(r'(\d+\.?\d*)\s*%\s*(of\s*)?(\d+\.?\d*)', r'(\1/100)*\3', cleaned)
    cleaned = re.sub(r'(\d+\.?\d*)\s*%', r'(\1/100)', cleaned)

    if not cleaned.strip():
        return f"Error: '{expression}' is not a valid mathematical expression."
    try:
        safe_globals = {"__builtins__": {}}
        safe_locals = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow, "sum": sum}
        result = eval(cleaned, safe_globals, safe_locals)
        result_str = f"{round(result, 6):g}" if isinstance(result, float) else str(result)
        return f"Result of {expression} = {result_str}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except SyntaxError:
        return f"Error: Invalid expression - '{expression}'."
    except Exception as e:
        return f"Calculation error: {e}"


# exchange rate

def getExchangeRate(currency_code: str) -> str:
    """Return the exchange rate of a currency vs ILS using frankfurter.app."""
    currency_code = currency_code.strip().upper()
    try:
        url = f"https://api.frankfurter.app/latest?from={currency_code}&to=ILS"
        data = requests.get(url, timeout=8).json()

        if "rates" not in data or "ILS" not in data.get("rates", {}):
            url2 = f"https://api.frankfurter.app/latest?from=ILS&to={currency_code}"
            data2 = requests.get(url2, timeout=8).json()
            if "rates" in data2 and currency_code in data2["rates"]:
                rate = round(1 / data2["rates"][currency_code], 4)
                name = CURRENCY_NAMES.get(currency_code, currency_code)
                return f"{name} ({currency_code}) rate: {rate:.4f} ILS (as of {data2['date']})"
            return f"Error: Currency code '{currency_code}' is not supported."

        rate = data["rates"]["ILS"]
        name = CURRENCY_NAMES.get(currency_code, currency_code)
        return f"{name} ({currency_code}) rate: {rate:.4f} ILS (as of {data['date']})"

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the exchange rate service."
    except requests.exceptions.Timeout:
        return f"Error: Exchange rate service timed out for '{currency_code}'."
    except Exception as e:
        return f"Error fetching exchange rate: {e}"
