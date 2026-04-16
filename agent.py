"""
Assignment 1 — Orchestration of Tool Use
Agent with LLM routing, external tools, and persistent memory.
Model: llama-3.3-70b via Groq
"""

import json
import os
import re
import requests
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
MODEL = "llama-3.3-70b-versatile"
HISTORY_FILE = "history.json"


def call_llm(messages: list, temperature: float = 1.0) -> str:
    """Send messages to Groq and return the response text."""
    clean = [{"role": m["role"], "content": m["content"]} for m in messages]
    response = client.chat.completions.create(model=MODEL, messages=clean, temperature=temperature)
    return response.choices[0].message.content


# --- Persistence ---

def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            print("Welcome back! Loaded previous conversation history.")
            return history
        except (json.JSONDecodeError, IOError):
            print("Warning: history file corrupted — starting fresh.")
            return []
    print("Hello! Starting a new conversation.")
    return []


def save_history(history: list) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Warning: could not save history — {e}")


def reset_history() -> list:
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    print("History cleared. Starting a new conversation.")
    return []


# --- Weather Tool ---

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
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return f"Error: Could not parse weather response for '{city}'. ({e})"
    except requests.exceptions.HTTPError as e:
        if "500" in str(e) or "404" in str(e):
            return f"Could not find weather data for '{city}'. Please check the city name."
        return f"Error fetching weather for '{city}': {e}"
    except Exception:
        return f"Could not find weather data for '{city}'. Please check the city name."


# --- Math Tool ---

def calculateMath(expression: str) -> str:
    """Evaluate a math expression deterministically using Python eval (sandboxed)."""
    cleaned = re.sub(r"[^\d\s\+\-\*\/\(\)\.\,\%\^]", "", expression)
    cleaned = cleaned.replace("^", "**").replace(",", ".")

    # Convert "X% of Y" and "X% * Y" patterns to "(X/100) * Y"
    cleaned = re.sub(r'(\d+\.?\d*)\s*%\s*(of\s*)?(\d+\.?\d*)', r'(\1/100)*\3', cleaned)
    # Convert standalone "X%" to "X/100"
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
        return f"Error: Invalid expression — '{expression}'."
    except Exception as e:
        return f"Calculation error: {e}"


# --- Exchange Rate Tool ---

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


def getExchangeRate(currency_code: str) -> str:
    """Return the exchange rate of a currency vs ILS using frankfurter.app."""
    currency_code = currency_code.strip().upper()
    try:
        url = f"https://api.frankfurter.app/latest?from={currency_code}&to=ILS"
        data = requests.get(url, timeout=8).json()

        if "rates" not in data or "ILS" not in data.get("rates", {}):
            # ILS is not supported as base — reverse the query
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


# --- General Chat ---

def generalChat(history: list, user_input: str) -> str:
    """Handle any input not matched by the other tools — free conversation."""
    system = {
        "role": "system",
        "content": "You are a helpful and concise assistant. Answer clearly and honestly."
    }
    messages = [system] + history + [{"role": "user", "content": user_input}]
    try:
        return call_llm(messages)
    except Exception as e:
        return f"Error running model: {e}\nMake sure your GROQ_API_KEY is set correctly."


# --- Router ---

ROUTER_SYSTEM_PROMPT = """You are a routing engine for an AI assistant system.
Your only job is to analyze the user's input and decide which tool to invoke.

Available tools:
1. getWeather       — weather, temperature, forecast for any city
2. calculateMath    — arithmetic, percentages, any mathematical calculation
3. getExchangeRate  — currency exchange rates (USD, EUR, GBP, etc.)
4. generalChat      — everything else (greetings, opinions, general questions)

Respond ONLY with valid JSON — no markdown, no explanation, no extra text:
{"tool": "tool_name", "params": {"param1": "value1"}}

Examples:
User: "How hot is it in Tel Aviv?" → {"tool": "getWeather", "params": {"city": "Tel Aviv"}}
User: "What is 50 times 3?"        → {"tool": "calculateMath", "params": {"expression": "50 * 3"}}
User: "What is the dollar rate?"   → {"tool": "getExchangeRate", "params": {"currency_code": "USD"}}
User: "How are you?"               → {"tool": "generalChat", "params": {}}
"""


def route_request(user_input: str) -> dict:
    """Ask the LLM to classify the request and return the appropriate tool + params."""
    WEATHER_KEYWORDS = {
        "weather", "temperature", "hot", "cold", "warm", "cool",
        "rain", "raining", "rainy", "sunny", "cloudy", "forecast",
        "degrees", "humid", "humidity", "wind", "windy", "snow",
        "snowing", "storm", "stormy", "climate", "heat", "freeze",
        "freezing", "foggy", "fog", "hail", "thunder", "lightning"
    }

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    try:
        raw = call_llm(messages, temperature=0).strip()
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            routing = json.loads(json_match.group())
            # Block getWeather if no explicit weather keyword is present
            if routing.get("tool") == "getWeather":
                if not set(user_input.lower().split()) & WEATHER_KEYWORDS:
                    return {"tool": "generalChat", "params": {}}
            return routing
        return {"tool": "generalChat", "params": {}}

    except (json.JSONDecodeError, AttributeError):
        return {"tool": "generalChat", "params": {}}
    except Exception as e:
        print(f"Warning: Router error — {e}")
        return {"tool": "generalChat", "params": {}}


# --- Dispatcher ---

def dispatch(routing: dict, history: list, user_input: str) -> str:
    """Execute the tool selected by the router."""
    tool = routing.get("tool", "generalChat")
    params = routing.get("params", {})
    print(f"  [tool] {tool} | params: {params}")

    if tool == "getWeather":
        return getWeather(params.get("city", user_input))
    elif tool == "calculateMath":
        return calculateMath(params.get("expression", user_input))
    elif tool == "getExchangeRate":
        return getExchangeRate(params.get("currency_code", "USD"))
    elif tool == "generalChat":
        return generalChat(history, user_input)
    else:
        print(f"  Warning: unknown tool '{tool}' — falling back to generalChat")
        return generalChat(history, user_input)


# --- Multi-Tool Orchestrator ---

MULTI_TOOL_SYSTEM_PROMPT = """You are a task orchestrator. Synthesize the collected tool
results into a single clear and concise final answer."""


def needs_multiple_tools(user_input: str) -> bool:
    """Return True if the question likely requires chaining more than one tool."""
    patterns = [
        r"(how many|how much).*(buy|get|purchase|exchange).*(with|for)",
        r"(how many|how much).*(can i get|will i get|do i get)",
        r"(times|how much more|hotter|colder|warmer|cheaper|expensive).*(than|compared)",
        r"(hotter|colder|warmer|cheaper|more expensive).*(than|compared to)",
        r"(convert|exchange).*(to|into)",
        r"\d+\s*(usd|eur|gbp|dollar|euro).*(buy|get|purchase|exchange).*(eur|ils|jpy|usd|shekel)",
        r"(difference|ratio|between).*(weather|temperature|rate|price)",
        r"[2-9]\d*\s*(shekel|shekels|ils).*(dollar|euro|pound|yen|franc|usd|eur|gbp|jpy|chf)",
        r"[2-9]\d*\s*(dollar|euro|pound|yen|franc|usd|eur|gbp|jpy|chf).*(shekel|shekels|ils)",
    ]
    return any(re.search(p, user_input, re.IGNORECASE) for p in patterns)


def extract_number(text: str) -> float | None:
    """Extract the first number from a tool result string."""
    match = re.search(r"[-+]?\d+\.?\d*", text)
    return float(match.group()) if match else None


def multi_tool_orchestrate(user_input: str, history: list) -> str:
    """
    Handle questions that require chaining multiple tools.
    Currency and weather comparisons are handled with deterministic Python logic
    to guarantee numeric precision — the LLM is only used for synthesis.
    """
    collected = []
    lower = user_input.lower()

    currency_codes = {
        "dollar": "USD", "dollars": "USD", "usd": "USD",
        "euro":   "EUR", "euros":   "EUR", "eur": "EUR",
        "pound":  "GBP", "pounds":  "GBP", "gbp": "GBP",
        "yen":    "JPY", "jpy": "JPY",
        "shekel": "ILS", "shekels": "ILS", "ils": "ILS",
        "franc":  "CHF", "francs":  "CHF", "chf": "CHF",
    }

    # Identify currencies by position in the sentence
    currency_positions = []
    for word, code in currency_codes.items():
        match = re.search(rf'\b{word}\b', lower)
        if match and code not in [c for _, c in currency_positions]:
            currency_positions.append((match.start(), code))
    currency_positions.sort(key=lambda x: x[0])
    found_currencies = [code for _, code in currency_positions]

    # "how many X can I buy WITH Y" → Y is the source currency
    with_match = re.search(r'how many\s+(\w+)\s+.{0,30}\bwith\b\s+.{0,10}(\w+)', lower)
    if with_match and len(found_currencies) >= 2:
        found_currencies = list(reversed(found_currencies))

    amount_match = re.search(r'\b(\d+\.?\d*)\b', user_input)
    amount = float(amount_match.group(1)) if amount_match else None

    # Case 1: Currency conversion
    if len(found_currencies) >= 2 and amount:
        from_currency = found_currencies[0]
        to_currency = found_currencies[1]

        if from_currency != "ILS":
            result1 = getExchangeRate(from_currency)
            collected.append(f"Step 1 (getExchangeRate): {result1}")
            rate1 = extract_number(result1)
        else:
            rate1 = 1.0  # ILS is already shekels

        if to_currency != "ILS":
            result2 = getExchangeRate(to_currency)
            collected.append(f"Step 2 (getExchangeRate): {result2}")
            rate2 = extract_number(result2)
        else:
            rate2 = 1.0

        if rate1 and rate2:
            if from_currency == "ILS":
                expression = f"{amount} / {rate2}"
            elif to_currency == "ILS":
                expression = f"{amount} * {rate1}"
            else:
                expression = f"{amount} * {rate1} / {rate2}"
            result3 = calculateMath(expression)
            collected.append(f"Step 3 (calculateMath): {result3}")

    # Case 2: Temperature comparison
    elif any(w in lower for w in ["hotter", "colder", "warmer", "temperature", "degrees"]) \
         and any(w in lower for w in ["than", "compared", "vs", "versus"]):

        city_prompt = (
            f"Extract exactly two city names from this question as JSON: {user_input}\n"
            f"Respond only with: {{\"city1\": \"...\", \"city2\": \"...\"}}"
        )
        try:
            city_raw = call_llm([{"role": "user", "content": city_prompt}], temperature=0).strip()
            city_match = re.search(r'\{.*\}', city_raw, re.DOTALL)
            cities = json.loads(city_match.group()) if city_match else {}
            city1 = cities.get("city1", "")
            city2 = cities.get("city2", "")
        except Exception:
            city1, city2 = "", ""

        if city1 and city2:
            result1 = getWeather(city1)
            collected.append(f"Step 1 (getWeather {city1}): {result1}")
            temp1 = extract_number(result1)

            result2 = getWeather(city2)
            collected.append(f"Step 2 (getWeather {city2}): {result2}")
            temp2 = extract_number(result2)

            if temp1 and temp2:
                result3 = calculateMath(f"{temp1} - {temp2}")
                collected.append(f"Step 3 (calculateMath): {result3}")

    if not collected:
        return generalChat(history, user_input)

    # Build final answer from Python-computed result — no LLM arithmetic
    if any("calculateMath" in s for s in collected):
        calc_result = next(
            s.split("calculateMath): ")[1]
            for s in reversed(collected) if "calculateMath" in s
        )
        num_match = re.search(r'=\s*([-\d.]+)', calc_result)
        num = float(num_match.group(1)) if num_match else None

        # Yes/No comparison questions get a natural language answer
        is_comparison = re.search(
            r'\b(is|are)\b.*(hotter|colder|warmer|cheaper|more expensive|higher|lower)',
            user_input, re.IGNORECASE
        )
        if is_comparison and num is not None:
            city_prompt = (
                f"Extract exactly two city names from this question as JSON: {user_input}\n"
                f"Respond only with: {{\"city1\": \"...\", \"city2\": \"...\"}}"
            )
            try:
                city_raw = call_llm([{"role": "user", "content": city_prompt}], temperature=0).strip()
                city_match = re.search(r'\{.*\}', city_raw, re.DOTALL)
                cities = json.loads(city_match.group()) if city_match else {}
                city1 = cities.get("city1", "City 1")
                city2 = cities.get("city2", "City 2")
            except Exception:
                city1, city2 = "City 1", "City 2"

            if num > 0:
                return f"Yes, {city1} is {abs(num):.1f}°C hotter than {city2}."
            elif num < 0:
                return f"No, {city2} is {abs(num):.1f}°C hotter than {city1}."
            else:
                return "They are the same temperature."

        return calc_result

    # For weather-only results — ask LLM to synthesize
    summary_prompt = (
        f"Original user question: {user_input}\n\n"
        f"Data collected:\n{chr(10).join(collected)}\n\n"
        f"Use ONLY the numbers above. Provide a clear, concise answer:"
    )
    try:
        return call_llm([
            {"role": "system", "content": MULTI_TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": summary_prompt}
        ])
    except Exception:
        return "\n".join(collected)


# --- Gradio ---

conversation_history = load_history()


def chat(user_input: str, gradio_history: list) -> str:
    """
    Gradio callback — routes each message to the correct tool and persists history.
    Uses gradio_history as LLM context during an active session.
    On first message after restart, falls back to the full history from disk.
    """
    global conversation_history

    if user_input.strip().lower() == "/reset":
        conversation_history = reset_history()
        return "History cleared. Starting a new conversation."

    if gradio_history:
        session_history = [{"role": h["role"], "content": h["content"]} for h in gradio_history]
    else:
        session_history = conversation_history

    if needs_multiple_tools(user_input):
        print("Detected multi-tool question — orchestrating...")
        response = multi_tool_orchestrate(user_input, session_history)
    else:
        routing = route_request(user_input)
        response = dispatch(routing, session_history, user_input)

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": response})
    save_history(conversation_history)

    return response


if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat,
        title="AI Agent — Orchestration of Tools",
        description=(
            "Ask about **weather**, **currency rates**, or **math**. "
            "Type `/reset` to clear conversation history."
        ),
        examples=[
            "How hot is it in Tel Aviv?",
            "What is the dollar rate?",
            "What is 150 plus 20?",
            "How many euros can I buy with 100 dollars?",
            "How much hotter is Dubai than Stockholm?",
        ],
    ).launch()
