# AI Agent — Assignments 1 & 2

## Overview

A two-part project building toward a full multi-agent system.

Assignment 1 is a single-file agent with LLM-based routing, external tools, and persistent memory.

Assignment 2 extends it using the OpenAI Agents SDK with proper agent architecture, handoffs, guardrails, and a persona.

## Assignment 2 — Multi-Agent System

### Architecture

```
User Input (Gradio UI)
    │
    ▼
Input Guardrail (safety check)
    │
    ▼
RouterAgent (gpt-5.4-mini, Few-Shot)
    │
    ├── handoff → WeatherAgent → get_weather tool
    ├── handoff → MathAgent → calculate_math tool
    ├── handoff → ExchangeAgent → get_exchange_rate tool
    └── handoff → ChatAgent → persona (cynical assistant)
                      │
                  Output Guardrail
    │
    ▼
Response + Saved to history.json
```

### What's new

- OpenAI Agents SDK with Agent, Runner, handoffs
- Few-Shot Router with border case examples
- Structured Output via Pydantic, no manual JSON parsing
- Input and Output Guardrails
- ChatAgent persona - cynical but helpful

### Running

```bash
echo "OPENAI_API_KEY=sk-..." >> .env
python agents_v2.py
```

Then open http://localhost:7860

### Unit Tests

```bash
python -m pytest test_agents_v2.py -v
```

### Files

```
ai_agent_project/
├── agents_v2.py      # main file for assignment 2
├── tools.py          # shared tools
├── prompts.py        # agent prompts
├── test_agents_v2.py
└── screenshots/
```

---

## Assignment 1 — Orchestration of Tool Use

An intelligent agent system built with LLM-based routing, external tool integration,
and persistent conversation memory. The agent classifies each user request and delegates
it to the appropriate tool — or chains multiple tools together to answer complex questions.

The interface is built with **Gradio**, providing a browser-based chat UI.

## Architecture

```
User Input (Gradio UI)
    │
    ▼
┌─────────────────────────────────┐
│   Router  (llama-3.3-70b/Groq)  │  ← Classifies intent, outputs JSON
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   Dispatcher                    │  ← Calls the selected tool
└─────────────────────────────────┘
    │
    ├── getWeather(city)          → wttr.in (free, no key)
    ├── calculateMath(expression) → Python eval (deterministic)
    ├── getExchangeRate(currency) → frankfurter.app (free, no key)
    └── generalChat(...)          → llama-3.3-70b via Groq (free)
    │
    ▼
Response + Saved to history.json
```

## Tools

| Tool | Description | Data Source |
|------|-------------|-------------|
| `getWeather(city)` | Current weather for any city | [wttr.in](https://wttr.in) |
| `calculateMath(expression)` | Evaluates math expressions | Python `eval` (sandboxed) |
| `getExchangeRate(currency_code)` | Exchange rate vs ILS | [frankfurter.app](https://frankfurter.app) |
| `generalChat(history, user_input)` | Open-ended conversation | llama-3.3-70b (Groq) |

## Agentic Patterns Used

- **Routing** — LLM classifies intent and selects the correct tool
- **Tool Use** — deterministic functions for weather, math, and currency
- **Multi-Tool Orchestration** — chains multiple tools for complex queries
- **Persistence** — conversation history saved across sessions via `history.json`

## Prerequisites

- Python 3.10+
- A free Groq API key from [console.groq.com](https://console.groq.com)
- Internet connection (for all tools and the Groq API)

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Add your Groq API key to a .env file
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

## Usage

```bash
python agent.py
```

Then open your browser at **http://localhost:7860**

### Special Commands

| Command | Description |
|---------|-------------|
| `/reset` | Clear conversation history and start a fresh session |

## Example Interactions

### Single Tool Calls

```
You: How hot is it in Tel Aviv?
  [tool] getWeather | params: {'city': 'Tel Aviv'}
Agent: Weather in Al Mas`Udiya: Sunny. Temperature: 31°C (feels like 32°C). Humidity: 29%. Wind: 10 km/h.

You: What is 150 plus 20?
  [tool] calculateMath | params: {'expression': '150 + 20'}
Agent: Result of 150 + 20 = 170

You: What is 15% of 200?
  [tool] calculateMath | params: {'expression': '15% of 200'}
Agent: Result of 15% of 200 = 30

You: What is the dollar rate?
  [tool] getExchangeRate | params: {'currency_code': 'USD'}
Agent: US Dollar (USD) rate: 3.0047 ILS (as of 2026-04-15)

You: How are you?
  [tool] generalChat | params: {}
Agent: I'm just a computer program, so I don't have feelings, but I'm ready to help!
```

### Multi-Tool Orchestration

```
You: How many euros can I buy with 100 dollars?
Detected multi-tool question — orchestrating...
  [tool] getExchangeRate | params: {'currency_code': 'USD'}
  [tool] getExchangeRate | params: {'currency_code': 'EUR'}
  [tool] calculateMath   | params: {'expression': '100.0 * 3.0047 / 3.5395'}
Agent: Result of 100.0 * 3.0047 / 3.5395 = 84.8905

You: If I have 500 shekels, how many British pounds can I get?
Detected multi-tool question — orchestrating...
  [tool] getExchangeRate | params: {'currency_code': 'GBP'}
  [tool] calculateMath   | params: {'expression': '500.0 / 4.0714'}
Agent: Result of 500.0 / 4.0714 = 122.808

You: How much hotter is Dubai than Stockholm?
Detected multi-tool question — orchestrating...
  [tool] getWeather | params: {'city': 'Dubai'}
  [tool] getWeather | params: {'city': 'Stockholm'}
  [tool] calculateMath | params: {'expression': '25.0 - 5.0'}
Agent: Result of 25.0 - 5.0 = 20

You: Is Dubai hotter than Stockholm?
Detected multi-tool question — orchestrating...
  [tool] getWeather | params: {'city': 'Dubai'}
  [tool] getWeather | params: {'city': 'Stockholm'}
  [tool] calculateMath | params: {'expression': '25.0 - 5.0'}
Agent: Yes, Dubai is 20.0°C hotter than Stockholm.
```

### Persistence

```
# First session
You: My name is Shay, remember this.
Agent: I have stored the information that your name is Shay.

# After restart — agent prints: "Welcome back! Loaded previous conversation history."
You: What is my name?
Agent: Your name is Shay.

# After /reset — agent prints: "History cleared. Starting a new conversation."
You: What is my name?
Agent: I don't know your name. You didn't tell me.
```

## Notes

- Weather results may show a neighborhood or district name rather than the exact city name — this is normal behavior from the wttr.in API.
- Exchange rates are live and will differ from the examples above.
- The agent operates in English only.
- Full conversation history is persisted to `history.json` on disk and restored on restart.

## Project Structure

```
ai_agent_project/
├── agent.py            # assignment 1 - single-file agent
├── agents_v2.py        # assignment 2 - multi-agent system
├── tools.py            # shared tools and persistence
├── prompts.py          # all agent prompts (assignment 2)
├── test_agents_v2.py   # unit tests
├── requirements.txt
├── README.md
└── screenshots/        # test run screenshots
```
