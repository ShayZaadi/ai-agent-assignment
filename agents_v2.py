"""
Assignment 2 - multi-agent system using the OpenAI Agents SDK.
Extends assignment 1 with proper agent architecture, guardrails, and handoffs.
"""

import gradio as gr
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    function_tool,
    GuardrailFunctionOutput,
    InputGuardrail,
    OutputGuardrail,
    RunContextWrapper,
    handoff,
    trace,
    gen_trace_id,
)
from tools import (
    getWeather, calculateMath, getExchangeRate,
    load_history, save_history, reset_history,
)
from prompts import (
    ROUTER_INSTRUCTIONS, WEATHER_INSTRUCTIONS, MATH_INSTRUCTIONS,
    EXCHANGE_INSTRUCTIONS, CHAT_INSTRUCTIONS,
    INPUT_GUARDRAIL_INSTRUCTIONS, OUTPUT_GUARDRAIL_INSTRUCTIONS,
)

load_dotenv(override=True)

MODEL = "gpt-5.4-mini"


# tool wrappers for the SDK

@function_tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return getWeather(city)

@function_tool
def calculate_math(expression: str) -> str:
    """Evaluate a math expression."""
    return calculateMath(expression)

@function_tool
def get_exchange_rate(currency_code: str) -> str:
    """Get exchange rate vs ILS."""
    return getExchangeRate(currency_code)


# structured output models for guardrails

class InputCheck(BaseModel):
    is_safe: bool
    reason: str

class OutputCheck(BaseModel):
    is_acceptable: bool
    reason: str


# guardrail agents

input_guardrail_agent = Agent(
    name="InputGuardrailAgent",
    instructions=INPUT_GUARDRAIL_INSTRUCTIONS,
    output_type=InputCheck,
    model=MODEL,
)

output_guardrail_agent = Agent(
    name="OutputGuardrailAgent",
    instructions=OUTPUT_GUARDRAIL_INSTRUCTIONS,
    output_type=OutputCheck,
    model=MODEL,
)


async def input_guardrail_fn(
    ctx: RunContextWrapper, agent: Agent, user_input: str
) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrail_agent, user_input, context=ctx.context)
    check = result.final_output_as(InputCheck)
    return GuardrailFunctionOutput(output_info=check, tripwire_triggered=not check.is_safe)


async def output_guardrail_fn(
    ctx: RunContextWrapper, agent: Agent, output: str
) -> GuardrailFunctionOutput:
    result = await Runner.run(output_guardrail_agent, output, context=ctx.context)
    check = result.final_output_as(OutputCheck)
    return GuardrailFunctionOutput(output_info=check, tripwire_triggered=not check.is_acceptable)


# task agents

weather_agent = Agent(
    name="WeatherAgent",
    instructions=WEATHER_INSTRUCTIONS,
    tools=[get_weather],
    model=MODEL,
)

math_agent = Agent(
    name="MathAgent",
    instructions=MATH_INSTRUCTIONS,
    tools=[calculate_math],
    model=MODEL,
)

exchange_agent = Agent(
    name="ExchangeAgent",
    instructions=EXCHANGE_INSTRUCTIONS,
    tools=[get_exchange_rate],
    model=MODEL,
)

chat_agent = Agent(
    name="ChatAgent",
    instructions=CHAT_INSTRUCTIONS,
    model=MODEL,
    output_guardrails=[OutputGuardrail(guardrail_function=output_guardrail_fn)],
)


# router - entry point for all requests

router_agent = Agent(
    name="RouterAgent",
    instructions=ROUTER_INSTRUCTIONS,
    model=MODEL,
    handoffs=[
        handoff(weather_agent),
        handoff(math_agent),
        handoff(exchange_agent),
        handoff(chat_agent),
    ],
    input_guardrails=[InputGuardrail(guardrail_function=input_guardrail_fn)],
)


# gradio interface

conversation_history = load_history()


async def chat(user_input: str, gradio_history: list) -> str:
    global conversation_history

    if user_input.strip().lower() == "/reset":
        conversation_history = reset_history()
        return "History cleared. Starting a new conversation."

    # build context from recent messages
    history_text = ""
    if gradio_history:
        for msg in gradio_history[-10:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                history_text += f"{role.capitalize()}: {content}\n"

    full_input = f"{history_text}User: {user_input}" if history_text else user_input

    response = "Something went wrong. Please try again."

    try:
        trace_id = gen_trace_id()
        with trace("agent-conversation", trace_id=trace_id):
            print(f"[trace] https://platform.openai.com/traces/trace?trace_id={trace_id}")
            result = await Runner.run(router_agent, full_input)
            response = result.final_output

            # print structured output for logging
            for item in result.new_items:
                if hasattr(item, "raw_item") and hasattr(item.raw_item, "content"):
                    for block in item.raw_item.content or []:
                        if hasattr(block, "text") and block.text:
                            print(f"[output] {block.text[:200]}")

    except Exception as e:
        if "tripwire" in str(e).lower() or "guardrail" in str(e).lower():
            response = "I can't help with that request."
        else:
            response = f"Error: {e}"

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": response})
    save_history(conversation_history)

    return response


if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat,
        title="AI Agent v2 - Multi-Agent System",
        description=(
            "Powered by OpenAI Agents SDK. "
            "Ask about weather, math, currency rates, or anything else. "
            "Type /reset to clear history."
        ),
        examples=[
            "How hot is it in Tel Aviv?",
            "What is 15% of 340?",
            "What is the dollar rate?",
            "Yossi had 5 apples, ate 2, then bought 10. How many does he have?",
            "Tell me a fun fact about space",
        ],
    ).launch()
