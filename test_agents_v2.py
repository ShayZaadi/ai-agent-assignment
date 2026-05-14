"""
Unit tests for assignment 2 - multi-agent system.
Tests cover: tools, guardrails, routing, persistence, and integration.

Run with:
    python -m pytest test_agents_v2.py -v
"""

import json
import os
import asyncio
import pytest
from unittest.mock import patch, MagicMock
from tools import calculateMath, getWeather, getExchangeRate, load_history, save_history, reset_history


# math tool tests

class TestCalculateMath:

    def test_addition(self):
        result = calculateMath("150 + 20")
        assert "170" in result

    def test_multiplication(self):
        result = calculateMath("50 * 3 / 2")
        assert "75" in result

    def test_power(self):
        result = calculateMath("2 ^ 10")
        assert "1024" in result

    def test_percentage(self):
        result = calculateMath("15% of 200")
        assert "30" in result

    def test_percentage_simple(self):
        result = calculateMath("50%")
        assert "0.5" in result

    def test_division_by_zero(self):
        result = calculateMath("100 / 0")
        assert "zero" in result.lower()

    def test_invalid_expression(self):
        result = calculateMath("abc + xyz")
        assert "error" in result.lower() or "Result" in result

    def test_decimal(self):
        result = calculateMath("1.5 * 4")
        assert "6" in result

    def test_parentheses(self):
        result = calculateMath("(2 + 3) * 4")
        assert "20" in result


# weather tool tests

class TestGetWeather:

    @patch("tools.requests.get")
    def test_valid_city(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "current_condition": [{
                "temp_C": "25",
                "FeelsLikeC": "26",
                "humidity": "60",
                "weatherDesc": [{"value": "Sunny"}],
                "windspeedKmph": "15"
            }],
            "nearest_area": [{"areaName": [{"value": "Tel Aviv"}]}]
        }
        result = getWeather("Tel Aviv")
        assert "25" in result
        assert "Sunny" in result

    @patch("tools.requests.get")
    def test_invalid_city_500(self, mock_get):
        mock_get.return_value.raise_for_status.side_effect = Exception("500 Server Error")
        result = getWeather("XYZABC123")
        assert "Could not find" in result or "Error" in result

    @patch("tools.requests.get")
    def test_connection_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError()
        result = getWeather("London")
        assert "Could not connect" in result

    @patch("tools.requests.get")
    def test_timeout(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.Timeout()
        result = getWeather("Paris")
        assert "timed out" in result.lower()


# exchange rate tool tests

class TestGetExchangeRate:

    @patch("tools.requests.get")
    def test_valid_currency(self, mock_get):
        mock_get.return_value.json.return_value = {
            "rates": {"ILS": 3.75},
            "date": "2026-04-15"
        }
        result = getExchangeRate("USD")
        assert "3.75" in result
        assert "ILS" in result
        assert "2026-04-15" in result

    @patch("tools.requests.get")
    def test_unsupported_currency(self, mock_get):
        mock_get.return_value.json.return_value = {}
        result = getExchangeRate("XYZ")
        assert "not supported" in result.lower() or "Error" in result

    @patch("tools.requests.get")
    def test_lowercase_input(self, mock_get):
        mock_get.return_value.json.return_value = {
            "rates": {"ILS": 4.10},
            "date": "2026-04-15"
        }
        result = getExchangeRate("gbp")
        assert "ILS" in result

    @patch("tools.requests.get")
    def test_connection_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError()
        result = getExchangeRate("EUR")
        assert "Could not connect" in result


# persistence tests

class TestPersistence:

    TEST_FILE = "test_history.json"

    def setup_method(self):
        
        import tools
        tools.HISTORY_FILE = self.TEST_FILE

    def teardown_method(self):
        import tools
        tools.HISTORY_FILE = "history.json"
        if os.path.exists(self.TEST_FILE):
            os.remove(self.TEST_FILE)

    def test_load_empty(self):
        if os.path.exists(self.TEST_FILE):
            os.remove(self.TEST_FILE)
        from tools import load_history
        history = load_history()
        assert history == []

    def test_save_and_load(self):
        from tools import save_history, load_history
        data = [{"role": "user", "content": "hello"}]
        save_history(data)
        loaded = load_history()
        assert loaded == data

    def test_reset(self):
        from tools import save_history, reset_history, load_history
        save_history([{"role": "user", "content": "test"}])
        reset_history()
        assert not os.path.exists(self.TEST_FILE)

    def test_corrupted_file(self):
        with open(self.TEST_FILE, "w") as f:
            f.write("not valid json{{{")
        from tools import load_history
        history = load_history()
        assert history == []


# input guardrail tests

class TestInputGuardrailLogic:
    """Checks the guardrail prompt content without calling the API."""

    def test_prompt_blocks_empty(self):
        from prompts import INPUT_GUARDRAIL_INSTRUCTIONS
        assert "empty" in INPUT_GUARDRAIL_INSTRUCTIONS.lower()

    def test_prompt_blocks_malicious(self):
        from prompts import INPUT_GUARDRAIL_INSTRUCTIONS
        assert "malicious" in INPUT_GUARDRAIL_INSTRUCTIONS.lower()

    def test_prompt_allows_political(self):
        from prompts import INPUT_GUARDRAIL_INSTRUCTIONS
        assert "political" in INPUT_GUARDRAIL_INSTRUCTIONS.lower()
        assert "safe" in INPUT_GUARDRAIL_INSTRUCTIONS.lower()


# output guardrail tests

class TestOutputGuardrailLogic:

    def test_prompt_blocks_malicious_output(self):
        from prompts import OUTPUT_GUARDRAIL_INSTRUCTIONS
        assert "malicious" in OUTPUT_GUARDRAIL_INSTRUCTIONS.lower()

    def test_prompt_blocks_empty_output(self):
        from prompts import OUTPUT_GUARDRAIL_INSTRUCTIONS
        assert "empty" in OUTPUT_GUARDRAIL_INSTRUCTIONS.lower()

    def test_prompt_allows_sarcasm(self):
        from prompts import OUTPUT_GUARDRAIL_INSTRUCTIONS
        assert "sarcastic" in OUTPUT_GUARDRAIL_INSTRUCTIONS.lower()


# router prompt tests

class TestRouterPrompt:

    def test_all_agents_mentioned(self):
        from prompts import ROUTER_INSTRUCTIONS
        for agent in ["weather_agent", "math_agent", "exchange_agent", "chat_agent"]:
            assert agent in ROUTER_INSTRUCTIONS

    def test_few_shot_examples_present(self):
        from prompts import ROUTER_INSTRUCTIONS
        assert "Few-Shot" in ROUTER_INSTRUCTIONS
        assert "Paris" in ROUTER_INSTRUCTIONS
        assert "coat" in ROUTER_INSTRUCTIONS

    def test_handoff_instruction(self):
        from prompts import ROUTER_INSTRUCTIONS
        assert "handoff" in ROUTER_INSTRUCTIONS.lower()
        assert "Do not answer" in ROUTER_INSTRUCTIONS


# pydantic model tests

class TestPydanticModels:

    def test_input_check_valid(self):
        from agents_v2 import InputCheck
        check = InputCheck(is_safe=True, reason="looks fine")
        assert check.is_safe is True
        assert check.reason == "looks fine"

    def test_input_check_unsafe(self):
        from agents_v2 import InputCheck
        check = InputCheck(is_safe=False, reason="malicious request")
        assert check.is_safe is False

    def test_output_check_valid(self):
        from agents_v2 import OutputCheck
        check = OutputCheck(is_acceptable=True, reason="normal response")
        assert check.is_acceptable is True

    def test_output_check_invalid(self):
        from agents_v2 import OutputCheck
        check = OutputCheck(is_acceptable=False, reason="contains malware")
        assert check.is_acceptable is False


# integration tests (require OPENAI_API_KEY)

class TestMathIntegration:
    """End-to-end tests. Skipped if OPENAI_API_KEY is not set."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_math_via_agent(self):
        from agents import Runner
        from agents_v2 import router_agent

        async def run():
            result = await Runner.run(router_agent, "What is 10 plus 5?")
            return result.final_output

        response = asyncio.run(run())
        assert "15" in response

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_guardrail_blocks_empty(self):
        from agents import Runner
        from agents_v2 import router_agent

        async def run():
            try:
                result = await Runner.run(router_agent, "   ")
                return result.final_output
            except Exception as e:
                return str(e)

        response = asyncio.run(run())
        assert "tripwire" in response.lower() or "can't help" in response.lower() or response == ""
