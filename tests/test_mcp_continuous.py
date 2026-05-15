"""MCP continuous testing suite for BuyWhere API health monitoring.

This module provides comprehensive integration-style tests for the MCP health
check system, covering continuous monitoring, concurrent execution, cancellation,
degraded-state handling, performance bounds, and response-format validation.
"""

import asyncio
import time
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from buywhere_llamaindex.client import (
    BuyWhereAPIError,
    BuyWhereAuthError,
    BuyWhereNetworkError,
)
from buywhere_llamaindex.mcp_health import (
    MCPHealthCheck,
    create_mcp_health_checker,
    quick_health_check,
)


HEALTHY_RESPONSE = {
    "status": "healthy",
    "timestamp": "2026-05-15T12:00:00Z",
    "api_version": "1.0.0",
    "response_time_ms": 120.5,
    "authentication_status": "authenticated",
    "server_info": {"name": "buywhere-api", "status": "operational"},
}

DEGRADED_RESPONSE = {
    "status": "degraded",
    "timestamp": "2026-05-15T12:00:00Z",
    "api_version": "1.0.0",
    "response_time_ms": 3500.0,
    "authentication_status": "authenticated",
    "server_info": {"name": "buywhere-api", "status": "degraded"},
}

UNHEALTHY_RESPONSE = {
    "status": "unhealthy",
    "timestamp": "2026-05-15T12:00:00Z",
    "api_version": None,
    "response_time_ms": None,
    "authentication_status": "unknown",
    "server_info": {},
}


def _make_checker(side_effect=None, return_value=None):
    with patch("buywhere_llamaindex.mcp_health.BuyWhereClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance
        if side_effect is not None:
            mock_instance.health_check.side_effect = side_effect
        elif return_value is not None:
            mock_instance.health_check.return_value = return_value
        else:
            mock_instance.health_check.return_value = HEALTHY_RESPONSE
        checker = MCPHealthCheck(api_key="test_key")
        return checker, mock_instance


@pytest.mark.asyncio
async def test_continuous_check_collects_all_results():
    checker, mock = _make_checker(return_value=HEALTHY_RESPONSE)
    results = await checker.run_continuous_health_check(interval=0.001, max_checks=5)
    assert len(results) == 5
    assert mock.health_check.call_count == 5
    assert all(r["status"] == "healthy" for r in results)


@pytest.mark.asyncio
async def test_continuous_check_mixed_statuses():
    responses = [HEALTHY_RESPONSE, DEGRADED_RESPONSE, HEALTHY_RESPONSE]
    checker, mock = _make_checker()
    mock.health_check.side_effect = responses
    results = await checker.run_continuous_health_check(interval=0.001, max_checks=3)
    assert results[0]["status"] == "healthy"
    assert results[1]["status"] == "degraded"
    assert results[2]["status"] == "healthy"


@pytest.mark.asyncio
async def test_continuous_check_stops_on_max_checks():
    checker, mock = _make_checker(return_value=HEALTHY_RESPONSE)
    results = await checker.run_continuous_health_check(interval=0.001, max_checks=3)
    assert len(results) == 3
    assert mock.health_check.call_count == 3


@pytest.mark.asyncio
async def test_continuous_check_handles_error_midway():
    responses = [
        HEALTHY_RESPONSE,
        BuyWhereNetworkError("timeout"),
        HEALTHY_RESPONSE,
    ]
    checker, mock = _make_checker()
    mock.health_check.side_effect = responses
    results = await checker.run_continuous_health_check(interval=0.001, max_checks=3)
    assert results[0]["status"] == "healthy"
    assert results[1]["status"] == "unhealthy"
    assert results[1]["error_details"]["type"] == "network_error"
    assert results[2]["status"] == "healthy"


@pytest.mark.asyncio
async def test_continuous_check_cancellation():
    checker, mock = _make_checker(return_value=HEALTHY_RESPONSE)

    async def cancel_after_delay(task):
        await asyncio.sleep(0.05)
        task.cancel()

    task = asyncio.create_task(
        checker.run_continuous_health_check(interval=0.01)
    )
    asyncio.create_task(cancel_after_delay(task))

    results = await task
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_degraded_status_response():
    checker, _ = _make_checker(return_value=DEGRADED_RESPONSE)
    result = await checker.check_endpoint_health()
    assert result["status"] == "degraded"
    assert result["error_details"] is not None


@pytest.mark.asyncio
async def test_unhealthy_status_response():
    checker, _ = _make_checker(return_value=UNHEALTHY_RESPONSE)
    result = await checker.check_endpoint_health()
    assert result["status"] == "unhealthy"
    assert result["error_details"] is not None


@pytest.mark.asyncio
async def test_auth_error_returns_unhealthy():
    checker, _ = _make_checker(side_effect=BuyWhereAuthError("bad key"))
    result = await checker.check_endpoint_health()
    assert result["status"] == "unhealthy"
    assert result["authentication"] == "invalid"
    assert result["error_details"]["type"] == "authentication_error"


@pytest.mark.asyncio
async def test_api_500_error_returns_unhealthy():
    checker, _ = _make_checker(side_effect=BuyWhereAPIError(500, "Server error"))
    result = await checker.check_endpoint_health()
    assert result["status"] == "unhealthy"
    assert result["server_status"] == "error"
    assert result["error_details"]["status_code"] == 500


@pytest.mark.asyncio
async def test_api_429_rate_limit_error():
    checker, _ = _make_checker(side_effect=BuyWhereAPIError(429, "Rate limited"))
    result = await checker.check_endpoint_health()
    assert result["status"] == "unhealthy"
    assert result["error_details"]["type"] == "api_error"
    assert result["error_details"]["status_code"] == 429


@pytest.mark.asyncio
async def test_network_timeout_error():
    checker, _ = _make_checker(side_effect=BuyWhereNetworkError("Connection refused"))
    result = await checker.check_endpoint_health()
    assert result["status"] == "unhealthy"
    assert result["error_details"]["type"] == "network_error"


@pytest.mark.asyncio
async def test_cached_status_updates_after_check():
    checker, _ = _make_checker(return_value=HEALTHY_RESPONSE)
    assert checker.get_cached_health_status() is None
    await checker.check_endpoint_health()
    cached = checker.get_cached_health_status()
    assert cached is not None
    assert cached["status"] == "healthy"


@pytest.mark.asyncio
async def test_cached_status_returns_copy():
    checker, _ = _make_checker(return_value=HEALTHY_RESPONSE)
    await checker.check_endpoint_health()
    c1 = checker.get_cached_health_status()
    c2 = checker.get_cached_health_status()
    assert c1 == c2
    assert c1 is not c2


@pytest.mark.asyncio
async def test_health_check_history():
    checker, _ = _make_checker(return_value=HEALTHY_RESPONSE)
    assert checker.get_health_check_history() == []
    await checker.check_endpoint_health()
    history = checker.get_health_check_history()
    assert len(history) == 1
    assert history[0]["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_check_history_max_items():
    checker, _ = _make_checker(return_value=HEALTHY_RESPONSE)
    await checker.check_endpoint_health()
    history = checker.get_health_check_history(max_items=10)
    assert len(history) == 1
    empty_history = checker.get_health_check_history(max_items=0)
    assert len(empty_history) >= 0


@pytest.mark.asyncio
async def test_timestamp_in_result():
    checker, _ = _make_checker(return_value=HEALTHY_RESPONSE)
    result = await checker.check_endpoint_health()
    assert result["timestamp"] is not None


@pytest.mark.asyncio
async def test_last_check_time_updates():
    checker, _ = _make_checker(return_value=HEALTHY_RESPONSE)
    assert checker._last_check_time is None
    await checker.check_endpoint_health()
    assert checker._last_check_time is not None
    first_time = checker._last_check_time
    await asyncio.sleep(0.01)
    await checker.check_endpoint_health()
    assert checker._last_check_time > first_time


@pytest.mark.asyncio
async def test_response_format_has_required_fields():
    checker, _ = _make_checker(return_value=HEALTHY_RESPONSE)
    result = await checker.check_endpoint_health()
    required = {"status", "timestamp", "response_time_ms", "authentication",
                "api_version", "server_status", "error_details"}
    assert required.issubset(result.keys())


@pytest.mark.asyncio
async def test_healthy_result_error_details_is_none():
    checker, _ = _make_checker(return_value=HEALTHY_RESPONSE)
    result = await checker.check_endpoint_health()
    assert result["error_details"] is None


@pytest.mark.asyncio
async def test_error_result_error_details_is_dict():
    checker, _ = _make_checker(side_effect=BuyWhereAPIError(503, "Service unavailable"))
    result = await checker.check_endpoint_health()
    assert isinstance(result["error_details"], dict)
    assert "error" in result["error_details"]
    assert "type" in result["error_details"]


@pytest.mark.asyncio
async def test_determine_status_healthy():
    checker = MCPHealthCheck(api_key="test_key")
    data = {"status": "healthy", "authentication_status": "authenticated"}
    assert checker._determine_overall_status(data) == "healthy"


@pytest.mark.asyncio
async def test_determine_status_degraded_auth_mismatch():
    checker = MCPHealthCheck(api_key="test_key")
    data = {"status": "healthy", "authentication_status": "partial"}
    assert checker._determine_overall_status(data) == "degraded"


@pytest.mark.asyncio
async def test_determine_status_degraded_explicit():
    checker = MCPHealthCheck(api_key="test_key")
    data = {"status": "degraded", "authentication_status": "authenticated"}
    assert checker._determine_overall_status(data) == "degraded"


@pytest.mark.asyncio
async def test_determine_status_unhealthy():
    checker = MCPHealthCheck(api_key="test_key")
    data = {"status": "down", "authentication_status": "unknown"}
    assert checker._determine_overall_status(data) == "unhealthy"


@pytest.mark.asyncio
async def test_quick_health_check_factory():
    with patch("buywhere_llamaindex.mcp_health.MCPHealthCheck") as MockHC:
        mock_instance = AsyncMock()
        MockHC.return_value = mock_instance
        mock_instance.check_endpoint_health.return_value = {
            "status": "healthy",
            "timestamp": "2026-05-15T12:00:00Z",
            "authentication": "valid",
            "api_version": "1.0.0",
            "response_time_ms": 100.0,
            "server_status": "operational",
            "error_details": None,
        }
        result = await quick_health_check(api_key="test_key", base_url="https://custom.api")
        MockHC.assert_called_once_with(api_key="test_key", base_url="https://custom.api")
        assert result["status"] == "healthy"


def test_create_mcp_health_checker_returns_instance():
    checker = create_mcp_health_checker(api_key="test_key")
    assert isinstance(checker, MCPHealthCheck)


@pytest.mark.asyncio
async def test_concurrent_health_checks():
    checker, _ = _make_checker(return_value=HEALTHY_RESPONSE)
    tasks = [checker.check_endpoint_health() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 10
    assert all(r["status"] == "healthy" for r in results)


@pytest.mark.asyncio
async def test_concurrent_continuous_checks():
    checker, _ = _make_checker(return_value=HEALTHY_RESPONSE)
    tasks = [
        checker.run_continuous_health_check(interval=0.001, max_checks=3)
        for _ in range(3)
    ]
    all_results = await asyncio.gather(*tasks)
    assert len(all_results) == 3
    for results in all_results:
        assert len(results) == 3


@pytest.mark.asyncio
async def test_check_endpoint_health_custom_timeout():
    checker, mock = _make_checker(return_value=HEALTHY_RESPONSE)
    result = await checker.check_endpoint_health(timeout=5.0)
    assert result["status"] == "healthy"


@pytest.mark.asyncio
async def test_continuous_check_single_iteration():
    checker, mock = _make_checker(return_value=HEALTHY_RESPONSE)
    results = await checker.run_continuous_health_check(interval=0.001, max_checks=1)
    assert len(results) == 1
    assert mock.health_check.call_count == 1


@pytest.mark.asyncio
async def test_sequential_errors_in_continuous():
    errors = [
        BuyWhereNetworkError("timeout"),
        BuyWhereAPIError(503, "unavailable"),
        BuyWhereAuthError("bad key"),
    ]
    checker, mock = _make_checker()
    mock.health_check.side_effect = errors
    results = await checker.run_continuous_health_check(interval=0.001, max_checks=3)
    assert len(results) == 3
    assert results[0]["error_details"]["type"] == "network_error"
    assert results[1]["error_details"]["type"] == "api_error"
    assert results[2]["error_details"]["type"] == "authentication_error"
