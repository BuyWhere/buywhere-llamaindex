"""MCP-facing health helpers for BuyWhere API checks."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from buywhere_llamaindex.client import (
    BuyWhereAPIError,
    BuyWhereAuthError,
    BuyWhereClient,
    BuyWhereNetworkError,
)


class MCPHealthCheck:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self._client = BuyWhereClient(api_key=api_key, base_url=base_url)
        self._history: List[Dict[str, Any]] = []
        self._last_check_time: Optional[datetime] = None
        self._last_health_status: Optional[Dict[str, Any]] = None

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _determine_overall_status(self, data: Dict[str, Any]) -> str:
        status = str(data.get("status", "")).lower()
        auth_status = str(
            data.get("authentication_status", data.get("authentication", "unknown"))
        ).lower()
        if status == "healthy" and auth_status in {"authenticated", "valid", "ok"}:
            return "healthy"
        if status == "degraded" or auth_status in {"partial", "degraded"}:
            return "degraded"
        return "unhealthy"

    async def check_endpoint_health(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        started = asyncio.get_running_loop().time()
        try:
            raw = await asyncio.to_thread(self._client.health_check, timeout)
            response_time_ms = raw.get("response_time_ms")
            if response_time_ms is None:
                response_time_ms = round((asyncio.get_running_loop().time() - started) * 1000, 2)

            status = self._determine_overall_status(raw)
            auth_status = str(raw.get("authentication_status", "unknown")).lower()
            result = {
                "status": status,
                "timestamp": raw.get("timestamp") or self._timestamp(),
                "response_time_ms": response_time_ms,
                "authentication": "valid" if auth_status == "authenticated" else auth_status,
                "api_version": raw.get("api_version"),
                "server_status": raw.get("server_info", {}).get("status", raw.get("status", "unknown")),
                "error_details": None,
            }
            if status != "healthy":
                result["error_details"] = {
                    "type": "degraded" if status == "degraded" else "unhealthy",
                    "error": raw.get("status", "Health check returned a non-healthy status"),
                }
        except BuyWhereAuthError as exc:
            result = {
                "status": "unhealthy",
                "timestamp": self._timestamp(),
                "response_time_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2),
                "authentication": "invalid",
                "api_version": None,
                "server_status": "unauthorized",
                "error_details": {"type": "authentication_error", "error": str(exc)},
            }
        except BuyWhereAPIError as exc:
            result = {
                "status": "unhealthy",
                "timestamp": self._timestamp(),
                "response_time_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2),
                "authentication": "unknown",
                "api_version": None,
                "server_status": "error",
                "error_details": {
                    "type": "api_error",
                    "error": str(exc),
                    "status_code": exc.status_code,
                },
            }
        except BuyWhereNetworkError as exc:
            result = {
                "status": "unhealthy",
                "timestamp": self._timestamp(),
                "response_time_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2),
                "authentication": "unknown",
                "api_version": None,
                "server_status": "unreachable",
                "error_details": {"type": "network_error", "error": str(exc)},
            }

        self._last_check_time = datetime.now(timezone.utc)
        self._last_health_status = deepcopy(result)
        self._history.append(deepcopy(result))
        return result

    async def run_continuous_health_check(
        self,
        interval: float = 5.0,
        max_checks: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        while max_checks is None or len(results) < max_checks:
            try:
                results.append(await self.check_endpoint_health())
                if max_checks is not None and len(results) >= max_checks:
                    break
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
        return results

    def get_cached_health_status(self) -> Optional[Dict[str, Any]]:
        if self._last_health_status is None:
            return None
        return deepcopy(self._last_health_status)

    def get_health_check_history(self, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        history = deepcopy(self._history)
        if max_items is None:
            return history
        if max_items <= 0:
            return history[0:0]
        return history[-max_items:]


def create_mcp_health_checker(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> MCPHealthCheck:
    return MCPHealthCheck(api_key=api_key, base_url=base_url)


async def quick_health_check(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    checker = MCPHealthCheck(api_key=api_key, base_url=base_url)
    return await checker.check_endpoint_health()
