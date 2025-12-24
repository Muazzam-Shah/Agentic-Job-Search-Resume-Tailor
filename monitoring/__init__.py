"""
Simple monitoring module for Job Hunter AI
"""

from .metrics import (
    metrics_server,
    track_latency,
    record_query,
    record_tool_call,
    record_error,
    record_tokens,
    agent_requests_total,
    agent_latency_seconds,
    agent_tool_calls_total,
    agent_errors_total,
    agent_tokens_used
)

__all__ = [
    'metrics_server',
    'track_latency',
    'record_query',
    'record_tool_call',
    'record_error',
    'record_tokens',
    'agent_requests_total',
    'agent_latency_seconds',
    'agent_tool_calls_total',
    'agent_errors_total',
    'agent_tokens_used'
]
