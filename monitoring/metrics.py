"""
Simple Prometheus Metrics for Job Hunter AI
Tracks: queries, latency, tool usage, errors
"""

from prometheus_client import Counter, Histogram, start_http_server, REGISTRY, CollectorRegistry
import time
from functools import wraps

# Use a custom registry to avoid conflicts with Streamlit reruns
try:
    # Try to get existing metrics from REGISTRY
    agent_requests_total = REGISTRY._names_to_collectors.get('agent_requests_total')
    agent_latency_seconds = REGISTRY._names_to_collectors.get('agent_latency_seconds')
    agent_tool_calls_total = REGISTRY._names_to_collectors.get('agent_tool_calls_total')
    agent_errors_total = REGISTRY._names_to_collectors.get('agent_errors_total')
    agent_tokens_used = REGISTRY._names_to_collectors.get('agent_tokens_used')
except:
    pass

# Create metrics only if they don't exist
if agent_requests_total is None:
    agent_requests_total = Counter(
        'agent_requests_total',
        'Total number of agent queries',
        ['status']  # success or error
    )

if agent_latency_seconds is None:
    agent_latency_seconds = Histogram(
        'agent_latency_seconds',
        'Agent response latency in seconds',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
    )

if agent_tool_calls_total is None:
    agent_tool_calls_total = Counter(
        'agent_tool_calls_total',
        'Total number of tool calls',
        ['tool_name']
    )

if agent_errors_total is None:
    agent_errors_total = Counter(
        'agent_errors_total',
        'Total number of errors',
        ['error_type']
    )

if agent_tokens_used is None:
    agent_tokens_used = Counter(
        'agent_tokens_used',
        'Total tokens used',
        ['model_name']
    )


# Decorator for tracking latency
def track_latency(func):
    """Decorator to track function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            agent_requests_total.labels(status='success').inc()
            return result
        except Exception as e:
            agent_requests_total.labels(status='error').inc()
            agent_errors_total.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            duration = time.time() - start
            agent_latency_seconds.observe(duration)
    return wrapper


class MetricsServer:
    """Simple metrics server running on port 8001"""
    
    def __init__(self, port=8001):
        self.port = port
        self.server = None
    
    def start(self):
        """Start the metrics server"""
        try:
            start_http_server(self.port)
            print(f"✅ Prometheus metrics server running on http://localhost:{self.port}/metrics")
            return True
        except OSError as e:
            if "address already in use" in str(e).lower():
                print(f"⚠️  Port {self.port} already in use - metrics server not started")
            else:
                print(f"❌ Failed to start metrics server: {e}")
            return False


# Singleton instance
metrics_server = MetricsServer()


# Helper functions for easy metric recording
def record_query(success=True):
    """Record a query (success or failure)"""
    agent_requests_total.labels(status='success' if success else 'error').inc()


def record_tool_call(tool_name: str):
    """Record a tool invocation"""
    agent_tool_calls_total.labels(tool_name=tool_name).inc()


def record_error(error_type: str):
    """Record an error"""
    agent_errors_total.labels(error_type=error_type).inc()


def record_tokens(model_name: str, tokens: int):
    """Record token usage"""
    agent_tokens_used.labels(model_name=model_name).inc(tokens)
