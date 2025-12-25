"""
Test script to verify metrics are being recorded correctly
"""

from monitoring.metrics import (
    record_query, 
    record_tool_call, 
    record_error, 
    record_tokens,
    agent_latency_seconds,
    metrics_server
)
import time
import requests

def test_metrics():
    """Test all metric types"""
    print("🧪 Testing metrics recording...\n")
    
    # Start metrics server
    print("1. Starting metrics server...")
    metrics_server.start()
    time.sleep(2)
    
    # Record some test metrics
    print("2. Recording test metrics...")
    
    # Record queries
    record_query(success=True)
    record_query(success=True)
    record_query(success=False)
    print("   ✅ Recorded 2 successful and 1 failed query")
    
    # Record tool calls
    record_tool_call("job_search")
    record_tool_call("resume_generation")
    record_tool_call("job_search")
    print("   ✅ Recorded 3 tool calls")
    
    # Record errors
    record_error("ValueError")
    print("   ✅ Recorded 1 error")
    
    # Record latency
    agent_latency_seconds.observe(1.5)
    agent_latency_seconds.observe(2.3)
    agent_latency_seconds.observe(0.8)
    print("   ✅ Recorded 3 latency measurements")
    
    # Record tokens
    record_tokens("gpt-4", 1500)
    print("   ✅ Recorded token usage")
    
    time.sleep(1)
    
    # Fetch and display metrics
    print("\n3. Fetching metrics from server...")
    try:
        response = requests.get("http://localhost:8001/metrics")
        if response.status_code == 200:
            print("   ✅ Metrics endpoint responding\n")
            
            # Parse and display relevant metrics
            lines = response.text.split('\n')
            print("📊 Current Metrics:\n")
            
            for line in lines:
                if line.startswith('agent_'):
                    # Skip metadata lines
                    if '_created' not in line and 'TYPE' not in line and 'HELP' not in line:
                        print(f"   {line}")
        else:
            print(f"   ❌ Failed to fetch metrics: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error fetching metrics: {e}")
    
    print("\n✅ Metrics test complete!")
    print("\n💡 Next steps:")
    print("   1. Check http://localhost:8001/metrics to see all metrics")
    print("   2. Use your app and watch metrics update in real-time")
    print("   3. View Grafana dashboard for visualizations")

if __name__ == "__main__":
    test_metrics()
