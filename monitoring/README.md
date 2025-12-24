# Monitoring Setup for Job Hunter AI

## Overview

Simple monitoring setup using Prometheus + Grafana to track your agent's performance in real-time.

## Architecture

- **Prometheus**: Collects metrics from your Streamlit app
- **Grafana**: Visualizes the metrics in dashboards
- **Prometheus Client**: Python library that exposes metrics

## Quick Start

### 1. Start Monitoring Stack

```bash
cd monitoring
docker-compose up -d
```

### 2. Start Streamlit App (with metrics)

```bash
# From project root
streamlit run app_streamlit.py
```

The metrics server will automatically start on port 8001.

### 3. Access Dashboards

- **Streamlit App**: http://localhost:8501
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Metrics Endpoint**: http://localhost:8001/metrics

## Metrics Tracked

### 1. Query Volume
- `agent_requests_total` - Total number of queries
- Breakdown by success/failure status

### 2. Latency
- `agent_latency_seconds` - Response time histogram
- Percentiles: p50, p95, p99

### 3. Tool Usage
- `agent_tool_calls_total` - Which tools are used
- Breakdown by tool name (job_fetcher, resume_generator, etc.)

### 4. Token Usage
- `agent_tokens_used` - Total tokens consumed
- Per-model tracking

### 5. Errors
- `agent_errors_total` - Error count
- Breakdown by error type

## Grafana Setup

1. **Login**: http://localhost:3000 (admin/admin)

2. **Add Prometheus Data Source**:
   - Go to: Connections → Data Sources → Add data source
   - Select: Prometheus
   - URL: `http://prometheus:9090`
   - Click: Save & Test

3. **Import Dashboard**:
   - Go to: Dashboards → New → Import
   - Upload: `grafana_dashboard.json` (from this folder)

### Dashboard Panels:

1. **Total Queries** - Cumulative query count
2. **Average Latency** - Mean response time
3. **Total Tool Calls** - Tool invocation count
4. **Total Errors** - Error count
5. **Queries Per Minute** - Request rate over time
6. **Latency Percentiles** - p50, p95, p99 response times
7. **Tool Usage Distribution** - Pie chart of tool usage
8. **Success vs Error Rate** - Success/error trends

## Sample Queries

**Total Queries**:
```promql
sum(agent_requests_total)
```

**Queries per Minute**:
```promql
rate(agent_requests_total[1m])
```

**Average Latency**:
```promql
rate(agent_latency_seconds_sum[5m]) / rate(agent_latency_seconds_count[5m])
```

**P95 Latency**:
```promql
histogram_quantile(0.95, rate(agent_latency_seconds_bucket[5m]))
```

**Tool Usage Distribution**:
```promql
sum by (tool_name) (agent_tool_calls_total)
```

**Error Rate**:
```promql
rate(agent_errors_total[5m])
```

## Troubleshooting

### Metrics not showing up?
- Check if metrics server is running: http://localhost:8001/metrics
- Verify Prometheus targets: http://localhost:9090/targets
- Ensure Docker containers are running: `docker-compose ps`

### Grafana can't connect to Prometheus?
- Use `http://prometheus:9090` (not localhost)
- Docker containers must be in same network

### Streamlit app won't start?
- Port 8001 might be in use - metrics will show warning but app will still work
- Check if `prometheus-client` is installed: `pip install prometheus-client`

## Stopping Monitoring

```bash
cd monitoring
docker-compose down
```

## Viewing Logs

```bash
docker-compose logs -f prometheus
docker-compose logs -f grafana
```

---

**Simple. Works. Done.** ✅

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit App  │────▶│  Prometheus  │────▶│   Grafana    │
│  (Port 8501)    │     │  (Port 9090) │     │  (Port 3000) │
└─────────────────┘     └──────────────┘     └──────────────┘
        │                       │                     │
        │                       ▼                     │
        │               ┌──────────────┐              │
        └──────────────▶│ AlertManager │◀─────────────┘
         Metrics:8000   │  (Port 9093) │
                        └──────────────┘
```

## 📈 Metrics Tracked

### 1. **Tool Usage Correctness**
- `jobhunter_tool_invocations_total{tool_name, status}` - Counter
- Tracks success vs error rate for each tool
- Labels: `tool_name` (job_fetcher, resume_parser, etc.), `status` (success/error)

### 2. **Operation Latency**
- `jobhunter_operation_latency_seconds` - Histogram
- Tracks P50, P95, P99 latency for all operations
- Buckets: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]s

### 3. **LLM Reasoning Length (Token Usage)**
- `jobhunter_llm_tokens_used{model_name, token_type}` - Summary
- Tracks prompt and completion tokens
- Calculates average reasoning length per call

### 4. **Cost Efficiency**
- `jobhunter_operation_cost_dollars{operation_type}` - Summary
- Tracks USD cost per operation
- Pricing: GPT-4o-mini ($0.150/$0.600 per 1M tokens)

### 5. **Task Success Rate**
- `jobhunter_task_completion_total{task_type, status}` - Counter
- Tracks success vs failure for each task type
- Calculate: `success_rate = success / (success + failure)`

### 6. **Additional Metrics**
- `jobhunter_active_sessions` - Gauge (current active users)
- `jobhunter_jobs_in_cache` - Gauge (cached job results)
- `jobhunter_average_match_score` - Gauge (resume-job matching)
- `jobhunter_errors_total{error_type}` - Counter (error tracking)
- `jobhunter_api_calls_total{api_name, status}` - Counter (API monitoring)
- `jobhunter_documents_generated_total{document_type}` - Counter (DOCX/PDF)

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose installed
- API keys configured (see below)

### Step 1: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Required API Keys
OPENAI_API_KEY=sk-your-openai-key
RAPIDAPI_KEY=your-rapidapi-key

# Optional: Custom ports (defaults shown)
STREAMLIT_PORT=8501
METRICS_PORT=8000
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
ALERTMANAGER_PORT=9093

# Grafana credentials
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin123
```

### Step 2: Start Monitoring Stack

```bash
# Build and start all services
docker-compose -f docker-compose.monitoring.yml up -d

# Check service health
docker-compose -f docker-compose.monitoring.yml ps

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f
```

### Step 3: Access Dashboards

- **Streamlit App**: http://localhost:8501
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)
- **AlertManager**: http://localhost:9093
- **Metrics Endpoint**: http://localhost:8000/metrics

## 📊 Grafana Dashboard

The comprehensive dashboard includes **21 panels**:

### **Overview Panels** (Row 1)
1. Active Sessions - Current users
2. Total Tasks (24h) - Task volume
3. Task Success Rate (1h) - Overall success %
4. LLM Cost (24h) - Total spend

### **Tool Correctness** (Row 2)
5. Tool Usage Correctness - Success vs Error rate per tool
6. Task Success Rate by Type - Success % per task type

### **Latency Metrics** (Row 3-4)
7. Operation Latency (P50/P95/P99) - Multi-percentile view
8. LLM Call Latency by Model - LLM-specific latency
14. API Call Latencies - Job search & resume parsing

### **Reasoning & Cost** (Row 5)
9. LLM Token Usage - Prompt vs completion tokens (reasoning length)
10. Cost Efficiency - USD per operation + hourly rate

### **Usage Distribution** (Row 6)
11. Tool Invocations Distribution - Pie chart of tool usage
12. User Intent Distribution - Donut chart of user intents
13. Average Match Score - Resume-job matching quality

### **Errors & Documents** (Row 7-8)
15. Error Rate by Type - Error tracking
16. Documents Generated - Resume/cover letter generation

### **Performance Summary** (Row 9)
17. Tool Performance Summary - Table view of all tools

### **Operational Metrics** (Row 10)
18. Jobs in Cache - Cached search results
19. Current Workflow Stage - Real-time workflow tracking
20. Total API Calls (24h) - External API usage

### **Heatmap** (Row 11)
21. Latency Heatmap - Visual latency distribution

## 🔔 Alerting

### Alert Rules Configured

1. **HighErrorRate** (Critical)
   - Trigger: Error rate > 0.1/s for 2 minutes
   - Action: Immediate notification

2. **HighLatency** (Warning)
   - Trigger: P95 latency > 30s for 5 minutes
   - Action: Warning notification

3. **SlowLLMCalls** (Warning)
   - Trigger: P95 LLM latency > 15s for 3 minutes
   - Action: Warning notification

4. **HighLLMCost** (Warning)
   - Trigger: Cost > $1.0/hour for 5 minutes
   - Action: Cost alert

5. **LowTaskSuccessRate** (Warning)
   - Trigger: Success rate < 80% for 5 minutes
   - Action: Reliability alert

6. **ApplicationDown** (Critical)
   - Trigger: App unreachable for 1 minute
   - Action: Critical alert

### Alert Notification Channels

Configured in `monitoring/alertmanager.yml`:
- Critical alerts → Immediate notification
- Warning alerts → Grouped notifications
- Info alerts → Daily digest
- Cost alerts → Finance team (configurable)

To add Slack/Email notifications:
```yaml
# Edit monitoring/alertmanager.yml
receivers:
  - name: 'critical-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts-critical'
    email_configs:
      - to: 'oncall@example.com'
```

## 🛠️ Usage in Code

### Instrument Your Functions

```python
from monitoring.metrics import metrics_collector, track_latency, track_llm_call

# Track function latency automatically
@track_latency('job_search')
def search_jobs(query: str):
    # Your code here
    pass

# Track LLM calls with cost calculation
@track_llm_call('gpt-4o-mini')
def generate_resume(job_data, resume_data):
    # Your LLM call here
    pass

# Manual metric recording
metrics_collector.record_tool_usage('job_fetcher', True)
metrics_collector.record_task_completion('resume_generation', True)
metrics_collector.record_match_score(0.85)
metrics_collector.update_active_sessions(5)
```

### Example Integration

```python
from monitoring.metrics import metrics_collector, track_latency
import time

class JobFetcher:
    @track_latency('job_search')
    def fetch_jobs(self, query: str):
        try:
            # Fetch jobs from API
            results = self._api_call(query)
            
            # Record success
            metrics_collector.record_tool_usage('job_fetcher', True)
            metrics_collector.record_api_call('jsearch', True)
            
            return results
        except Exception as e:
            # Record failure
            metrics_collector.record_tool_usage('job_fetcher', False)
            metrics_collector.record_api_call('jsearch', False)
            metrics_collector.record_error('api_error')
            raise
```

## 📝 Monitoring Best Practices

### 1. **Metric Naming**
- Use consistent prefixes: `jobhunter_*`
- Include units in names: `_seconds`, `_total`, `_dollars`
- Use labels for dimensions, not metric names

### 2. **Cardinality Management**
- Limit label values (avoid user IDs, timestamps)
- Use aggregation for high-cardinality data
- Current labels: tool_name, status, model_name, etc.

### 3. **Alert Fatigue**
- Set appropriate thresholds
- Use inhibition rules (in alertmanager.yml)
- Group related alerts

### 4. **Dashboard Design**
- Most important metrics at top
- Use appropriate visualization types
- Set reasonable refresh intervals (30s default)

## 🔍 Troubleshooting

### Metrics Not Appearing

```bash
# Check if metrics server is running
curl http://localhost:8000/metrics

# Check Prometheus targets
# Visit http://localhost:9090/targets
# jobhunter-app should be "UP"

# Check container logs
docker-compose -f docker-compose.monitoring.yml logs jobhunter-app
docker-compose -f docker-compose.monitoring.yml logs prometheus
```

### Grafana Dashboard Not Loading

```bash
# Verify Prometheus datasource
# Grafana UI → Configuration → Data Sources
# Should show "Prometheus" as default

# Check dashboard provisioning
docker exec -it jobhunter-grafana ls /etc/grafana/provisioning/dashboards/

# Restart Grafana
docker-compose -f docker-compose.monitoring.yml restart grafana
```

### High Memory Usage

```bash
# Prometheus data retention (default: 30 days)
# Edit docker-compose.monitoring.yml:
# --storage.tsdb.retention.time=15d

# Clear old data
docker-compose -f docker-compose.monitoring.yml down -v
docker-compose -f docker-compose.monitoring.yml up -d
```

## 📊 Query Examples

### Prometheus Queries

```promql
# Average success rate per tool (last 5 minutes)
sum by (tool_name) (rate(jobhunter_tool_invocations_total{status="success"}[5m])) / 
sum by (tool_name) (rate(jobhunter_tool_invocations_total[5m]))

# P95 latency for job search
histogram_quantile(0.95, rate(jobhunter_job_search_latency_seconds_bucket[5m]))

# Total cost in last hour
sum(increase(jobhunter_operation_cost_dollars_sum[1h]))

# Average tokens per LLM call
rate(jobhunter_llm_tokens_used_sum[5m]) / rate(jobhunter_llm_tokens_used_count[5m])

# Error rate by type
sum by (error_type) (rate(jobhunter_errors_total[5m]))
```

## 🔄 Maintenance

### Backup Grafana Dashboards

```bash
# Export dashboard JSON
curl -u admin:admin123 http://localhost:3000/api/dashboards/uid/jobhunter-main > backup.json

# Import dashboard
curl -X POST -u admin:admin123 \
  -H "Content-Type: application/json" \
  -d @backup.json \
  http://localhost:3000/api/dashboards/db
```

### Update Monitoring Stack

```bash
# Pull latest images
docker-compose -f docker-compose.monitoring.yml pull

# Restart with new images
docker-compose -f docker-compose.monitoring.yml up -d

# Clean up old images
docker image prune -f
```

### Scale Services

```bash
# Scale Prometheus for HA
docker-compose -f docker-compose.monitoring.yml up -d --scale prometheus=2

# Add more AlertManager instances
docker-compose -f docker-compose.monitoring.yml up -d --scale alertmanager=3
```

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [prometheus_client Python Library](https://github.com/prometheus/client_python)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)

## 🎯 Evaluation Metrics Summary

| Metric | Type | Purpose | Target |
|--------|------|---------|--------|
| Tool Correctness | Counter | Track success vs failures | >95% success |
| Latency | Histogram | P95/P99 response times | <10s P95 |
| Reasoning Length | Summary | Token usage per call | Minimize cost |
| Cost Efficiency | Summary | USD per operation | <$0.01/operation |
| Task Success Rate | Counter | Overall reliability | >90% success |

## 📞 Support

For issues or questions:
1. Check container logs: `docker-compose logs -f`
2. Verify Prometheus targets: http://localhost:9090/targets
3. Check Grafana datasources: http://localhost:3000/datasources
4. Review alert rules: http://localhost:9090/alerts

---

**Built with ❤️ for comprehensive AI agent evaluation**
