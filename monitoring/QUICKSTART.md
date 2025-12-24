# Simple Monitoring Setup ✅

## What You Have

- ✅ Prometheus (metrics collection)
- ✅ Grafana (visualization)
- ✅ Simple metrics tracking in Streamlit app

## How to Use

### 1. Start Monitoring (One Time)

```bash
cd monitoring
docker-compose up -d
```

### 2. Start Your App

```bash
# From project root
streamlit run app_streamlit.py
```

The app will automatically start the metrics server on port 8001.

### 3. Open Dashboards

- **Streamlit App**: http://localhost:8501
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Raw Metrics**: http://localhost:8001/metrics

### 4. Setup Grafana (First Time Only)

1. Open http://localhost:3000 (login: admin/admin)
2. Click menu → **Connections** → **Data sources** → **Add data source**
3. Select **Prometheus**
4. URL: `http://prometheus:9090`
5. Click **Save & Test** ✅
6. Click menu → **Dashboards** → **New** → **Import**
7. Upload `grafana_dashboard.json` from the monitoring folder
8. Done! Your dashboard is live 📊

### 5. Use the App & Watch Metrics

- Ask questions in Streamlit
- Search for jobs
- Generate resumes
- Watch Grafana update in real-time!

## What's Tracked

- **Total Queries** - How many requests
- **Latency** - Response times (P50, P95, P99)
- **Tool Usage** - Which tools are used (job_fetcher, resume_generator, etc.)
- **Errors** - Error count
- **Success Rate** - Success vs failures

## Stop Everything

```bash
# Stop monitoring
cd monitoring
docker-compose down

# Stop Streamlit (Ctrl+C in terminal)
```

## Troubleshooting

**Metrics not showing?**
- Make sure Streamlit is running
- Visit http://localhost:8001/metrics to check
- Ask at least one query in Streamlit first

**Grafana shows "No data"?**
- Check Prometheus targets: http://localhost:9090/targets
- Should show "jobhunter-agent" as UP
- Make sure data source URL is `http://prometheus:9090` (not localhost)

**Port 8001 already in use?**
- Metrics server will show a warning but app will still work
- Or change port in `monitoring/metrics.py`

---

**That's it! Simple, clean, works.** ✨
