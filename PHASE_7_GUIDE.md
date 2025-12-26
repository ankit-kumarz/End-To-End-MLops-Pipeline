# 🚀 PHASE 7: Model Serving Architecture - Complete Guide

## 📖 Table of Contents
1. [What is Model Serving?](#what-is-model-serving)
2. [Why FastAPI for ML?](#why-fastapi-for-ml)
3. [Architecture Overview](#architecture-overview)
4. [Key Components](#key-components)
5. [API Endpoints](#api-endpoints)
6. [Testing the API](#testing-the-api)
7. [Interview Focus Points](#interview-focus-points)

---

## What is Model Serving?

**Model Serving** = Making your trained ML model accessible to other systems via API

### The Production Journey

```
Training (Data Scientist)          Serving (Software Engineer)
┌─────────────────┐               ┌──────────────────────┐
│  Jupyter        │               │   FastAPI Server     │
│  Notebook       │               │                      │
│                 │               │   POST /predict      │
│  model.fit()    │──────────────▶│   {features: [...]}  │
│  model.pkl      │               │                      │
│                 │               │   Return prediction  │
└─────────────────┘               └──────────────────────┘
      ❌ Not usable                      ✅ Production-ready
```

### Why API Serving?

1. **Language Agnostic**: Any language can call HTTP APIs
2. **Scalability**: Can deploy multiple instances behind load balancer
3. **Security**: Add authentication, rate limiting, monitoring
4. **Versioning**: Deploy multiple model versions simultaneously
5. **Maintainability**: Update models without changing client code

---

## Why FastAPI for ML?

### FastAPI vs Flask/Django

| Feature | Flask | Django | FastAPI |
|---------|-------|--------|---------|
| **Speed** | Slow | Slow | **Fast** (ASGI) |
| **Type Validation** | Manual | Manual | **Automatic** (Pydantic) |
| **Async Support** | Limited | Limited | **Native** |
| **Auto Docs** | ❌ | ❌ | **✅ (Swagger/ReDoc)** |
| **Modern Python** | Old syntax | Old syntax | **Python 3.7+** |

### Why FastAPI Wins for ML Serving

1. **Automatic Input Validation**: Pydantic catches bad inputs before prediction
2. **Built-in API Documentation**: Auto-generated Swagger UI at `/docs`
3. **High Performance**: Comparable to Node.js/Go (ASGI async)
4. **Type Safety**: IDE autocomplete, fewer bugs
5. **Production Ready**: Used by Netflix, Microsoft, Uber

**Interview Tip**: "I chose FastAPI because it provides automatic input validation with Pydantic, which is crucial for ML APIs where bad inputs can crash models. Plus, the auto-generated documentation makes integration easy for frontend teams."

---

## Architecture Overview

### Request Flow

```
┌──────────────────────────────────────────────────────────┐
│                    FASTAPI SERVER                         │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. CLIENT REQUEST                                        │
│     POST /predict                                         │
│     {"feature_1": 0.5, "feature_2": 1.2, ...}           │
│                                                           │
│  2. PYDANTIC VALIDATION ✓                                │
│     - Check all required fields present                  │
│     - Validate types (float, int)                        │
│     - Validate ranges (feature_4: 0-100)                 │
│                                                           │
│  3. FEATURE ENGINEERING                                   │
│     - Apply same transformations as training             │
│     - Create polynomial, interaction features            │
│     - Handle one-hot encoding                            │
│                                                           │
│  4. SCALING                                               │
│     - Load saved scaler.pkl                              │
│     - Transform features (StandardScaler)                │
│                                                           │
│  5. MODEL PREDICTION                                      │
│     - Load model.pkl                                     │
│     - model.predict()                                    │
│     - model.predict_proba()                              │
│                                                           │
│  6. RESPONSE                                              │
│     {                                                     │
│       "prediction": 1,                                   │
│       "probability": 0.95,                               │
│       "model_version": "random_forest",                  │
│       "timestamp": "2025-12-26T..."                      │
│     }                                                     │
│                                                           │
│  7. LOGGING                                               │
│     - Log request details                                │
│     - Log predictions (for monitoring)                   │
│     - Track errors                                       │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Critical Production Patterns

#### 1. Application Lifespan Management

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ STARTUP: Load model ONCE (not per request)
    load_model_and_artifacts()
    yield
    # ✅ SHUTDOWN: Cleanup resources
    logger.info("Shutting down...")
```

**Why?** Loading model on every request = 100x slower response times

#### 2. Pydantic Input Validation

```python
class PredictionInput(BaseModel):
    feature_4: float = Field(..., ge=0, le=100)  # Range validation
    
    @validator('feature_4')
    def validate_feature_4(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Invalid range')
        return v
```

**Why?** Bad inputs can crash models or produce nonsense predictions

#### 3. Feature Engineering Consistency

```python
def engineer_features(input_data):
    # ✅ EXACT SAME CODE as training pipeline
    df['feature_1_squared'] = df['feature_1'] ** 2
    # ...
```

**Why?** Training/serving skew = model performs poorly in production

---

## Key Components

### 1. Input/Output Schemas (Pydantic)

```python
class PredictionInput(BaseModel):
    """Define EXACTLY what API expects"""
    feature_1: float
    feature_2: float
    # ...
    
    class Config:
        json_schema_extra = {  # Example in docs
            "example": {"feature_1": 0.5, ...}
        }
```

### 2. Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_accuracy": 0.995
    }
```

**Interview Question**: "Why do you need a health endpoint?"

**Answer**: "For Kubernetes readiness/liveness probes. Load balancers use it to check if instance is ready to serve traffic. If health check fails, traffic is routed away."

### 3. Batch Prediction Support

```python
@app.post("/predict/batch")
async def predict_batch(input_data: BatchPredictionInput):
    predictions = []
    for instance in input_data.instances:
        # Process each
        predictions.append(...)
    return predictions
```

**Why?** 10x more efficient than 100 individual API calls

---

## API Endpoints

### GET /
- **Purpose**: API information and available endpoints
- **Use Case**: Documentation reference

### GET /health
- **Purpose**: Service health check
- **Response**: Status, model loaded, accuracy
- **Use Case**: Kubernetes probes, monitoring

### GET /model/info
- **Purpose**: Model metadata
- **Response**: Algorithm, hyperparameters, metrics, training info
- **Use Case**: Model version tracking, debugging

### POST /predict
- **Purpose**: Single prediction
- **Input**: Raw feature values
- **Output**: Prediction, probability, version, timestamp
- **Use Case**: Real-time predictions

### POST /predict/batch
- **Purpose**: Batch predictions
- **Input**: List of feature instances
- **Output**: List of predictions
- **Use Case**: Bulk processing

---

## Testing the API

### 1. Start the Server

```bash
# Method 1: Direct
python src/serve.py

# Method 2: Uvicorn with reload
uvicorn src.serve:app --reload --port 8000
```

### 2. Access Auto-Generated Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

**Try it in the browser!** Click "Try it out" buttons in Swagger.

### 3. Test with Python Script

```bash
# Run the test script
python test_api.py
```

### 4. Test with cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "feature_1": 0.5,
    "feature_2": 1.2,
    "feature_3": -0.3,
    "feature_4": 75.5,
    "feature_5": 5
  }'
```

### 5. Test with Postman

1. Create new POST request to `http://localhost:8000/predict`
2. Set Headers: `Content-Type: application/json`
3. Set Body (raw JSON):
```json
{
  "feature_1": 0.5,
  "feature_2": 1.2,
  "feature_3": -0.3,
  "feature_4": 75.5,
  "feature_5": 5
}
```

---

## Interview Focus Points

### Question 1: "How do you ensure training/serving consistency?"

**Answer**:
```
1. Same feature engineering code in serve.py as data_pipeline.py
2. Save and load the SAME scaler used in training
3. Validate feature order matches training
4. Log predictions for monitoring (detect drift)
```

### Question 2: "How do you handle model updates without downtime?"

**Answer**:
```
1. Blue-Green Deployment:
   - Deploy new version alongside old
   - Switch traffic gradually
   - Rollback if issues

2. Canary Deployment:
   - Route 5% traffic to new model
   - Monitor performance
   - Gradually increase if good

3. Model Registry:
   - Load model from MLflow by version
   - Change version without code deploy
   - Environment variable: MODEL_VERSION=v2
```

### Question 3: "How do you validate inputs in production?"

**Answer**:
```
Pydantic provides:
1. Type validation (int, float, string)
2. Range validation (Field(ge=0, le=100))
3. Custom validators (@validator)
4. Automatic 422 error responses

This prevents:
- Wrong data types crashing model
- Out-of-range values causing errors
- Missing fields causing NoneType errors
```

### Question 4: "How do you monitor API performance?"

**Answer**:
```
1. Request Logging:
   - Log every prediction (input, output, latency)
   - Store in database for analysis

2. Metrics:
   - Response time (p50, p95, p99)
   - Error rate (4xx, 5xx)
   - Throughput (requests/sec)
   - Model metrics (accuracy on live data)

3. Alerting:
   - High latency (> 500ms)
   - High error rate (> 1%)
   - Data drift detected
   - Model performance degradation

4. Tools:
   - Prometheus + Grafana
   - DataDog
   - New Relic
```

### Question 5: "How do you scale API for high traffic?"

**Answer**:
```
1. Horizontal Scaling:
   - Deploy multiple FastAPI instances
   - Load balancer distributes traffic
   - Auto-scaling based on CPU/memory

2. Caching:
   - Cache predictions for common inputs
   - Redis for distributed cache

3. Async Processing:
   - FastAPI native async support
   - Non-blocking I/O operations

4. Model Optimization:
   - Use ONNX Runtime (faster inference)
   - Quantization (reduce model size)
   - Batch predictions when possible

5. Infrastructure:
   - Kubernetes for orchestration
   - Horizontal Pod Autoscaler
   - Service mesh (Istio)
```

---

## Real-World Production Considerations

### 1. Authentication & Authorization

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict")
async def predict(input_data: PredictionInput, 
                  token: HTTPAuthorizationCredentials = Depends(security)):
    # Verify token
    if not verify_token(token.credentials):
        raise HTTPException(401, "Invalid token")
    # ...
```

### 2. Rate Limiting

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/minute")  # Max 100 requests per minute
async def predict(...):
    ...
```

### 3. Request ID Tracking

```python
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

### 4. Model Version in Response

```python
# Allow A/B testing multiple models
@app.post("/predict")
async def predict(model_version: str = "v1"):
    model = load_model(version=model_version)
    # ...
```

### 5. Structured Logging

```python
logger.info(
    "prediction_made",
    extra={
        "request_id": request_id,
        "model_version": "v1",
        "prediction": prediction,
        "latency_ms": latency,
        "client_ip": request.client.host
    }
)
```

---

## Next Steps

### Immediate Actions:
1. ✅ Start the API: `python src/serve.py`
2. ✅ Test endpoints: `python test_api.py`
3. ✅ Explore Swagger UI: http://localhost:8000/docs

### Phase 8 Preview: CI/CD with GitHub Actions
- Automate testing on every push
- Automate model retraining on data changes
- Automate API deployment

---

## 🎯 Key Takeaways

1. **FastAPI** = Modern, fast, auto-validated ML serving
2. **Pydantic** = Input validation prevents 90% of production bugs
3. **Health Checks** = Required for Kubernetes/cloud deployment
4. **Batch Endpoints** = 10x more efficient than individual calls
5. **Logging** = Essential for monitoring and debugging
6. **Training/Serving Consistency** = Same code, same scaler, same order

**You now have a production-grade ML serving API!** 🚀

The model that was "just a .pkl file" is now:
- ✅ Accessible via REST API
- ✅ Auto-validated inputs
- ✅ Auto-documented (Swagger)
- ✅ Production-ready
- ✅ Interview-impressive

