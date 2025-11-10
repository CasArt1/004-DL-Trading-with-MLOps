# Security Policy

## Reporting Security Issues

If you discover a security vulnerability, please email the maintainers directly instead of opening a public issue.

## Security Considerations

### 1. Model Loading

**Issue**: The API uses `joblib.load()` to load the scaler, which can execute arbitrary Python code from pickle files.

**Mitigation**:
- Path validation ensures only files within the `models/` directory can be loaded
- The `/load-model` endpoint rejects path traversal attempts (`..` in paths)
- Only load model artifacts from trusted sources
- In production, consider using safer serialization formats (JSON, ONNX) for the scaler

**Recommendation**: 
- Keep the `models/` directory secure and writable only by trusted processes
- Do not allow user-uploaded files in the `models/` directory
- Consider implementing additional authentication for the `/load-model` endpoint

### 2. API Authentication

**Current State**: The API does not implement authentication.

**Production Recommendations**:
- Implement API key authentication or OAuth 2.0
- Use HTTPS/TLS for all API communications
- Implement rate limiting to prevent abuse
- Add request logging for audit trails

Example with API key authentication:
```python
from fastapi.security import APIKeyHeader
from fastapi import Security, HTTPException

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/predict")
async def predict(market_data: MarketData, api_key: str = Security(verify_api_key)):
    # ... prediction logic
```

### 3. Input Validation

**Current State**: Basic validation is implemented for market data format.

**Best Practices**:
- Validate all input data types and ranges
- Sanitize user inputs
- Implement request size limits
- Use Pydantic models for automatic validation

### 4. Data Privacy

**Considerations**:
- Market data may contain sensitive trading information
- Predictions could be considered proprietary
- Log data carefully to avoid exposing sensitive information

**Recommendations**:
- Implement data encryption at rest and in transit
- Use secure storage for sensitive data
- Implement proper access controls
- Consider data retention policies

### 5. Dependency Security

**Current Practices**:
- Use specific version numbers in `requirements.txt`
- Regularly update dependencies

**Recommendations**:
```bash
# Check for vulnerabilities
pip install safety
safety check

# Update dependencies
pip list --outdated
pip install --upgrade package_name
```

### 6. Container Security

**Docker Best Practices**:
- Use official base images
- Run containers as non-root user
- Scan images for vulnerabilities
- Keep base images updated

Example for running as non-root:
```dockerfile
FROM python:3.10-slim
RUN useradd -m -u 1000 appuser
USER appuser
```

### 7. Environment Variables

**Best Practices**:
- Never commit secrets to version control
- Use environment variables for sensitive configuration
- Use secret management tools (AWS Secrets Manager, HashiCorp Vault)

Example `.env` file (not committed):
```
API_KEY=your-secret-api-key
MLFLOW_TRACKING_URI=your-mlflow-server
DATABASE_URL=your-database-connection
```

### 8. MLFlow Security

**Considerations**:
- MLFlow UI exposes experiment data
- Model registry contains proprietary models

**Recommendations**:
- Secure MLFlow tracking server with authentication
- Use separate MLFlow instances for dev/prod
- Implement access controls for model registry
- Encrypt experiment data

### 9. Network Security

**Production Recommendations**:
- Use firewalls to restrict access
- Implement VPC/network segmentation
- Use reverse proxy (nginx, Traefik) with SSL
- Implement IP whitelisting if appropriate

### 10. Code Security

**Best Practices**:
- Regular security audits
- Use static analysis tools (CodeQL, Bandit)
- Follow OWASP guidelines
- Implement proper error handling without exposing internal details

Run security checks:
```bash
# Install security tools
pip install bandit

# Run security scan
bandit -r src/

# Check for known vulnerabilities
pip install safety
safety check
```

## Security Checklist for Production Deployment

- [ ] Enable HTTPS/TLS
- [ ] Implement authentication and authorization
- [ ] Add rate limiting
- [ ] Secure model loading paths
- [ ] Use environment variables for secrets
- [ ] Implement request logging
- [ ] Set up monitoring and alerting
- [ ] Regular security audits
- [ ] Update dependencies regularly
- [ ] Use non-root container user
- [ ] Implement data encryption
- [ ] Secure MLFlow tracking server
- [ ] Set up network security (firewall, VPC)
- [ ] Implement backup and disaster recovery
- [ ] Document incident response procedures

## Responsible Disclosure

We appreciate responsible disclosure of security vulnerabilities. Please:
1. Email security concerns to the maintainers
2. Allow reasonable time for fixes before public disclosure
3. Provide detailed information to help reproduce the issue
4. Do not exploit vulnerabilities for malicious purposes

## Updates

This security policy is regularly reviewed and updated. Last update: 2024-01-10
