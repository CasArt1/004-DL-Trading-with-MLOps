# Contributing to DL Trading with MLOps

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/004-DL-Trading-with-MLOps.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Commit and push: `git push origin feature/your-feature-name`
7. Open a Pull Request

## Development Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify installation:
   ```bash
   python scripts/verify_installation.py
   ```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

Example:
```python
def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        data: DataFrame with price data
        period: RSI period (default: 14)
    
    Returns:
        Series with RSI values
    """
    # Implementation here
```

## Testing

Before submitting a PR:

1. Run syntax check:
   ```bash
   python -m py_compile src/**/*.py scripts/*.py
   ```

2. Test your changes manually
3. Ensure all scripts run without errors

## Pull Request Process

1. Update README.md if needed
2. Update documentation for any API changes
3. Describe your changes clearly in the PR description
4. Link any related issues

## Adding New Features

When adding new features:

1. **Feature Engineering**: Add to `src/feature_engineering/`
2. **Model Architectures**: Add to `src/models/`
3. **API Endpoints**: Add to `src/api/app.py`
4. **Monitoring**: Add to `src/monitoring/`
5. **Backtesting**: Add to `src/backtesting/`

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces

## Feature Requests

We welcome feature requests! Please:

- Check if the feature already exists
- Describe the use case
- Explain why it would be useful
- Provide examples if possible

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Questions?

Open an issue with the "question" label or reach out to the maintainers.

Thank you for contributing! 🚀
