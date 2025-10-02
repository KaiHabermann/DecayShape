#!/bin/bash
# Development setup script for DecayShape

set -e

echo "🚀 Setting up DecayShape development environment..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: You're not in a virtual environment."
    echo "   Consider creating one with: python -m venv venv && source venv/bin/activate"
    echo ""
fi

# Install the package in development mode
echo "📦 Installing DecayShape in development mode..."
pip install -e .

# Install development dependencies
echo "🔧 Installing development dependencies..."
pip install \
    pytest \
    pytest-cov \
    black \
    isort \
    flake8 \
    flake8-docstrings \
    flake8-bugbear \
    mypy \
    bandit \
    pydocstyle \
    pyupgrade \
    autoflake \
    pre-commit

# Install optional dependencies
echo "📊 Installing optional dependencies..."
pip install matplotlib pandas || echo "⚠️  Could not install matplotlib/pandas (optional)"
pip install jax jaxlib || echo "⚠️  Could not install JAX (optional)"

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Run pre-commit on all files to check setup
echo "✅ Running pre-commit on all files to verify setup..."
pre-commit run --all-files || echo "⚠️  Some pre-commit checks failed - this is normal for first setup"

echo ""
echo "✨ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run tests: pytest"
echo "  2. Check code style: pre-commit run --all-files"
echo "  3. Start developing! Pre-commit hooks will run automatically."
echo ""
echo "Useful commands:"
echo "  pytest                    # Run all tests"
echo "  pytest --cov=decayshape   # Run tests with coverage"
echo "  black decayshape/         # Format code"
echo "  isort decayshape/         # Sort imports"
echo "  flake8 decayshape/        # Check linting"
echo "  pre-commit run --all-files # Run all pre-commit hooks"
echo ""
echo "Optional advanced checks:"
echo "  mypy decayshape/                    # Type checking"
echo "  bandit -r decayshape/               # Security checks"
echo "  pydocstyle decayshape/              # Docstring style"
echo "  pre-commit run --hook-stage manual  # Run optional hooks"
