@echo off
REM Development setup script for DecayShape (Windows)

echo 🚀 Setting up DecayShape development environment...

REM Check if we're in a virtual environment
if "%VIRTUAL_ENV%"=="" (
    echo ⚠️  Warning: You're not in a virtual environment.
    echo    Consider creating one with: python -m venv venv ^&^& venv\Scripts\activate
    echo.
)

REM Install the package in development mode
echo 📦 Installing DecayShape in development mode...
pip install -e .

REM Install development dependencies
echo 🔧 Installing development dependencies...
pip install pytest pytest-cov black isort flake8 flake8-docstrings flake8-bugbear mypy bandit pydocstyle pyupgrade autoflake pre-commit

REM Install optional dependencies
echo 📊 Installing optional dependencies...
pip install matplotlib pandas 2>nul || echo ⚠️  Could not install matplotlib/pandas (optional)
pip install jax jaxlib 2>nul || echo ⚠️  Could not install JAX (optional)

REM Install pre-commit hooks
echo 🪝 Installing pre-commit hooks...
python -m pre_commit install

REM Run pre-commit on all files to check setup
echo ✅ Running pre-commit on all files to verify setup...
python -m pre_commit run --all-files 2>nul || echo ⚠️  Some pre-commit checks failed - this is normal for first setup

echo.
echo ✨ Development environment setup complete!
echo.
echo Next steps:
echo   1. Run tests: pytest
echo   2. Check code style: pre-commit run --all-files
echo   3. Start developing! Pre-commit hooks will run automatically.
echo.
echo Useful commands:
echo   pytest                    # Run all tests
echo   pytest --cov=decayshape   # Run tests with coverage
echo   black decayshape/         # Format code
echo   isort decayshape/         # Sort imports
echo   flake8 decayshape/        # Check linting
echo   mypy decayshape/          # Type checking
echo   pre-commit run --all-files # Run all pre-commit hooks

pause
