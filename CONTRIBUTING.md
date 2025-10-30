# Contributing to Paper2Code

Thank you for your interest in contributing to Paper2Code! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Paper2Code.git
   cd Paper2Code
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/Vikaash-dev/Paper2Code.git
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the [code style guidelines](#code-style)

3. Add tests for new functionality

4. Update documentation as needed

## Testing

Run tests before submitting your changes:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=codes --cov-report=html

# Run specific test file
pytest tests/test_utils.py
```

## Code Style

This project follows these coding standards:

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Add type hints to all functions
- Maximum line length: 100 characters

### Code Formatting

Use the following tools (pre-commit hooks will run these automatically):

```bash
# Format code with Black
black codes/

# Sort imports with isort
isort codes/

# Check code with flake8
flake8 codes/

# Type check with mypy
mypy codes/
```

### Documentation

- Add docstrings to all modules, classes, and functions
- Include type hints in function signatures
- Update README.md if adding new features
- Add examples for new functionality

### Commit Messages

Write clear, descriptive commit messages:

```
feat: Add support for additional LLM providers

- Add Anthropic Claude support
- Update configuration handling
- Add tests for new providers

Closes #123
```

Use conventional commit prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

## Submitting Changes

1. Ensure all tests pass and code follows style guidelines

2. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request on GitHub:
   - Provide a clear title and description
   - Reference any related issues
   - Include screenshots for UI changes
   - List any breaking changes

4. Wait for review and address feedback

## Areas for Contribution

We welcome contributions in these areas:

### High Priority
- Adding unit tests for existing functionality
- Improving error handling and validation
- Adding support for more LLM providers
- Performance optimizations

### Documentation
- Improving code documentation
- Adding tutorials and examples
- Creating architecture diagrams
- Translating documentation

### Features
- Progress tracking and resumption
- Caching for expensive operations
- Better cost estimation and monitoring
- Enhanced evaluation metrics

### Bug Fixes
- Check the [issue tracker](https://github.com/Vikaash-dev/Paper2Code/issues) for known bugs
- Report new bugs with detailed reproduction steps

## Questions?

If you have questions, please:
1. Check existing documentation
2. Search through [existing issues](https://github.com/Vikaash-dev/Paper2Code/issues)
3. Open a new issue with the `question` label

Thank you for contributing to Paper2Code! 🎉
