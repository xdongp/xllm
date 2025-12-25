# Contributing to xLLM

Thank you for your interest in contributing to xLLM! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find that the problem has already been reported. When creating a bug report, please include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Any relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- A clear and descriptive title
- A detailed description of the proposed enhancement
- Explain why this enhancement would be useful
- Provide examples of how the enhancement would be used

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Update documentation as needed
6. Ensure all tests pass
7. Submit a pull request with a clear description

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/xllm.git
cd xllm
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

4. Run tests:
```bash
python -m pytest tests/
```

## Coding Standards

- Follow PEP 8 style guidelines
- Write clear, descriptive variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Add type hints where appropriate

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting a PR
- Test on multiple Python versions when possible
- Include integration tests for complex features

## Documentation

- Update README.md for user-facing changes
- Add inline comments for complex logic
- Update API documentation for API changes
- Add examples for new features

## Commit Messages

Follow the conventional commits format:

```
type(scope): subject

body

footer
```

Types:
- feat: A new feature
- fix: A bug fix
- docs: Documentation changes
- style: Code style changes (formatting, etc.)
- refactor: Code refactoring
- test: Adding or updating tests
- chore: Maintenance tasks

Example:
```
feat(scheduler): add priority queue for request scheduling

Implement a priority queue to better manage request scheduling
based on request priority and estimated completion time.

Closes #123
```

## Questions?

Feel free to open an issue or discussion if you have any questions about contributing to xLLM!
