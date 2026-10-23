# Contributing to Air Prediction

Thank you for contributing to the Air Prediction project! We welcome code contributions, issue reports, and documentation improvements.

---

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pratham-malhotr/air_prediction.git
   cd air_prediction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies in editable mode with development tools:**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

---

## 🌿 Branching & Git Workflow

1. Create a feature or bugfix branch off `main`:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. Keep commits atomic and use descriptive commit messages (following conventional commits, e.g., `feat:`, `fix:`, `docs:`, `test:`, `refactor:`).

3. Always run the test suite locally before pushing:
   ```bash
   pytest
   ```

4. Push your branch to GitHub and open a Pull Request against `main`.

---

## 🧪 Testing Guidelines

- New modules or features must be accompanied by corresponding unit or integration tests under `tests/`.
- Ensure tests run fast and do not depend on external network requests (mock downloads if needed).
- Confirm all 25+ existing test cases continue to pass without regression.

---

## 📝 Code Style & Conventions

- Follow **PEP 8** style guidelines.
- Use explicit type annotations and dataclasses where appropriate.
- Avoid introducing circular imports within `air_prediction`.
- Avoid committing large dataset files (`data/raw/` is ignored by default; keep only `.gitkeep`).
