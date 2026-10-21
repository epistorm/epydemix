# Contributing to Epydemix

First off, thank you for considering contributing to Epydemix!

## Getting Started

1.  Fork the repository on GitHub.
2.  Clone your forked repository locally.
3.  Install the dependencies (including development dependencies):

    ```bash
    pip install -r requirements.txt
    pip install -r dev-requirements.txt
    ```

4.  Install the pre-commit hook:

    ```bash
    pre-commit install
    ```

## Development Workflow

1.  Create a new branch for your feature or bug fix.
2.  Make your changes.
3.  Run `ruff` to format and lint your code:

    ```bash
    ruff format .
    ruff check .
    ```

4.  Commit your changes. The pre-commit hook will automatically run `ruff` and fix any issues it can.
5.  Push your changes to your forked repository.
6.  Open a pull request on the main Epydemix repository.

## Code Style

We use `ruff` to format and lint our code.

Please make sure your code conforms to the `ruff` configuration in `pyproject.toml`.

## Testing

We use `pytest` for testing. Please make sure all tests pass before submitting a pull request.

To run the tests, use the following command:

```bash
pytest
```

## Documentation

The documentation is in `docs/source/` (Markdown, built with Sphinx and the Furo theme) and is published on [Read the Docs](https://epydemix.readthedocs.io/). The tutorial notebooks in `tutorials/` are included in the site with their saved outputs; they are not re-run during the build, so re-run and save a notebook when its outputs should change.

To build the documentation locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html -W --keep-going docs/source docs/build/html
```

Then open `docs/build/html/index.html`. The `-W` flag turns warnings into errors, as on Read the Docs and in CI.
