# IDS Research Lab

## Quick Start

### Set up the environment

1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:

```bash
make setup
make activate
```

### Install new packages

To install new PyPI packages, run:

```bash
poetry add <package-name>
```

### Run Python scripts

To run the Python scripts to process data, train model, and run a notebook, type the following:

```bash
make centralized_pipeline
make federated_pipeline
make split_pipeline
```

### View all flow runs

A [flow](https://docs.prefect.io/concepts/flows/) is the basis of all Prefect workflows.

To view your flow runs from a UI, sign in to your Prefect Cloud account or spin up a Prefect Orion server on your local machine:

```bash
prefect orion start
```

Open the URL http://127.0.0.1:4200/

### Auto-generate API documentation

To auto-generate API document for your project, run:

```bash
make docs_save
```

### Run tests when creating a PR

When creating a PR, the tests in your `tests` folder will automatically run.
