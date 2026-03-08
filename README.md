# PROMICE AWS Data Ingestion Pipeline

This project is a modular MLOps pipeline designed to automatically fetch, process, and store Automated Weather Station (AWS) data from the Geological Survey of Denmark and Greenland (GEUS). 

Currently, the pipeline downloads hourly Level 3 meteorological data from the GEUS THREDDS server, resamples it to daily averages, and stores the processed datasets locally for downstream machine learning tasks.

## Prerequisites

* **Python:** `3.12`
* **Package Manager:** `uv` (An extremely fast Python package installer and resolver)

## Project Workflows

When adding new features or stages to this MLOps pipeline, follow this standard 9-step development workflow:

1. Update `config.yaml`
2. Update `schema.yaml`
3. Update `params.yaml`
4. Update the entity (dataclass in `src/mlProject/entity`)
5. Update the configuration manager (`src/mlProject/config/configuration.py`)
6. Update the components (`src/mlProject/components`)
7. Update the pipeline (`src/mlProject/pipeline`)
8. Update `main.py`
9. Update `app.py`

## Setup and Installation

**1. Install `uv`**
If you don't have `uv` installed yet, you can install it globally via curl (macOS/Linux) or PowerShell (Windows):
```bash
# macOS / Linux
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Windows
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"

# Or via pip
pip install uv