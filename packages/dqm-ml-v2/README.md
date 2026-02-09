# DQM-ML V2 CLI Wrapper

This package provides the main entry point and CLI for the DQM-ML V2 framework. It consolidates the modular packages (`core`, `pipeline`, `images`, `pytorch`) into a single command-line interface.

## Installation

```bash
# Basic installation
pip install dqm-ml-v2

# Installation with optional components
pip install "dqm-ml-v2[all]" # Everything
pip install "dqm-ml-v2[pipeline]" # Just the pipeline and core
pip install "dqm-ml-v2[notebooks]" # Jupyter support
```

## Quick Start

Process a dataset using a configuration file:

```bash
dqm-ml process -p my_config.yaml
```

List all available metrics and loaders registered in your environment:

```bash
dqm-ml list
```

## Commands

* process: Execute a data quality pipeline.
* list: Show available plugins.
* version: Display version information.