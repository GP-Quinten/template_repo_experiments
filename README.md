# ML Experiments

This repository contains machine learning experiments and shared resources as template repo/project. It provides a structured approach to running reproducible ML workflows, managing shared datasets, and leveraging common libraries and scripts.

## Structure Overview

### `datasets`

- Contains datasets shared across multiple experiments.
- Managed using [Data Version Control (DVC)](https://dvc.org/) for efficient version tracking and reproducibility.

### `projects`

- Independent projects focused on specific machine learning topics.
- Each project is self-contained and includes:
  - **Untracked Data**: Specific datasets unique to the project (not managed by DVC).
  - **Reproducible Workflows**: Defined using [Snakemake](https://snakemake.readthedocs.io/en/stable/) for automated, scalable, and reproducible analysis.

**Example Project (`P00_template`):**

```
projects/
└── P00_template/
    ├── README.md
    ├── Snakefile
    ├── data
    │   ├── D00_random_data
    │   ├── R01_generate_dataset
    │   └── R02_clean_dataset
    ├── notebooks
    │   └── N01_hello_word.ipynb
    ├── poetry.lock
    ├── pyproject.toml
    └── src
```

For more details, see the [README](projects/P00_template/README.md) of the `P00_template` project.

### Shared Code

- Shared code is available at:
  - `src/more_europa`: Common modules, helpers, and settings.
  - `src/llm_inference`: Modules for inference using language models.

## Testing

Tests are available under the `tests/` directory and can be run with:

```bash
pytest
```

---

Feel free to contribute by improving the existing codebase or adding new projects related to the More Europa ML initiative.

