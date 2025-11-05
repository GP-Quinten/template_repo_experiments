#!/usr/bin/env bash
set -e

# Check if pyenv is installed
if ! command -v pyenv >/dev/null; then
    echo "pyenv is not installed. Please install pyenv first." >&2
    exit 1
fi

# Load pyenv and the virtualenv commands
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Desired Python version prefix
PYTHON_VERSION_PREFIX="3.11"

# Check if a Python version starting with 3.11 exists in pyenv and select the first one found
PYTHON_VERSION=$(pyenv versions --bare | grep "^${PYTHON_VERSION_PREFIX}" | head -n 1)
if [ -z "$PYTHON_VERSION" ]; then
    echo "No Python version starting with ${PYTHON_VERSION_PREFIX} is installed in pyenv. Please install it and try again." >&2
    exit 1
fi

echo "Using Python version: ${PYTHON_VERSION}"

# Create a virtual environment named after the project folder (project name + "-env")
PROJECT_NAME=$(basename "$PWD")
ENV_NAME="${PROJECT_NAME}-env"
ENV_PATH="$HOME/.pyenv/versions/${PYTHON_VERSION}/envs/${ENV_NAME}"

# Check if the virtual environment already exists
if pyenv virtualenvs --bare | grep -q "^${ENV_NAME}$"; then
    echo "Virtual environment ${ENV_NAME} already exists."
else
    echo "Creating virtual environment ${ENV_NAME} using Python ${PYTHON_VERSION}..."
    pyenv virtualenv "${PYTHON_VERSION}" "${ENV_NAME}"
fi

# Deactivate any currently active virtual environment
# echo "Deactivating any active virtual environment..."
# deactivate 2>/dev/null || true
# pyenv deactivate 2>/dev/null || true

# Set the local pyenv version to the newly created environment (writes to .python-version)
pyenv local "${ENV_NAME}"
echo "Set local pyenv environment: ${ENV_NAME}"

# Check if the correct virtual environment is already activated
CURRENT_ENV=$(pyenv version-name)
if [ "$CURRENT_ENV" = "$ENV_NAME" ]; then
    echo "Virtual environment ${ENV_NAME} is already activated."
else
    echo "Activating virtual environment: ${ENV_NAME}"
    pyenv activate "${ENV_NAME}"
fi

# Validate Poetry Environment
echo "Checking Poetry environment..."
POETRY_ENV_INFO=$(poetry env info 2>/dev/null)

# Extract only the first "Path:" line, which corresponds to the virtual environment path
POETRY_ENV_PATH=$(echo "$POETRY_ENV_INFO" | awk '/Path:/ {print $2; exit}')
VALID_ENV=$(echo "$POETRY_ENV_INFO" | awk '/Valid:/ {print $2}')

if [[ "$VALID_ENV" != "True" ]]; then
    echo "Error: Poetry reports that the environment is not valid." >&2
    echo "$POETRY_ENV_INFO" >&2
    exit 1
fi

if [[ "$POETRY_ENV_PATH" != "$ENV_PATH" ]]; then
    echo "Error: Poetry environment path ($POETRY_ENV_PATH) does not match expected path ($ENV_PATH)." >&2
    echo "$POETRY_ENV_INFO" >&2
    exit 1
fi

echo "Verified correct Poetry virtual environment: $POETRY_ENV_PATH"

# Check if Poetry is installed after virtualenv verification
if ! command -v poetry >/dev/null; then
    echo "Poetry is not installed. Please install Poetry first." >&2
    exit 1
fi

# Run poetry install after successful virtual environment verification
echo "Running 'poetry install'..."
poetry install

echo "Setup complete."
