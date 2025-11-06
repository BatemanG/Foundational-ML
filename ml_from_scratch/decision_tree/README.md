# CW-70050-Decision-Trees


## 🚀 Getting Started

Follow these instructions to set up the development environment.

### Prerequisites
Before you begin, ensure you have the following tools installed:

1. `git`
2. `pyenv` for managing Python versions.

#### Setup Instructions

1. Clone the repository: 

```bash
git clone git@gitlab.doc.ic.ac.uk:ak7025/cw-70050-decision-trees.git

cd cw-70050-decision-trees
```

2. Install the correct Python version: pyenv will automatically read the .python-version file in this repository. Run the following command to install the required version:

```bash
pyenv install
```

3. Create and activate the virtual environment:

```bash
# Create the virtual environment using the pyenv-managed Python
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

4. Install project dependencies:
```bash
pip install -r requirements.txt
```

5. You're all set! You can now run the application:
```bash
python3 main.py
```

