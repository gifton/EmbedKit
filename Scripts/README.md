# EmbedKit Scripts

This directory contains utility scripts for working with EmbedKit.

## Model Conversion Scripts

### convert_to_coreml.py
Python script for converting machine learning models to CoreML format compatible with EmbedKit.

**Requirements:**
- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

**Usage:**
```bash
python convert_to_coreml.py <input_model> <output_path>
```

### quick_convert.sh
Bash wrapper script for quick model conversion.

**Usage:**
```bash
./quick_convert.sh <model_name>
```

## Model Downloads

**Important:** Model files are not included in the repository due to their size (173MB+). 

To obtain models for EmbedKit:
1. Download pre-converted models from [releases page] (coming soon)
2. Or convert your own models using the provided scripts
3. Place models in a `Models/` directory (which is git-ignored)

## Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```