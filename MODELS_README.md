# EmbedKit Models

## Important: Models Not Included

The CoreML model files required for EmbedKit are not included in this repository due to their size (173MB+). This keeps the repository lightweight and fast to clone.

## Obtaining Models

### Option 1: Download Pre-converted Models (Recommended)
Pre-converted CoreML models optimized for EmbedKit will be available from:
- GitHub Releases page (coming soon)
- Direct download links in documentation

### Option 2: Convert Your Own Models
Use the provided conversion scripts in the `Scripts/` directory:

```bash
cd Scripts
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python convert_to_coreml.py <your_model> ../Models/<output_name>
```

### Option 3: Use Model Providers
Some model providers offer direct CoreML exports. Check:
- Hugging Face Model Hub (filter by CoreML)
- Apple's Core ML Models gallery

## Model Directory Structure

Once obtained, place models in the following structure:
```
Models/
├── sentence-transformers_all-MiniLM-L6-v2/
│   ├── model.mlpackage/
│   ├── model_info.json
│   └── tokenizer_config.json
├── BAAI_bge-small-en-v1.5/
│   └── ...
└── thenlper_gte-small/
    └── ...
```

## Supported Models

EmbedKit has been tested with:
- sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- BAAI/bge-small-en-v1.5 (384 dimensions)
- thenlper/gte-small (384 dimensions)

## Model Requirements

- CoreML format (.mlpackage or .mlmodelc)
- Text embedding models with fixed output dimensions
- Compatible tokenizer configuration

## Git LFS (Alternative)

For teams that prefer to version control models, consider using Git LFS:
```bash
git lfs track "*.mlpackage"
git lfs track "*.mlmodel"
git add .gitattributes
```

Note: This will increase repository size for all users.