#!/usr/bin/env python3
"""
Convert HuggingFace embedding models to Core ML format.
Optimized for on-device inference with EmbedKit.
"""

import os
import json
import argparse
import numpy as np
import torch
import coremltools as ct
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional


class EmbeddingModelConverter:
    """Convert HuggingFace sentence embedding models to Core ML."""
    
    # Popular embedding models with their configurations
    SUPPORTED_MODELS = {
        "sentence-transformers/all-MiniLM-L6-v2": {
            "max_length": 256,
            "embedding_dim": 384,
            "pooling": "mean",
            "normalize": True
        },
        "sentence-transformers/all-mpnet-base-v2": {
            "max_length": 384,
            "embedding_dim": 768,
            "pooling": "mean",
            "normalize": True
        },
        "BAAI/bge-small-en-v1.5": {
            "max_length": 512,
            "embedding_dim": 384,
            "pooling": "cls",
            "normalize": True
        },
        "BAAI/bge-base-en-v1.5": {
            "max_length": 512,
            "embedding_dim": 768,
            "pooling": "cls",
            "normalize": True
        },
        "thenlper/gte-small": {
            "max_length": 512,
            "embedding_dim": 384,
            "pooling": "mean",
            "normalize": True
        }
    }
    
    def __init__(self, model_name: str, max_length: Optional[int] = None):
        self.model_name = model_name
        self.config = self.SUPPORTED_MODELS.get(model_name, {})
        self.max_length = max_length or self.config.get("max_length", 256)
        
        print(f"📦 Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
    def create_traced_model(self) -> torch.jit.ScriptModule:
        """Create a traced version of the model suitable for Core ML conversion."""
        
        class EmbeddingWrapper(torch.nn.Module):
            def __init__(self, model, pooling_strategy="mean", normalize=True):
                super().__init__()
                self.model = model
                self.pooling_strategy = pooling_strategy
                self.normalize = normalize
                
            def forward(self, input_ids, attention_mask):
                # Get model outputs
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Extract token embeddings
                token_embeddings = outputs.last_hidden_state
                
                # Apply pooling
                if self.pooling_strategy == "cls":
                    # Use CLS token (first token)
                    pooled = token_embeddings[:, 0, :]
                elif self.pooling_strategy == "mean":
                    # Mean pooling with attention mask
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    pooled = sum_embeddings / sum_mask
                else:
                    # Max pooling
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    token_embeddings[input_mask_expanded == 0] = -1e9
                    pooled = torch.max(token_embeddings, 1)[0]
                
                # Normalize if configured
                if self.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                
                return pooled
        
        # Create wrapper model
        wrapper = EmbeddingWrapper(
            self.model, 
            self.config.get("pooling", "mean"),
            self.config.get("normalize", True)
        )
        wrapper.eval()
        
        # Create sample inputs
        sample_inputs = self.tokenizer(
            ["Sample text for tracing"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Trace the model
        print("🔧 Tracing model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(
                wrapper,
                (sample_inputs["input_ids"], sample_inputs["attention_mask"])
            )
        
        return traced_model
    
    def convert_to_coreml(self, traced_model: torch.jit.ScriptModule) -> ct.models.MLModel:
        """Convert traced PyTorch model to Core ML."""
        
        print("🔄 Converting to Core ML...")
        
        # Define input types
        inputs = [
            ct.TensorType(
                name="input_ids",
                shape=(1, self.max_length),
                dtype=np.int32
            ),
            ct.TensorType(
                name="attention_mask",
                shape=(1, self.max_length),
                dtype=np.int32
            )
        ]
        
        # Convert model
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=[ct.TensorType(name="embeddings", dtype=np.float32)],
            compute_precision=ct.precision.FLOAT16,  # Use FP16 for efficiency
            minimum_deployment_target=ct.target.iOS16,  # iOS 16+ for latest optimizations
            compute_units=ct.ComputeUnit.ALL  # Use Neural Engine when available
        )
        
        # Add metadata
        mlmodel.author = "EmbedKit Converter"
        mlmodel.short_description = f"Embedding model: {self.model_name}"
        mlmodel.version = "1.0"
        
        # Add custom metadata for EmbedKit
        mlmodel.user_defined_metadata["embedding_dim"] = str(self.config.get("embedding_dim", 384))
        mlmodel.user_defined_metadata["max_length"] = str(self.max_length)
        mlmodel.user_defined_metadata["pooling_strategy"] = self.config.get("pooling", "mean")
        mlmodel.user_defined_metadata["normalize"] = str(self.config.get("normalize", True))
        mlmodel.user_defined_metadata["source_model"] = self.model_name
        
        return mlmodel
    
    def save_tokenizer_config(self, output_dir: str):
        """Save tokenizer configuration for Swift."""
        
        print("💾 Saving tokenizer configuration...")
        
        # Get tokenizer vocab
        vocab = self.tokenizer.get_vocab()
        
        # Create tokenizer config
        tokenizer_config = {
            "type": "wordpiece" if "bert" in self.model_name.lower() else "sentencepiece",
            "vocab_size": len(vocab),
            "max_length": self.max_length,
            "padding_token": self.tokenizer.pad_token,
            "unk_token": self.tokenizer.unk_token,
            "cls_token": getattr(self.tokenizer, "cls_token", None),
            "sep_token": getattr(self.tokenizer, "sep_token", None),
            "mask_token": getattr(self.tokenizer, "mask_token", None),
            "vocab": vocab
        }
        
        # Save config
        config_path = os.path.join(output_dir, "tokenizer_config.json")
        with open(config_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        
        # Save special tokens
        special_tokens = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "unk_token_id": self.tokenizer.unk_token_id,
            "cls_token_id": getattr(self.tokenizer, "cls_token_id", None),
            "sep_token_id": getattr(self.tokenizer, "sep_token_id", None),
            "mask_token_id": getattr(self.tokenizer, "mask_token_id", None)
        }
        
        special_tokens_path = os.path.join(output_dir, "special_tokens.json")
        with open(special_tokens_path, "w") as f:
            json.dump(special_tokens, f, indent=2)
            
    def convert(self, output_path: str):
        """Complete conversion pipeline."""
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Trace model
        traced_model = self.create_traced_model()
        
        # Convert to Core ML
        mlmodel = self.convert_to_coreml(traced_model)
        
        # Save Core ML model
        model_path = os.path.join(output_path, "model.mlpackage")
        mlmodel.save(model_path)
        print(f"✅ Core ML model saved to: {model_path}")
        
        # Save tokenizer config
        self.save_tokenizer_config(output_path)
        print(f"✅ Tokenizer config saved to: {output_path}")
        
        # Create model info file
        model_info = {
            "source_model": self.model_name,
            "embedding_dimensions": self.config.get("embedding_dim", 384),
            "max_sequence_length": self.max_length,
            "pooling_strategy": self.config.get("pooling", "mean"),
            "normalize_embeddings": self.config.get("normalize", True),
            "model_size_mb": os.path.getsize(model_path) / (1024 * 1024)
        }
        
        info_path = os.path.join(output_path, "model_info.json")
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"\n📊 Model Statistics:")
        print(f"   - Embedding dimensions: {model_info['embedding_dimensions']}")
        print(f"   - Max sequence length: {model_info['max_sequence_length']}")
        print(f"   - Model size: {model_info['model_size_mb']:.1f} MB")
        print(f"   - Pooling: {model_info['pooling_strategy']}")
        print(f"   - Normalized: {model_info['normalize_embeddings']}")


def test_converted_model(model_path: str):
    """Test the converted Core ML model."""
    
    print("\n🧪 Testing converted model...")
    
    # Load model
    model = ct.models.MLModel(model_path)
    
    # Get model info
    print(f"Inputs: {model.get_spec().description.input}")
    print(f"Outputs: {model.get_spec().description.output}")
    
    # Create test inputs
    test_input_ids = np.random.randint(0, 1000, size=(1, 256), dtype=np.int32)
    test_attention_mask = np.ones((1, 256), dtype=np.int32)
    
    # Run inference
    predictions = model.predict({
        "input_ids": test_input_ids,
        "attention_mask": test_attention_mask
    })
    
    embeddings = predictions["embeddings"]
    print(f"\n✅ Test successful!")
    print(f"   Output shape: {embeddings.shape}")
    print(f"   Output dtype: {embeddings.dtype}")
    print(f"   Sample values: {embeddings[0][:5]}")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace models to Core ML")
    parser.add_argument(
        "model_name",
        type=str,
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./CoreMLModels",
        help="Output directory for Core ML model"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum sequence length (default: model-specific)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the converted model"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List supported models"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("📋 Supported models:")
        for model_name, config in EmbeddingModelConverter.SUPPORTED_MODELS.items():
            print(f"   - {model_name}")
            print(f"     Dimensions: {config['embedding_dim']}, Max length: {config['max_length']}")
        return
    
    # Create converter
    converter = EmbeddingModelConverter(args.model_name, args.max_length)
    
    # Convert model
    output_dir = os.path.join(args.output, args.model_name.replace("/", "_"))
    converter.convert(output_dir)
    
    # Test if requested
    if args.test:
        model_path = os.path.join(output_dir, "model.mlpackage")
        test_converted_model(model_path)


if __name__ == "__main__":
    main()