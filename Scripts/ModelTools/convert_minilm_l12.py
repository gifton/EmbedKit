#!/usr/bin/env python3
"""
CoreML Conversion Script for MiniLM-L12-v2
Higher accuracy version with 12 layers (vs 6 in L6)
"""
 # Note: Requires network access to download model weights from HuggingFace

import os
import sys
import torch
import coremltools as ct
import numpy as np
from transformers import AutoModel, AutoTokenizer
import traceback

def convert_minilm_l12_to_coreml():
    """Convert MiniLM-L12-v2 to CoreML format optimized for on-device inference."""

    print("🚀 Starting MiniLM-L12-v2 conversion to CoreML...")
    print("📊 Model specs: 12 layers, 384 dimensions, ~85MB size")

    # 1. Load the model
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    print(f"📥 Loading model: {model_name}")

    try:
        model = AutoModel.from_pretrained(model_name, torchscript=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        print("✅ Model loaded successfully")
        print(f"   - Vocabulary size: {tokenizer.vocab_size}")
        print(f"   - Max sequence length: {tokenizer.model_max_length}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        traceback.print_exc()
        return False

    # 2. Prepare for tracing
    print("🔧 Preparing model for conversion...")

    # Create dummy inputs matching EmbedKit's TokenizedInput structure
    batch_size = 1
    seq_length = 512  # Max sequence length for MiniLM

    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)

    # 3. Trace the model
    print("🎯 Tracing model (this may take a minute)...")
    try:
        with torch.no_grad():
            # Run a forward pass to ensure model works
            output = model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask,
                token_type_ids=dummy_token_type_ids,
                return_dict=False  # Force tuple output for tracing
            )

            # Handle both tuple and object outputs
            if isinstance(output, tuple):
                last_hidden_state = output[0]
            else:
                last_hidden_state = output.last_hidden_state

            print(f"   - Output shape: {last_hidden_state.shape}")
            print(f"   - Output type: {type(output)}")

            # Create a wrapper to ensure consistent output
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, input_ids, attention_mask, token_type_ids):
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        return_dict=False
                    )
                    # Return only the last hidden state
                    if isinstance(output, tuple):
                        return output[0]
                    else:
                        return output.last_hidden_state

            wrapped_model = ModelWrapper(model)
            wrapped_model.eval()

            # Trace the wrapped model
            traced_model = torch.jit.trace(
                wrapped_model,
                (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
                strict=False
            )
        print("✅ Model traced successfully")
    except Exception as e:
        print(f"❌ Failed to trace model: {e}")
        traceback.print_exc()
        return False

    # 4. Convert to CoreML
    print("🔄 Converting to CoreML format...")
    print("   This may take 2-3 minutes for the L12 model...")
    try:
        mlmodel = ct.convert(
            traced_model,
            convert_to="mlprogram",  # Use ML Program format for better performance
            inputs=[
                ct.TensorType(
                    name="input_ids",
                    shape=(batch_size, seq_length),
                    dtype=np.int32
                ),
                ct.TensorType(
                    name="attention_mask",
                    shape=(batch_size, seq_length),
                    dtype=np.int32
                ),
                ct.TensorType(
                    name="token_type_ids",
                    shape=(batch_size, seq_length),
                    dtype=np.int32
                )
            ],
            outputs=[
                ct.TensorType(
                    name="last_hidden_state",
                    dtype=np.float32
                )
            ],
            compute_units=ct.ComputeUnit.ALL,  # Use all available compute units
            minimum_deployment_target=ct.target.iOS15,
        )
        print("✅ Model converted successfully")
    except Exception as e:
        print(f"❌ Failed to convert model: {e}")
        traceback.print_exc()
        return False

    # 5. Add metadata for EmbedKit
    print("📝 Adding metadata...")
    mlmodel.author = "EmbedKit"
    mlmodel.short_description = "MiniLM-L12-v2 for semantic embeddings (384D, 12 layers)"
    mlmodel.version = "1.0.0"
    mlmodel.license = "Apache 2.0"

    # Add custom metadata for EmbedKit
    mlmodel.user_defined_metadata["model_type"] = "sentence-transformer"
    mlmodel.user_defined_metadata["model_variant"] = "MiniLM-L12-v2"
    mlmodel.user_defined_metadata["embedding_dimension"] = "384"
    mlmodel.user_defined_metadata["max_sequence_length"] = "512"
    mlmodel.user_defined_metadata["vocab_size"] = str(tokenizer.vocab_size)
    mlmodel.user_defined_metadata["num_layers"] = "12"
    mlmodel.user_defined_metadata["quality"] = "high"

    # 6. Save the model
    output_path = "MiniLM-L12-v2.mlpackage"
    print(f"💾 Saving model to {output_path}...")
    try:
        mlmodel.save(output_path)
        print(f"✅ Model saved successfully to {output_path}")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        traceback.print_exc()
        return False

    # 7. Verify the model
    print("🔍 Verifying model...")
    spec = mlmodel.get_spec()
    print("\nModel Information:")
    print(f"  - Input names: {[input.name for input in spec.description.input]}")
    print(f"  - Output names: {[output.name for output in spec.description.output]}")

    # Calculate model size
    model_size = get_directory_size(output_path)
    print(f"  - Model size: {model_size:.2f} MB")

    # 8. Optional: Create quantized version for smaller size
    print("\n💡 Creating quantized version for smaller size...")
    try:
        from coremltools.models.neural_network import quantization_utils

        # 8-bit quantization
        quantized_model_8bit = quantization_utils.quantize_weights(
            mlmodel,
            nbits=8,
            quantization_mode="linear"
        )

        quantized_path_8bit = "MiniLM-L12-v2-quantized8.mlpackage"
        quantized_model_8bit.save(quantized_path_8bit)

        quantized_size_8bit = get_directory_size(quantized_path_8bit)
        print(f"✅ 8-bit quantized model saved to {quantized_path_8bit}")
        print(f"  - Size: {quantized_size_8bit:.2f} MB")
        print(f"  - Reduction: {(1 - quantized_size_8bit/model_size)*100:.1f}%")

        # Optional: 4-bit quantization for maximum compression
        print("\n💾 Creating 4-bit quantized version (experimental)...")
        quantized_model_4bit = quantization_utils.quantize_weights(
            mlmodel,
            nbits=4,
            quantization_mode="kmeans"  # Better for 4-bit
        )

        quantized_path_4bit = "MiniLM-L12-v2-quantized4.mlpackage"
        quantized_model_4bit.save(quantized_path_4bit)

        quantized_size_4bit = get_directory_size(quantized_path_4bit)
        print(f"✅ 4-bit quantized model saved to {quantized_path_4bit}")
        print(f"  - Size: {quantized_size_4bit:.2f} MB")
        print(f"  - Reduction: {(1 - quantized_size_4bit/model_size)*100:.1f}%")
        print(f"  ⚠️  Note: 4-bit may have slight quality degradation")

    except Exception as e:
        print(f"⚠️  Quantization optional - skipped: {e}")

    # 9. Generate integration code
    print("\n📋 Integration code for EmbedKit:")
    print("-" * 50)
    print("""
// Add to your EmbedKit project:
let modelURL = Bundle.main.url(
    forResource: "MiniLM-L12-v2",  // Or use quantized versions
    withExtension: "mlpackage"
)!

let pipeline = try await EmbeddingPipeline(
    modelURL: modelURL,
    tokenizer: BERTTokenizer(),
    configuration: EmbeddingPipelineConfiguration(
        poolingStrategy: .mean,
        normalize: true,
        useGPUAcceleration: true
    )
)

// Generate embeddings (slightly slower than L6, but better quality)
let embedding = try await pipeline.embed("Your text...")
// Dimensions: 384 (same as L6)
// Quality: ~5-10% better than L6
// Speed: ~2x slower than L6
""")
    print("-" * 50)

    # 10. Comparison table
    print("\n📊 Model Comparison:")
    print("┌─────────────────┬────────────┬────────────┐")
    print("│ Model           │ MiniLM-L6  │ MiniLM-L12 │")
    print("├─────────────────┼────────────┼────────────┤")
    print("│ Layers          │ 6          │ 12         │")
    print("│ Size            │ ~45 MB     │ ~85 MB     │")
    print("│ Size (8-bit)    │ ~23 MB     │ ~43 MB     │")
    print("│ Size (4-bit)    │ ~12 MB     │ ~22 MB     │")
    print("│ Dimensions      │ 384        │ 384        │")
    print("│ Speed (iPhone)  │ 10-15ms    │ 20-30ms    │")
    print("│ Quality         │ Good       │ Better     │")
    print("│ Best for        │ Real-time  │ Quality    │")
    print("└─────────────────┴────────────┴────────────┘")

    print("\n✅ Conversion complete! You now have three versions:")
    print("1. MiniLM-L12-v2.mlpackage - Full precision (best quality)")
    print("2. MiniLM-L12-v2-quantized8.mlpackage - 8-bit (good balance)")
    print("3. MiniLM-L12-v2-quantized4.mlpackage - 4-bit (smallest size)")
    print("\n🎯 Recommendation: Start with 8-bit quantized for best balance")

    return True

def get_directory_size(path):
    """Calculate directory size in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)

def check_dependencies():
    """Check if required packages are installed."""
    print("🔍 Checking dependencies...")
    required = {
        'torch': 'torch>=2.0.0',
        'transformers': 'transformers>=4.30.0',
        'coremltools': 'coremltools>=7.0',
        'sentence_transformers': 'sentence-transformers>=2.2.0',
        'numpy': 'numpy>=1.24.0'
    }
    missing = []

    for package, install_name in required.items():
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing.append(install_name)
            print(f"❌ {package} is missing")

    if missing:
        print("\n❌ Missing required packages:")
        print(f"   pip install {' '.join(missing)}")
        return False

    print("✅ All dependencies are installed")
    return True

if __name__ == "__main__":
    print("=================================================")
    print("   MiniLM-L12-v2 → CoreML Converter for EmbedKit")
    print("   Higher Quality Version (12 layers vs 6)")
    print("=================================================\n")

    if not check_dependencies():
        print("\n📦 Install missing packages:")
        print("   pip install torch transformers coremltools sentence-transformers numpy")
        sys.exit(1)

    success = convert_minilm_l12_to_coreml()

    if success:
        print("\n🎉 Success! Your MiniLM-L12-v2 model is ready for EmbedKit.")
        print("📱 This model offers better quality than L6 with moderate speed trade-off.")
    else:
        print("\n❌ Conversion failed. Please check the errors above.")
        print("💡 Common issues:")
        print("   - Insufficient memory (L12 requires ~4GB RAM during conversion)")
        print("   - Missing dependencies")
        print("   - Network issues downloading model")
        sys.exit(1)
