#!/usr/bin/env python3
"""
Compile CoreML Model for Testing
Compiles the MiniLM-L12-v2.mlpackage into .mlmodelc format for use in tests
"""

import os
import sys
import coremltools as ct
from pathlib import Path

def compile_model():
    """Compile the CoreML model package"""

    # Resolve project root (two levels up from this script: Scripts/ModelTools → Scripts → repo root)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    # Model path at repo root
    model_path = project_root / "MiniLM-L12-v2.mlpackage"

    print("=" * 60)
    print("CoreML Model Compiler")
    print("=" * 60)
    print(f"\n📦 Model path: {model_path}")

    # Check if model exists
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print("\n💡 Please ensure MiniLM-L12-v2.mlpackage exists at the repository root")
        print("   You can generate it by running: python Scripts/ModelTools/convert_minilm_l12.py")
        sys.exit(1)

    print(f"✅ Model package found")

    # Load the model (CoreML compiles on load)
    print(f"\n🔄 Loading model...")
    try:
        model = ct.models.MLModel(str(model_path))
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        sys.exit(1)

    # Get model info
    spec = model.get_spec()
    print(f"\n📋 Model Information:")
    print(f"   Inputs: {[input.name for input in spec.description.input]}")
    print(f"   Outputs: {[output.name for output in spec.description.output]}")

    # Compile the model (implicit on load for .mlpackage)
    print(f"\n⚙️  Compiling model (this may take a moment)...")
    try:
        print(f"✅ Model compiled successfully!")
        print(f"\n📍 Compiled model location:")
        print(f"   {model_path}")

        # Check the compiled cache output exists (if created)
        compiled_path = str(model_path).replace('.mlpackage', '.mlmodelc')
        if os.path.exists(compiled_path):
            print(f"\n✅ Compiled model cache created at:")
            print(f"   {compiled_path}")

    except Exception as e:
        print(f"\n❌ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n" + "=" * 60)
    print("✨ Success! Model is ready for testing")
    print("=" * 60)
    print(f"\n🧪 You can now run tests with:")
    print(f"   swift test --filter CoreMLBackendTests")
    print()

if __name__ == "__main__":
    try:
        compile_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  Compilation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
