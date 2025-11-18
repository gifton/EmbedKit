#!/bin/bash

################################################################################
# XcodeBuildPhase.sh - Xcode Build Phase for Metal Shaders
#
# This script compiles Metal shader files (.metal) to a binary metallib for
# use in Xcode-based projects. It is designed to be called from an Xcode
# "Run Script" build phase.
#
# Features:
# - Automatic compilation on build
# - Incremental builds (timestamp-based)
# - Xcode-formatted error messages
# - Multi-platform support (macOS, iOS, tvOS, etc.)
# - Debug/Release configuration awareness
#
# Environment Variables (provided by Xcode):
# - SRCROOT: Project root directory
# - BUILT_PRODUCTS_DIR: Build output directory
# - TARGET_NAME: Target being built
# - CONFIGURATION: Debug or Release
# - PLATFORM_NAME: macosx, iphoneos, iphonesimulator, etc.
# - OBJROOT: Intermediate build files directory
#
# Xcode Integration:
# 1. Add "Run Script" build phase to your target
# 2. Set shell to: /bin/bash
# 3. Set script: ${SRCROOT}/Scripts/XcodeBuildPhase.sh
# 4. Add input files (see documentation)
# 5. Add output files (see documentation)
# 6. Move phase before "Compile Sources"
#
# Exit Codes:
# 0 - Success
# 1 - Compilation error
# 2 - Configuration error
#
# Author: EmbedKit Development Team
# Version: 1.0.0
################################################################################

set -e  # Exit immediately on error
set -u  # Error on undefined variables
set -o pipefail  # Pipe failures cause script to fail

# ============================================================================
# Configuration from Xcode Environment
# ============================================================================

# Detect script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use Xcode environment variables if available, otherwise use defaults
PROJECT_ROOT="${SRCROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SHADERS_DIR="${PROJECT_ROOT}/Sources/EmbedKit/Shaders"
KERNELS_DIR="${SHADERS_DIR}/Kernels"
COMMON_DIR="${SHADERS_DIR}/Common"

# Output location (Xcode-specific)
OUTPUT_DIR="${BUILT_PRODUCTS_DIR:-${PROJECT_ROOT}/.build/xcode}"
OUTPUT_METALLIB="${OUTPUT_DIR}/EmbedKitShaders.metallib"

# Temp directory for .air files (Xcode intermediate files)
TEMP_DIR="${OBJROOT:-${PROJECT_ROOT}/.build/xcode-temp}/metal-compile"

# Build configuration
CONFIGURATION="${CONFIGURATION:-Debug}"
PLATFORM_NAME="${PLATFORM_NAME:-macosx}"
TARGET_NAME="${TARGET_NAME:-EmbedKit}"

# Metal compiler settings based on configuration
if [ "$CONFIGURATION" = "Release" ]; then
    METAL_OPTIMIZATION="-O3"
    FAST_MATH="-ffast-math"
else
    METAL_OPTIMIZATION="-O0"
    FAST_MATH=""
fi

# Metal language version
METAL_VERSION="metal3.0"

# Expected kernel list for validation
EXPECTED_KERNELS=(
    "l2_normalize"
    "l2_normalize_batch_optimized"
    "mean_pool"
    "max_pool"
    "attention_weighted_pool"
    "cosine_similarity"
    "cosine_similarity_batch"
)

# ============================================================================
# Xcode-Formatted Output Functions
# ============================================================================

# Print error in Xcode format: <file>:<line>: error: <message>
xcode_error() {
    local file="$1"
    local line="${2:-1}"
    local message="$3"
    echo "${file}:${line}: error: ${message}" >&2
}

# Print warning in Xcode format: <file>:<line>: warning: <message>
xcode_warning() {
    local file="$1"
    local line="${2:-1}"
    local message="$3"
    echo "${file}:${line}: warning: ${message}" >&2
}

# Print note in Xcode format
xcode_note() {
    local message="$1"
    echo "note: ${message}"
}

# ============================================================================
# Validation Functions
# ============================================================================

# Check if Metal compiler is available
check_metal_compiler() {
    if ! command -v xcrun &> /dev/null; then
        xcode_error "$0" 1 "xcrun not found. Please install Xcode Command Line Tools."
        exit 2
    fi

    if ! xcrun metal --version &> /dev/null; then
        xcode_error "$0" 1 "Metal compiler not found. Please install Xcode Command Line Tools."
        exit 2
    fi
}

# Validate directory structure
check_directories() {
    if [ ! -d "$SHADERS_DIR" ]; then
        xcode_error "$SHADERS_DIR" 1 "Shaders directory not found"
        exit 2
    fi

    if [ ! -d "$KERNELS_DIR" ]; then
        xcode_error "$KERNELS_DIR" 1 "Kernels directory not found"
        exit 2
    fi

    # Create output directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$TEMP_DIR"
}

# Check if compilation is needed based on file timestamps
needs_compilation() {
    # If metallib doesn't exist, we need to compile
    if [ ! -f "$OUTPUT_METALLIB" ]; then
        return 0  # true, needs compilation
    fi

    # Get metallib modification time
    local metallib_mtime
    if [[ "$OSTYPE" == "darwin"* ]]; then
        metallib_mtime=$(stat -f %m "$OUTPUT_METALLIB" 2>/dev/null || echo 0)
    else
        metallib_mtime=$(stat -c %Y "$OUTPUT_METALLIB" 2>/dev/null || echo 0)
    fi

    # Check if any .metal file is newer than metallib
    for metal_file in "$KERNELS_DIR"/*.metal; do
        if [ -f "$metal_file" ]; then
            local metal_mtime
            if [[ "$OSTYPE" == "darwin"* ]]; then
                metal_mtime=$(stat -f %m "$metal_file")
            else
                metal_mtime=$(stat -c %Y "$metal_file")
            fi

            if [ "$metal_mtime" -gt "$metallib_mtime" ]; then
                xcode_note "Shader file modified: $(basename "$metal_file")"
                return 0  # true, needs compilation
            fi
        fi
    done

    # Check common header
    if [ -f "$COMMON_DIR/MetalCommon.h" ]; then
        local header_mtime
        if [[ "$OSTYPE" == "darwin"* ]]; then
            header_mtime=$(stat -f %m "$COMMON_DIR/MetalCommon.h")
        else
            header_mtime=$(stat -c %Y "$COMMON_DIR/MetalCommon.h")
        fi

        if [ "$header_mtime" -gt "$metallib_mtime" ]; then
            xcode_note "Common header modified: MetalCommon.h"
            return 0  # true, needs compilation
        fi
    fi

    return 1  # false, no compilation needed
}

# ============================================================================
# Compilation Functions
# ============================================================================

# Compile a single .metal file to .air
compile_metal_file() {
    local input_file="$1"
    local output_file="$2"
    local basename=$(basename "$input_file")

    xcode_note "Compiling ${basename}..."

    # Build compiler command
    local compile_cmd=(
        xcrun metal
        -std="$METAL_VERSION"
        "$METAL_OPTIMIZATION"
    )

    # Add fast-math for release builds
    if [ -n "$FAST_MATH" ]; then
        compile_cmd+=("$FAST_MATH")
    fi

    compile_cmd+=(
        -I "$COMMON_DIR"
        -c "$input_file"
        -o "$output_file"
    )

    # Execute compilation
    local compile_output
    if compile_output=$("${compile_cmd[@]}" 2>&1); then
        xcode_note "✓ Compiled ${basename}"
        return 0
    else
        # Parse error output and format for Xcode
        echo "$compile_output" | while IFS= read -r line; do
            # Try to extract file:line:col format from Metal compiler errors
            if [[ "$line" =~ ^([^:]+):([0-9]+):([0-9]+):(.*) ]]; then
                local error_file="${BASH_REMATCH[1]}"
                local error_line="${BASH_REMATCH[2]}"
                local error_msg="${BASH_REMATCH[4]}"
                xcode_error "$error_file" "$error_line" "$error_msg"
            else
                # Fallback: attribute to source file
                xcode_error "$input_file" 1 "$line"
            fi
        done
        return 1
    fi
}

# Link all .air files into .metallib
link_metallib() {
    local air_files=("$@")

    xcode_note "Linking metallib..."

    local link_cmd=(
        xcrun metallib
        "${air_files[@]}"
        -o "$OUTPUT_METALLIB"
    )

    local link_output
    if link_output=$("${link_cmd[@]}" 2>&1); then
        xcode_note "✓ Created ${OUTPUT_METALLIB}"
        return 0
    else
        xcode_error "$OUTPUT_METALLIB" 1 "Failed to create metallib: $link_output"
        return 1
    fi
}

# Validate the generated metallib
validate_metallib() {
    xcode_note "Validating metallib..."

    # Check if file exists
    if [ ! -f "$OUTPUT_METALLIB" ]; then
        xcode_error "$OUTPUT_METALLIB" 1 "Metallib file not found after compilation"
        return 1
    fi

    # Use metal-nm to list functions
    local nm_output
    if ! nm_output=$(xcrun metal-nm "$OUTPUT_METALLIB" 2>&1); then
        xcode_warning "$OUTPUT_METALLIB" 1 "Failed to validate metallib with metal-nm"
        return 0  # Non-fatal
    fi

    # Extract function names
    local found_kernels=()
    while IFS= read -r line; do
        # metal-nm output format: "address T function_name"
        if [[ $line =~ T[[:space:]](.+)$ ]]; then
            found_kernels+=("${BASH_REMATCH[1]}")
        fi
    done <<< "$nm_output"

    # Check for expected kernels
    local missing_kernels=()
    for expected in "${EXPECTED_KERNELS[@]}"; do
        local found=false
        for actual in "${found_kernels[@]}"; do
            if [ "$actual" = "$expected" ]; then
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            missing_kernels+=("$expected")
        fi
    done

    # Report results
    if [ ${#missing_kernels[@]} -gt 0 ]; then
        xcode_warning "$OUTPUT_METALLIB" 1 "Missing expected kernels: ${missing_kernels[*]}"
    else
        xcode_note "✓ All ${#EXPECTED_KERNELS[@]} expected kernels present"
    fi

    # Print file size
    local file_size
    if [[ "$OSTYPE" == "darwin"* ]]; then
        file_size=$(du -h "$OUTPUT_METALLIB" | cut -f1)
    else
        file_size=$(du -h "$OUTPUT_METALLIB" | cut -f1)
    fi
    xcode_note "Metallib size: $file_size"

    return 0
}

# Main compilation function
compile_shaders() {
    xcode_note "Metal Shader Compilation"
    xcode_note "Platform: ${PLATFORM_NAME}, Configuration: ${CONFIGURATION}"
    xcode_note "Target: ${TARGET_NAME}"

    # Find all .metal files
    local metal_files=()
    for file in "$KERNELS_DIR"/*.metal; do
        if [ -f "$file" ]; then
            metal_files+=("$file")
        fi
    done

    if [ ${#metal_files[@]} -eq 0 ]; then
        xcode_error "$KERNELS_DIR" 1 "No .metal files found in Kernels directory"
        exit 1
    fi

    xcode_note "Found ${#metal_files[@]} shader file(s)"

    # Compile each .metal file to .air
    local air_files=()
    local failed=0

    for metal_file in "${metal_files[@]}"; do
        local basename=$(basename "$metal_file" .metal)
        local air_file="${TEMP_DIR}/${basename}.air"

        if compile_metal_file "$metal_file" "$air_file"; then
            air_files+=("$air_file")
        else
            ((failed++))
        fi
    done

    # Check for compilation failures
    if [ $failed -gt 0 ]; then
        xcode_error "$KERNELS_DIR" 1 "$failed file(s) failed to compile"
        exit 1
    fi

    # Link all .air files into .metallib
    if ! link_metallib "${air_files[@]}"; then
        exit 1
    fi

    # Validate the metallib
    validate_metallib

    xcode_note "✓ Metal shader compilation complete"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    # Banner
    xcode_note "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    xcode_note "EmbedKit Metal Shader Build Phase"
    xcode_note "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Pre-flight checks
    check_metal_compiler
    check_directories

    # Check if compilation is needed
    if needs_compilation; then
        xcode_note "Changes detected, compiling shaders..."
        compile_shaders
    else
        xcode_note "✓ Metal shaders are up-to-date, skipping compilation"
        xcode_note "Output: ${OUTPUT_METALLIB}"
    fi

    xcode_note "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Run main function
main

exit 0
