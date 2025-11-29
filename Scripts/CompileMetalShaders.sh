#!/bin/bash

################################################################################
# CompileMetalShaders.sh
#
# Compiles Metal shader files (.metal) to a binary metallib for EmbedKit.
#
# This script:
# 1. Finds all .metal files in Sources/EmbedKit/Shaders/Kernels/
# 2. Compiles each to .air (Metal intermediate representation)
# 3. Links all .air files into EmbedKitShaders.metallib
# 4. Outputs metallib to Sources/EmbedKit/Resources/
# 5. Validates the metallib contains all expected kernels
#
# Usage:
#   ./Scripts/CompileMetalShaders.sh [-v|--verbose] [-c|--clean]
#
# Options:
#   -v, --verbose    Enable verbose output
#   -c, --clean      Clean intermediate files after compilation
#   -h, --help       Show this help message
#
# Requirements:
#   - macOS with Xcode Command Line Tools
#   - xcrun metal compiler
#
# Author: EmbedKit Development Team
# Version: 1.0.0
################################################################################

set -e  # Exit immediately on error
set -u  # Error on undefined variables
set -o pipefail  # Pipe failures cause script to fail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHADERS_DIR="${PROJECT_ROOT}/Sources/EmbedKit/Shaders"
KERNELS_DIR="${SHADERS_DIR}/Kernels"
RESOURCES_DIR="${PROJECT_ROOT}/Sources/EmbedKit/Resources"
TEMP_DIR="${PROJECT_ROOT}/.build/metal-temp"

OUTPUT_METALLIB="${RESOURCES_DIR}/EmbedKitShaders.metallib"
METALLIB_NAME="EmbedKitShaders"

# Metal compiler settings
METAL_VERSION="metal3.0"
OPTIMIZATION_LEVEL="-O3"

# Expected kernels (for validation)
EXPECTED_KERNELS=(
    # Legacy kernels (single-item operations)
    "l2_normalize"
    "l2_normalize_batch_optimized"
    "mean_pool"
    "max_pool"
    "attention_weighted_pool"
    "cosine_similarity"
    "cosine_similarity_batch"

    # Metal 4 Tensor Pooling kernels (Phase 3)
    "tensor_mean_pool"
    "tensor_max_pool"
    "tensor_cls_pool"
    "tensor_pool_unified"
    "tensor_mean_pool_cooperative"
    "tensor_attention_pool"

    # Metal 4 Tensor Normalization kernels (Phase 3)
    "tensor_l2_normalize_with_norms"
    "tensor_compute_norms"
    "tensor_l2_normalize_fused"
    "tensor_l2_normalize_stable"
    "tensor_l2_normalize_inplace"

    # Metal 4 Fused Operations kernels (Phase 3)
    "fused_mean_pool_normalize"
    "fused_max_pool_normalize"
    "fused_pool_normalize_unified"
    "fused_attention_pool_normalize"
    "tensor_similarity_matrix_normalized"
    "tensor_similarity_matrix_full"
    "fused_embed_compare_pipeline"
)

# ============================================================================
# Colors for Output
# ============================================================================

if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m'  # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    BOLD=''
    NC=''
fi

# ============================================================================
# Command Line Options
# ============================================================================

VERBOSE=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            head -n 30 "$0" | grep "^#" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Error:${NC} Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BOLD}[VERBOSE]${NC} $1"
    fi
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

check_metal_compiler() {
    log_info "Checking for Metal compiler..."

    if ! command -v xcrun &> /dev/null; then
        log_error "xcrun not found. Please install Xcode Command Line Tools."
        log_error "Run: xcode-select --install"
        exit 1
    fi

    if ! xcrun metal --version &> /dev/null; then
        log_error "Metal compiler not found."
        log_error "Please ensure Xcode Command Line Tools are properly installed."
        exit 1
    fi

    local metal_version=$(xcrun metal --version 2>&1 | head -n 1)
    log_verbose "Metal compiler: $metal_version"
    log_success "Metal compiler available"
}

check_directories() {
    log_info "Checking directory structure..."

    if [[ ! -d "$SHADERS_DIR" ]]; then
        log_error "Shaders directory not found: $SHADERS_DIR"
        exit 1
    fi

    if [[ ! -d "$KERNELS_DIR" ]]; then
        log_error "Kernels directory not found: $KERNELS_DIR"
        exit 1
    fi

    # Create output directories if they don't exist
    mkdir -p "$RESOURCES_DIR"
    mkdir -p "$TEMP_DIR"

    log_verbose "Shaders dir: $SHADERS_DIR"
    log_verbose "Kernels dir: $KERNELS_DIR"
    log_verbose "Resources dir: $RESOURCES_DIR"
    log_verbose "Temp dir: $TEMP_DIR"

    log_success "Directory structure validated"
}

find_metal_files() {
    log_info "Finding Metal shader files..."

    METAL_FILES=($(find "$KERNELS_DIR" -name "*.metal" -type f | sort))

    if [[ ${#METAL_FILES[@]} -eq 0 ]]; then
        log_error "No .metal files found in $KERNELS_DIR"
        exit 1
    fi

    log_success "Found ${#METAL_FILES[@]} Metal shader file(s):"
    for file in "${METAL_FILES[@]}"; do
        local basename=$(basename "$file")
        echo "  • $basename"
    done
}

# ============================================================================
# Compilation Functions
# ============================================================================

compile_metal_file() {
    local input_file="$1"
    local output_file="$2"
    local basename=$(basename "$input_file")

    log_info "Compiling $basename..."

    local compile_cmd=(
        xcrun metal
        -std="$METAL_VERSION"
        "$OPTIMIZATION_LEVEL"
        -ffast-math
        -c "$input_file"
        -o "$output_file"
    )

    if [[ "$VERBOSE" == "true" ]]; then
        log_verbose "Command: ${compile_cmd[*]}"
    fi

    local compile_output
    if compile_output=$("${compile_cmd[@]}" 2>&1); then
        log_success "Compiled $basename"
        if [[ -n "$compile_output" && "$VERBOSE" == "true" ]]; then
            echo "$compile_output" | sed 's/^/    /'
        fi
        return 0
    else
        log_error "Failed to compile $basename"
        echo "$compile_output" | sed 's/^/    /'
        return 1
    fi
}

compile_all_shaders() {
    log_info "Compiling all Metal shaders..."

    local air_files=()
    local failed=0

    for metal_file in "${METAL_FILES[@]}"; do
        local basename=$(basename "$metal_file" .metal)
        local air_file="${TEMP_DIR}/${basename}.air"

        if compile_metal_file "$metal_file" "$air_file"; then
            air_files+=("$air_file")
        else
            ((failed++))
        fi
    done

    if [[ $failed -gt 0 ]]; then
        log_error "$failed file(s) failed to compile"
        exit 1
    fi

    AIR_FILES=("${air_files[@]}")
    log_success "All shaders compiled successfully"
}

# ============================================================================
# Linking Functions
# ============================================================================

link_metallib() {
    log_info "Linking metallib..."

    if [[ ${#AIR_FILES[@]} -eq 0 ]]; then
        log_error "No .air files to link"
        exit 1
    fi

    local link_cmd=(
        xcrun metallib
        "${AIR_FILES[@]}"
        -o "$OUTPUT_METALLIB"
    )

    if [[ "$VERBOSE" == "true" ]]; then
        log_verbose "Command: ${link_cmd[*]}"
        log_verbose "Output: $OUTPUT_METALLIB"
    fi

    local link_output
    if link_output=$("${link_cmd[@]}" 2>&1); then
        log_success "Metallib created: $(basename "$OUTPUT_METALLIB")"
        if [[ -n "$link_output" && "$VERBOSE" == "true" ]]; then
            echo "$link_output" | sed 's/^/    /'
        fi
    else
        log_error "Failed to create metallib"
        echo "$link_output" | sed 's/^/    /'
        exit 1
    fi
}

# ============================================================================
# Validation Functions
# ============================================================================

validate_metallib() {
    log_info "Validating metallib..."

    if [[ ! -f "$OUTPUT_METALLIB" ]]; then
        log_error "Metallib file not found: $OUTPUT_METALLIB"
        exit 1
    fi

    # Get file size
    local size=$(du -h "$OUTPUT_METALLIB" | cut -f1)
    log_verbose "Metallib size: $size"

    # List functions in metallib
    log_verbose "Listing functions in metallib..."
    local functions_output
    if ! functions_output=$(xcrun metal-nm "$OUTPUT_METALLIB" 2>&1); then
        log_error "Failed to inspect metallib"
        echo "$functions_output" | sed 's/^/    /'
        exit 1
    fi

    # Extract function names
    local found_kernels=()
    while IFS= read -r line; do
        if [[ $line =~ T[[:space:]](.+)$ ]]; then
            found_kernels+=("${BASH_REMATCH[1]}")
        fi
    done <<< "$functions_output"

    # Verify all expected kernels are present
    local missing_kernels=()
    for expected in "${EXPECTED_KERNELS[@]}"; do
        local found=false
        for actual in "${found_kernels[@]}"; do
            if [[ "$actual" == "$expected" ]]; then
                found=true
                break
            fi
        done
        if [[ "$found" == "false" ]]; then
            missing_kernels+=("$expected")
        fi
    done

    # Report results
    if [[ ${#missing_kernels[@]} -gt 0 ]]; then
        log_error "Missing expected kernels:"
        for kernel in "${missing_kernels[@]}"; do
            echo "  ✗ $kernel"
        done
        exit 1
    fi

    log_success "All expected kernels present:"
    for kernel in "${EXPECTED_KERNELS[@]}"; do
        echo "  ✓ $kernel"
    done

    log_success "Metallib validated successfully"
}

# ============================================================================
# Cleanup Functions
# ============================================================================

cleanup() {
    if [[ "$CLEAN" == "true" ]]; then
        log_info "Cleaning intermediate files..."
        rm -rf "$TEMP_DIR"
        log_success "Cleanup complete"
    else
        log_verbose "Intermediate files kept in: $TEMP_DIR"
    fi
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo ""
    echo -e "${BOLD}==============================================================================${NC}"
    echo -e "${BOLD}  EmbedKit Metal Shader Compilation${NC}"
    echo -e "${BOLD}==============================================================================${NC}"
    echo ""

    # Pre-flight checks
    check_metal_compiler
    check_directories
    find_metal_files

    echo ""
    echo -e "${BOLD}--- Compilation Phase ---${NC}"
    echo ""

    # Compile shaders
    compile_all_shaders

    echo ""
    echo -e "${BOLD}--- Linking Phase ---${NC}"
    echo ""

    # Link metallib
    link_metallib

    echo ""
    echo -e "${BOLD}--- Validation Phase ---${NC}"
    echo ""

    # Validate output
    validate_metallib

    echo ""
    echo -e "${BOLD}--- Cleanup Phase ---${NC}"
    echo ""

    # Cleanup
    cleanup

    echo ""
    echo -e "${BOLD}==============================================================================${NC}"
    echo -e "${GREEN}${BOLD}  ✓ Metal Shader Compilation Complete${NC}"
    echo -e "${BOLD}==============================================================================${NC}"
    echo ""
    echo -e "  Output: ${BOLD}$OUTPUT_METALLIB${NC}"
    echo -e "  Size:   ${BOLD}$(du -h "$OUTPUT_METALLIB" | cut -f1)${NC}"
    echo -e "  Kernels: ${BOLD}${#EXPECTED_KERNELS[@]}${NC}"
    echo ""
}

# ============================================================================
# Entry Point
# ============================================================================

# Trap errors and provide helpful message
trap 'log_error "Compilation failed at line $LINENO. Use --verbose for more details."' ERR

# Run main function
main

exit 0
