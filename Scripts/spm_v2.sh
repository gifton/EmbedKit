#!/usr/bin/env bash

# SwiftPM helper for EmbedKitV2 that keeps all caches inside the workspace
# and avoids macOS sandbox issues in restricted environments.

set -euo pipefail

here() { cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P; }
ROOT="$(cd "$(here)/.." && pwd -P)"

setup_env() {
  mkdir -p "$ROOT/.build/swift-module-cache" \
           "$ROOT/.build/clang-module-cache" \
           "$ROOT/.build/tmp" \
           "$ROOT/.build/xdg-cache"

  export CLANG_MODULE_CACHE_PATH="$ROOT/.build/clang-module-cache"
  export SWIFT_MODULECACHE_PATH="$ROOT/.build/swift-module-cache"
  export XDG_CACHE_HOME="$ROOT/.build/xdg-cache"
  export TMPDIR="$ROOT/.build/tmp"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  build               Build the EmbedKitV2 library only
  build-tests         Build the EmbedKitV2Tests target only
  test [--filter X]   Build tests then run them with an optional filter
  env                 Print the effective environment/cache paths
  clean-caches        Remove local module/tmp caches used by this script

Examples:
  $(basename "$0") build
  $(basename "$0") build-tests
  $(basename "$0") test --filter Week1IntegrationTests
EOF
}

build_v2() {
  setup_env
  swift build --disable-sandbox -v \
    --target EmbedKitV2 \
    -Xswiftc -module-cache-path -Xswiftc "$SWIFT_MODULECACHE_PATH" \
    -Xcc -fmodules-cache-path="$CLANG_MODULE_CACHE_PATH"
}

build_v2_tests() {
  setup_env
  swift build --disable-sandbox -v \
    --target EmbedKitV2Tests \
    -Xswiftc -module-cache-path -Xswiftc "$SWIFT_MODULECACHE_PATH" \
    -Xcc -fmodules-cache-path="$CLANG_MODULE_CACHE_PATH"
}

run_tests() {
  local filter=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --filter)
        filter="${2:-}"; shift 2 ;;
      *)
        echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
  done

  build_v2_tests

  setup_env
  if [[ -z "$filter" ]]; then
    # Default to running the core Week 1 integration tests in the v2 test target
    filter="EmbedKitV2Tests.Week1IntegrationTests"
  else
    # If the filter doesn't include a module/class prefix, default to the v2 test target
    if [[ "$filter" != *.* ]]; then
      filter="EmbedKitV2Tests.$filter"
    fi
  fi

  swift test --disable-sandbox --skip-build -v --filter "$filter"
}

print_env() {
  setup_env
  echo "ROOT=$ROOT"
  echo "CLANG_MODULE_CACHE_PATH=$CLANG_MODULE_CACHE_PATH"
  echo "SWIFT_MODULECACHE_PATH=$SWIFT_MODULECACHE_PATH"
  echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
  echo "TMPDIR=$TMPDIR"
}

clean_caches() {
  rm -rf "$ROOT/.build/swift-module-cache" \
         "$ROOT/.build/clang-module-cache" \
         "$ROOT/.build/tmp" \
         "$ROOT/.build/xdg-cache"
  echo "Cleaned local caches."
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  build)          build_v2 ;;
  build-tests)    build_v2_tests ;;
  test)           run_tests "$@" ;;
  env)            print_env ;;
  clean-caches)   clean_caches ;;
  help|--help|-h) usage ;;
  *)              echo "Unknown command: $cmd" >&2; usage; exit 2 ;;
esac
