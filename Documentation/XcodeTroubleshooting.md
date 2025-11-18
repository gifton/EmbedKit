# Xcode Troubleshooting

This guide helps diagnose issues integrating EmbedKit’s Metal shader build phase.

Common checks
- Verify Xcode Command Line Tools: `xcrun metal --version`
- Confirm shader paths in build phase input files
- Ensure the build phase runs before “Compile Sources”
- Clean build folder (Shift+Cmd+K) if outputs are stale

Frequent errors
- "xcrun: error: unable to find utility \"metal\"" — install Command Line Tools: `xcode-select --install`
- "Shaders directory not found" — verify `SRCROOT` and script path
- Missing expected kernels — check shader names match validation list

Diagnostics
- Enable build phase tracing by adding `set -x` at the top of the run script
- Print environment: `env | sort`
- Manually run the script with exported Xcode vars to reproduce locally

If problems persist, capture the full build log and your Xcode/macOS versions.
