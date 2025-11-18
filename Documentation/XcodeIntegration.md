# EmbedKit Xcode Integration Guide

**Version**: 1.0.0
**Last Updated**: 2025-10-24
**Applies To**: Xcode 15.0+, EmbedKit 0.1.0+

---

## Overview

This guide explains how to integrate EmbedKit's Metal shader compilation into Xcode-based projects. After following this guide, Metal shaders will compile automatically on every build, with full support for incremental builds.

### What You'll Get

✅ Automatic Metal shader compilation on build
✅ Incremental builds (only recompile changed shaders)
✅ Xcode-integrated error messages
✅ Multi-platform support (macOS, iOS, tvOS, etc.)
✅ Debug and Release build configurations

---

## Prerequisites

### Required

- **Xcode**: Version 15.0 or later
- **macOS**: 14.0 (Sonoma) or later
- **Command Line Tools**: Xcode Command Line Tools installed
- **EmbedKit**: Added as SPM package dependency

### Verify Command Line Tools

```bash
xcrun metal --version
# Should output: Apple metal version...
```

If not installed:
```bash
xcode-select --install
```

---

## Quick Start (5 Minutes)

### Step 1: Add Build Phase

1. Open your Xcode project
2. Select your **app target** (not the package)
3. Go to **Build Phases** tab
4. Click **+ → New Run Script Phase**
5. Rename it to: **"Compile Metal Shaders"**
6. **Drag it above** "Compile Sources" (important!)

### Step 2: Configure Script

In the script text area, enter:

```bash
${SRCROOT}/Packages/EmbedKit/Scripts/XcodeBuildPhase.sh
```

> **Note**: Adjust path if EmbedKit is in a different location

### Step 3: Add Input Files

Click **+** under "Input Files" and add:

```
$(SRCROOT)/Packages/EmbedKit/Sources/EmbedKit/Shaders/Kernels/Normalization.metal
$(SRCROOT)/Packages/EmbedKit/Sources/EmbedKit/Shaders/Kernels/Pooling.metal
$(SRCROOT)/Packages/EmbedKit/Sources/EmbedKit/Shaders/Kernels/Similarity.metal
$(SRCROOT)/Packages/EmbedKit/Sources/EmbedKit/Shaders/Common/MetalCommon.h
```

### Step 4: Add Output Files

Click **+** under "Output Files" and add:

```
$(BUILT_PRODUCTS_DIR)/EmbedKitShaders.metallib
```

### Step 5: Build

Press **⌘B** (Build). You should see:

```
note: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
note: EmbedKit Metal Shader Build Phase
note: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
note: Changes detected, compiling shaders...
note: Platform: macosx, Configuration: Debug
note: Found 3 shader file(s)
note: Compiling Normalization.metal...
note: ✓ Compiled Normalization.metal
...
note: ✓ Metal shader compilation complete
```

**Done!** 🎉

---

## Detailed Configuration

### Build Phase Settings

#### Shell

Set to: `/bin/bash` (default)

#### Script Body

```bash
# For SPM package in default location
${SRCROOT}/Packages/EmbedKit/Scripts/XcodeBuildPhase.sh

# For local package
${SRCROOT}/EmbedKit/Scripts/XcodeBuildPhase.sh

# For embedded framework
${SRCROOT}/Frameworks/EmbedKit/Scripts/XcodeBuildPhase.sh
```

#### Input Files

Input files tell Xcode to run the script when these files change:

```
$(SRCROOT)/Packages/EmbedKit/Sources/EmbedKit/Shaders/Kernels/*.metal
$(SRCROOT)/Packages/EmbedKit/Sources/EmbedKit/Shaders/Common/*.h
```

> **Tip**: Xcode doesn't expand wildcards in input files. List each file explicitly.

#### Output Files

Output files tell Xcode what this phase produces:

```
$(BUILT_PRODUCTS_DIR)/EmbedKitShaders.metallib
```

This enables incremental builds - Xcode skips the script if outputs are up-to-date.

#### Additional Options

- **☐ Based on dependency analysis**: Leave unchecked
- **☐ Run script only when installing**: Leave unchecked
- **☐ Show environment variables in build log**: Check for debugging

### Phase Ordering

The build phase **must run before** "Compile Sources" because Swift code needs the metallib at compile time for bundle resource lookup.

**Correct Order**:
1. Target Dependencies
2. **Compile Metal Shaders** ← Your script
3. Compile Sources
4. Link Binary
5. Copy Bundle Resources

---

## Multi-Platform Support

### iOS Apps

Same steps as macOS, but verify platform-specific settings:

1. Select **iOS** target
2. Follow Quick Start steps
3. Test on both **Simulator** and **Device**

#### iOS-Specific Notes

- Script auto-detects `PLATFORM_NAME` (iphoneos vs iphonesimulator)
- Metal 3 required (iOS 16.0+)
- Metallib is platform-specific (different for sim vs device)

### Universal Apps (Catalyst)

For Mac Catalyst apps:

1. Add build phase to iOS target
2. Script handles both `iphoneos` and `macosx` platforms
3. Separate metalllibs generated for each platform

### tvOS / watchOS

Same process, tested platforms:
- tvOS 17.0+
- watchOS 10.0+ (limited Metal support)

---

## Build Configurations

### Debug vs Release

Script automatically adjusts based on `CONFIGURATION`:

| Configuration | Metal Flags | Fast Math | Optimization |
|---------------|-------------|-----------|--------------|
| Debug | `-O0` | Disabled | None (faster builds) |
| Release | `-O3` | Enabled | Maximum performance |

No additional configuration needed - script detects automatically.

### Custom Configurations

For custom configs (e.g., "Staging"):

The script treats any non-"Release" config as Debug. To override:

```bash
# In build phase script, before calling XcodeBuildPhase.sh:
export CONFIGURATION="Release"
${SRCROOT}/Packages/EmbedKit/Scripts/XcodeBuildPhase.sh
```

---

## Incremental Builds

### How It Works

1. Script checks timestamps of .metal files vs metallib
2. If metallib is newer than all .metal files → skip
3. If any .metal file is newer → recompile all

### Force Recompilation

To force recompilation:

**Option 1**: Clean build folder
```
Product → Clean Build Folder (⇧⌘K)
```

**Option 2**: Touch a shader file
```bash
touch Sources/EmbedKit/Shaders/Kernels/Normalization.metal
```

**Option 3**: Delete metallib
```bash
rm -f ${BUILT_PRODUCTS_DIR}/EmbedKitShaders.metallib
```

---

## Verification

### Check Build Log

After building, check the build log for:

```
✓ All expected kernels present
✓ Metallib size: 28K
✓ Metal shader compilation complete
```

### Verify Metallib Exists

```bash
# macOS
ls ~/Library/Developer/Xcode/DerivedData/YourApp-*/Build/Products/Debug/EmbedKitShaders.metallib

# Should exist and be ~20-30KB
```

### Verify App Bundle

```bash
# macOS app bundle
ls YourApp.app/Contents/Resources/EmbedKitShaders.metallib

# iOS app bundle
ls YourApp.app/EmbedKitShaders.metallib
```

### Test Runtime Loading

Add to your app:

```swift
import EmbedKit

// Test Metal library loading
Task {
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Metal not available")
        return
    }

    do {
        let (library, source) = try await MetalLibraryLoader.loadLibrary(device: device)
        print("✓ Loaded Metal library: \(source.description)")

        // Test a specific kernel
        if let function = library.makeFunction(name: "l2_normalize") {
            print("✓ Found l2_normalize kernel")
        }
    } catch {
        print("✗ Failed to load Metal library: \(error)")
    }
}
```

Expected output:
```
✓ Loaded Metal library: Precompiled metallib: EmbedKitShaders.metallib
✓ Found l2_normalize kernel
```

---

## Common Issues & Solutions

### Issue: "xcrun: error: unable to find utility "metal""

**Cause**: Command Line Tools not installed
**Solution**:
```bash
xcode-select --install
```

### Issue: Build phase runs but no metallib created

**Cause**: Output directory doesn't exist
**Solution**: Script creates it automatically. Check build log for errors.

### Issue: "error: Shaders directory not found"

**Cause**: Incorrect path to EmbedKit in script
**Solution**: Verify `${SRCROOT}/Packages/EmbedKit` path is correct

Check actual path:
```bash
ls ${SRCROOT}/Packages/
# Should list EmbedKit
```

### Issue: Metallib not found at runtime

**Cause**: Metallib not copied to app bundle
**Solution**: Add "Copy Files" build phase:

1. Add "Copy Files" phase
2. Destination: Resources
3. Add `EmbedKitShaders.metallib` from build products

### Issue: "Multiple commands produce metallib"

**Cause**: Build phase configured on multiple targets
**Solution**: Only add to main app target, not framework targets

### Issue: Slow incremental builds

**Cause**: Timestamp checking not working
**Solution**: Verify input/output files are configured correctly

---

## Advanced Configuration

### Parallel Compilation

For projects with many shaders, enable parallel compilation:

Edit `XcodeBuildPhase.sh`, add before compilation loop:

```bash
# Enable parallel compilation (experimental)
export METAL_PARALLEL_JOBS=4
```

### Custom Metallib Location

To output metallib to custom location:

```bash
# In build phase, before script call:
export OUTPUT_METALLIB="${SRCROOT}/CustomPath/Shaders.metallib"
${SRCROOT}/Packages/EmbedKit/Scripts/XcodeBuildPhase.sh
```

### Verbose Logging

Enable detailed logging:

```bash
# In build phase:
set -x  # Enable bash tracing
${SRCROOT}/Packages/EmbedKit/Scripts/XcodeBuildPhase.sh
```

---

## Performance Optimization

### Build Time Impact

Typical impact on build times:

| Scenario | Impact |
|----------|--------|
| Clean build | +2-3 seconds |
| Incremental (no changes) | +0.1 seconds (timestamp check) |
| Incremental (shader changed) | +1-2 seconds |

### Optimization Tips

1. **Use Input/Output Files**: Enables Xcode's dependency tracking
2. **Enable Parallel Jobs**: For many shaders (see Advanced)
3. **Disable in Development**: Comment out script call for faster iteration
4. **Use Release Builds**: Optimized metallib is smaller and faster to link

---

## Troubleshooting

For detailed troubleshooting, see:
- [XcodeTroubleshooting.md](./XcodeTroubleshooting.md)
- GitHub Issues: https://github.com/your-org/embedkit/issues

### Enable Debug Output

Temporarily enable debug mode:

```bash
# In build phase, at the top:
export DEBUG_METAL_BUILD=1
${SRCROOT}/Packages/EmbedKit/Scripts/XcodeBuildPhase.sh
```

### Manual Script Test

Test the script manually:

```bash
cd /path/to/your/project
export SRCROOT=$(pwd)
export CONFIGURATION=Debug
export PLATFORM_NAME=macosx
export BUILT_PRODUCTS_DIR=$(pwd)/build
./Packages/EmbedKit/Scripts/XcodeBuildPhase.sh
```

---

## Migration from Manual Compilation

If you were running `CompileMetalShaders.sh` manually:

1. **Add build phase** (follow Quick Start)
2. **Remove manual step** from your workflow
3. **Remove metallib from git** (optional)
   ```bash
   git rm Sources/EmbedKit/Resources/EmbedKitShaders.metallib
   echo "EmbedKitShaders.metallib" >> .gitignore
   ```
4. **Clean build** to regenerate
5. **Verify** metallib loads correctly

---

## FAQ

### Q: Do I need to run CompileMetalShaders.sh anymore?

**A**: No! The Xcode build phase replaces it. However, `CompileMetalShaders.sh` still works for SPM-only projects.

### Q: Can I use both SPM and Xcode builds?

**A**: Yes! They use separate scripts and don't interfere.

### Q: What if I don't use Xcode?

**A**: SPM projects don't need this. Use `CompileMetalShaders.sh` directly.

### Q: Does this work with Objective-C projects?

**A**: Yes! The build phase is language-agnostic.

### Q: Can I customize the metallib name?

**A**: Yes, but you'll need to update `MetalLibraryLoader.swift` to match.

### Q: Does this work with Swift Playgrounds?

**A**: Playgrounds don't support build phases. Use precompiled metallib.

---

## Support

### Getting Help

1. Check [XcodeTroubleshooting.md](./XcodeTroubleshooting.md)
2. Search existing issues
3. Open new issue with:
   - Xcode version
   - macOS version
   - Build log excerpt
   - Steps to reproduce

### Reporting Bugs

Include:
- Complete build log
- Output of `xcrun metal --version`
- Project configuration (SPM/manual/etc.)

---

## Changelog

### v1.0.0 (2025-10-24)
- Initial Xcode integration support
- Incremental build support
- Multi-platform support
- Comprehensive error messages

---

## Related Documentation

- [CompileMetalShaders.sh](../Scripts/README.md) - Manual compilation for SPM
- [Metal Shader Development](../Sources/EmbedKit/Shaders/README.md) - Shader development guide
- [XcodeTroubleshooting.md](./XcodeTroubleshooting.md) - Detailed troubleshooting

---

**Questions?** Open an issue or discussion on GitHub.
