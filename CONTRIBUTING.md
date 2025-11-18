# Contributing to EmbedKit

Thank you for your interest in contributing to EmbedKit! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

### Our Pledge
We are committed to providing a friendly, safe, and welcoming environment for all contributors.

### Expected Behavior
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully

## Getting Started

### Prerequisites
- macOS 14.0+ or iOS 17.0+ development environment
- Xcode 15.0+
- Swift 6.0+
- Git
- Python 3.8+ (for model conversion)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
```bash
git clone https://github.com/yourusername/VSK.git
cd VSK/EmbedKit
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/originalrepo/VSK.git
```

## Development Setup

### 1. Open in Xcode
```bash
open Package.swift
```

### 2. Build the Package
```bash
swift build
```

### 3. Run Tests
```bash
swift test
```

### 4. Set Up Model Conversion (Optional)
```bash
pip install torch transformers coremltools sentence-transformers numpy
python Scripts/ModelTools/convert_minilm_l12.py
```

## How to Contribute

### Types of Contributions

#### 🐛 Bug Reports
- Use the issue tracker
- Include reproduction steps
- Provide system information
- Include relevant logs/errors

#### 💡 Feature Requests
- Discuss in issues first
- Explain use case clearly
- Consider implementation impact

#### 📝 Documentation
- Fix typos and clarify text
- Add examples
- Improve API documentation
- Translate documentation

#### 🔧 Code Contributions
- Bug fixes
- Performance improvements
- New features
- Test coverage

### Finding Issues to Work On

Look for issues labeled:
- `good first issue` - Great for newcomers
- `help wanted` - We need help with these
- `enhancement` - New features
- `bug` - Something needs fixing

## Coding Standards

### Swift Style Guide

#### Naming
```swift
// Types: UpperCamelCase
struct EmbeddingPipeline { }
protocol ModelBackend { }

// Functions/Properties: lowerCamelCase
func generateEmbeddings() { }
var isLoaded: Bool { }

// Constants: lowerCamelCase
let maxBatchSize = 32

// Generics: Single letter or descriptive
struct Embedding<D: Dimension> { }
```

#### Code Organization
```swift
// MARK: - Section Headers
// MARK: - Properties
// MARK: - Initialization
// MARK: - Public Methods
// MARK: - Private Methods
```

#### Documentation
```swift
/// Brief description of the function.
///
/// More detailed explanation if needed.
///
/// - Parameters:
///   - input: Description of input
///   - options: Description of options
/// - Returns: What the function returns
/// - Throws: When the function throws
/// - Complexity: O(n) where n is...
public func process(
    input: String,
    options: Options = .default
) async throws -> Result {
    // Implementation
}
```

### Performance Guidelines

#### Use `@inlinable` for Hot Paths
```swift
@inlinable
public func cosineSimilarity(to other: Self) -> Float {
    // Performance-critical code
}
```

#### Prefer Value Types
```swift
// Good: Struct with copy-on-write
struct Embedding {
    private var storage: Storage
}

// Avoid: Class unless needed for identity
class Model {  // Only for stateful components
}
```

#### Actor-Based Concurrency
```swift
// All stateful components should be actors
public actor TokenizerCache {
    private var cache: [String: TokenizedInput] = [:]
}
```

### Error Handling

#### Define Specific Errors
```swift
enum TokenizationError: LocalizedError {
    case textTooLong(Int, max: Int)
    case invalidCharacter(Character)

    var errorDescription: String? {
        switch self {
        case .textTooLong(let length, let max):
            return "Text too long: \(length) > \(max)"
        case .invalidCharacter(let char):
            return "Invalid character: \(char)"
        }
    }
}
```

## Testing Guidelines

### Unit Tests

#### Test File Organization
```
Tests/
  EmbedKitTests/
    Core/
      EmbeddingTests.swift
      PipelineTests.swift
    Tokenization/
      BERTTokenizerTests.swift
    Acceleration/
      MetalTests.swift
```

#### Test Naming
```swift
func testEmbeddingInitialization() throws { }
func testCosineSimilarityWithNormalizedVectors() throws { }
func testTokenizationWithEmptyString() throws { }
```

#### Test Structure
```swift
func testFeature() async throws {
    // Arrange
    let input = "test input"
    let expected = ExpectedResult()

    // Act
    let result = try await sut.process(input)

    // Assert
    XCTAssertEqual(result, expected)
}
```

### Performance Tests
```swift
func testEmbeddingPerformance() throws {
    measure {
        // Performance-critical code
        _ = try! Embedding384.random()
    }
}
```

### Integration Tests
```swift
func testEndToEndPipeline() async throws {
    // Test complete flow from text to embedding
}
```

## Documentation

### Code Comments
- Explain "why" not "what"
- Document complex algorithms
- Include mathematical formulas

```swift
// Applying log-sum-exp trick for numerical stability
// softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
let maxVal = values.max() ?? 0
let expValues = values.map { exp($0 - maxVal) }
```

### API Documentation
- All public APIs must be documented
- Include usage examples
- Document edge cases

### README Updates
- Update README for new features
- Keep examples current
- Update performance metrics

## Pull Request Process

### Before Creating a PR

1. **Update from upstream**
```bash
git fetch upstream
git rebase upstream/main
```

2. **Create feature branch**
```bash
git checkout -b feature/your-feature-name
```

3. **Make your changes**
- Write code
- Add tests
- Update documentation

4. **Run tests locally**
```bash
swift test
```

5. **Check for warnings**
```bash
swift build -Xswiftc -warnings-as-errors
```

### PR Guidelines

#### Title Format
```
[Component] Brief description

Examples:
[Tokenizer] Add support for SentencePiece
[Metal] Optimize pooling kernel performance
[Docs] Update installation instructions
```

#### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance impact measured

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No warnings
```

### Review Process

1. **Automated Checks**
   - CI/CD runs tests
   - Code coverage analysis
   - Swift format check

2. **Code Review**
   - At least one approval required
   - Address all feedback
   - Keep discussions constructive

3. **Merge**
   - Squash and merge for clean history
   - Delete feature branch

## Performance Considerations

When contributing performance improvements:

1. **Measure Before and After**
```swift
// Benchmark before
let before = CFAbsoluteTimeGetCurrent()
// ... code ...
let after = CFAbsoluteTimeGetCurrent()
print("Time: \(after - before)s")
```

2. **Profile with Instruments**
- Time Profiler
- Allocations
- GPU Frame Capture

3. **Document Performance Impact**
- Include benchmarks in PR
- Note any trade-offs
- Consider different devices

## Metal Shader Contributions

For GPU kernel contributions:

1. **Follow Metal Best Practices**
- Use fast math when appropriate
- Optimize memory access patterns
- Consider threadgroup memory

2. **Test on Multiple Devices**
- Different GPU families
- Various memory configurations
- Fallback paths

3. **Document Kernel Behavior**
```metal
/// Performs L2 normalization on input vector
/// Uses SIMD group operations for efficiency
/// Handles non-uniform threadgroups
kernel void l2_normalize(...) { }
```

## Questions?

- Create an issue for discussion
- Join our Discord community
- Email maintainers

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to EmbedKit! 🚀
