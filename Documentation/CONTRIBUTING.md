# Contributing to EmbedKit

We love your input! We want to make contributing to EmbedKit as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html)

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code follows the existing style.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](LICENSE) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/yourusername/EmbedKit/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/EmbedKit/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Process

### Prerequisites

- Xcode 16.0+
- Swift 6.1+
- macOS 13.0+ for development

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/EmbedKit.git
   cd EmbedKit
   ```

2. Open the package in Xcode:
   ```bash
   open Package.swift
   ```

3. Build the package:
   ```bash
   swift build
   ```

### Running Tests

```bash
swift test
```

### Code Style

- Follow Swift API Design Guidelines
- Use meaningful variable and function names
- Keep functions focused and small
- Document public APIs with clear comments
- Use Swift's type system to prevent errors

### Testing Guidelines

- Write tests for all new functionality
- Use Swift Testing framework (`@Test` attribute)
- Aim for high code coverage
- Test edge cases and error conditions
- Performance tests for critical paths

### Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable
2. Update the CHANGELOG.md with your changes
3. Increase version numbers if this is a release (following [SemVer](http://semver.org/))
4. The PR will be merged once you have the sign-off of at least one maintainer

## Performance Considerations

When contributing performance-critical code:

- Benchmark before and after changes
- Consider memory usage and allocations
- Use Instruments to profile your code
- Document performance characteristics
- Consider Metal acceleration where appropriate

## Documentation

- Add inline documentation for all public APIs
- Update README for new features
- Include code examples where helpful
- Keep documentation up-to-date with code changes

## Community

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Questions?

Feel free to open an issue with your question or reach out to the maintainers directly.

Thank you for contributing to EmbedKit! 🎉