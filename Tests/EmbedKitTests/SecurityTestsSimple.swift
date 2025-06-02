import Foundation
import Testing
import Security
import CryptoKit
@testable import EmbedKit

@Suite("Security Components Tests - Simple")
struct SecurityTestsSimple {
    
    // MARK: - Basic Tests without SecKey
    
    @Test("ModelSignature format validation")
    func testModelSignatureFormatValidation() async {
        let verifier = ModelSignatureVerifier()
        
        // Test valid signature format
        let validSignature = ModelSignature(
            signature: Data("valid".utf8).base64EncodedString(),
            info: ModelSignatureInfo(
                fileName: "test.mlmodel",
                fileSize: 100,
                hashAlgorithm: "SHA256",
                signatureAlgorithm: "RSA-PKCS1",
                signerIdentity: "test",
                signedAt: Date(),
                version: "1.0"
            ),
            publicKeyFingerprint: "test"
        )
        
        let isValid = await verifier.validateSignatureFormat(validSignature)
        #expect(isValid)
        
        // Test invalid signature format - empty filename
        let invalidSignature = ModelSignature(
            signature: Data("valid".utf8).base64EncodedString(),
            info: ModelSignatureInfo(
                fileName: "",
                fileSize: 100,
                hashAlgorithm: "SHA256",
                signatureAlgorithm: "RSA-PKCS1",
                signerIdentity: "test",
                signedAt: Date(),
                version: "1.0"
            ),
            publicKeyFingerprint: "test"
        )
        
        let isInvalid = await verifier.validateSignatureFormat(invalidSignature)
        #expect(!isInvalid)
    }
    
    @Test("SignedModelPackage structure")
    func testSignedModelPackageStructure() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("SecurityTest_\(UUID())")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        
        let modelFile = tempDir.appendingPathComponent("test.mlmodel")
        try "test model".data(using: .utf8)!.write(to: modelFile)
        
        let signature = ModelSignature(
            signature: "dummy-signature",
            info: ModelSignatureInfo(
                fileName: "test.mlmodel",
                fileSize: 10,
                hashAlgorithm: "SHA256",
                signatureAlgorithm: "RSA-PKCS1",
                signerIdentity: "test-signer",
                signedAt: Date(),
                version: "1.0"
            ),
            publicKeyFingerprint: "dummy-fingerprint"
        )
        
        let metadata = ["version": "1.0", "author": "Test"]
        let package = SignedModelPackage(
            modelURL: modelFile,
            signature: signature,
            metadata: metadata
        )
        
        // Save package
        let packageDir = tempDir.appendingPathComponent("packages")
        try package.save(to: packageDir)
        
        // Verify package structure
        let expectedDir = packageDir.appendingPathComponent("test.embedkit")
        #expect(FileManager.default.fileExists(atPath: expectedDir.path))
        
        // Load package
        let loaded = try SignedModelPackage.load(from: expectedDir)
        #expect(loaded.signature.info.signerIdentity == "test-signer")
        #expect(loaded.metadata["version"] == "1.0")
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Trust level enum")
    func testTrustLevelEnum() {
        let levels = TrustLevel.allCases
        #expect(levels.count == 3)
        #expect(levels.contains(.trusted))
        #expect(levels.contains(.validButUntrusted))
        #expect(levels.contains(.untrusted))
        
        for level in levels {
            #expect(!level.rawValue.isEmpty)
        }
    }
    
    @Test("SignatureError descriptions")
    func testSignatureErrorDescriptions() {
        let errors: [SignatureError] = [
            .fileNotFound("/path/to/file"),
            .signingFailed("Test failure"),
            .keyGenerationFailed("Key gen failed"),
            .keyConversionFailed,
            .keychainUnavailable,
            .keychainStorageFailed(-1),
            .invalidSignature,
            .untrustedSigner("untrusted")
        ]
        
        for error in errors {
            #expect(error.errorDescription != nil)
            #expect(!error.errorDescription!.isEmpty)
        }
    }
    
    @Test("ModelSignatureInfo creation")
    func testModelSignatureInfoCreation() {
        let info = ModelSignatureInfo(
            fileName: "model.mlmodel",
            fileSize: 1024,
            hashAlgorithm: "SHA256",
            signatureAlgorithm: "RSA-PKCS1",
            signerIdentity: "com.example.signer",
            signedAt: Date(),
            version: "1.0"
        )
        
        #expect(info.fileName == "model.mlmodel")
        #expect(info.fileSize == 1024)
        #expect(info.hashAlgorithm == "SHA256")
        #expect(info.signatureAlgorithm == "RSA-PKCS1")
        #expect(info.signerIdentity == "com.example.signer")
        #expect(info.version == "1.0")
    }
    
    @Test("SignatureVerificationResult creation")
    func testSignatureVerificationResultCreation() {
        let result = SignatureVerificationResult(
            isValid: true,
            trustLevel: .trusted,
            issues: [],
            verifiedAt: Date(),
            signerIdentity: "test-signer"
        )
        
        #expect(result.isValid)
        #expect(result.trustLevel == .trusted)
        #expect(result.issues.isEmpty)
        #expect(result.signerIdentity == "test-signer")
    }
    
    @Test("SignatureVerificationIssue cases")
    func testSignatureVerificationIssueCases() {
        let issues: [SignatureVerificationIssue] = [
            .signatureVerificationFailed,
            .publicKeyNotFound,
            .invalidSignatureFormat,
            .fileSizeChanged,
            .signatureTooOld,
            .keyFingerprintMismatch
        ]
        
        // Just verify all cases are valid
        #expect(issues.count == 6)
    }
}