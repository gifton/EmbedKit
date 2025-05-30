import Foundation
import Security
import CryptoKit
import OSLog

/// Model signature verification system for ensuring model integrity and authenticity
public actor ModelSignatureVerifier {
    private let logger = Logger(subsystem: "EmbedKit", category: "ModelSignatureVerifier")
    
    private let publicKeys: [String: SecKey] = [:]
    private let trustedSigners: Set<String>
    private let useKeychain: Bool
    
    public init(trustedSigners: Set<String> = Set(), useKeychain: Bool = true) {
        self.trustedSigners = trustedSigners
        self.useKeychain = useKeychain
    }
    
    /// Generate a digital signature for a model file
    public func signModel(fileURL: URL, signerIdentity: String, privateKey: SecKey) async throws -> ModelSignature {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw SignatureError.fileNotFound(fileURL.path)
        }
        
        logger.info("Signing model at \(fileURL.path) with identity \(signerIdentity)")
        
        // Calculate file hash
        let fileData = try Data(contentsOf: fileURL)
        let fileHash = SHA256.hash(data: fileData)
        let hashData = Data(fileHash)
        
        // Create signature data
        let modelInfo = ModelSignatureInfo(
            fileName: fileURL.lastPathComponent,
            fileSize: Int64(fileData.count),
            hashAlgorithm: "SHA256",
            signatureAlgorithm: "RSA-PKCS1",
            signerIdentity: signerIdentity,
            signedAt: Date(),
            version: "1.0"
        )
        
        let signatureData = try JSONEncoder().encode(modelInfo)
        let combinedData = hashData + signatureData
        
        // Sign the combined data
        var error: Unmanaged<CFError>?
        guard let signature = SecKeyCreateSignature(
            privateKey,
            .rsaSignatureMessagePKCS1v15SHA256,
            combinedData as CFData,
            &error
        ) else {
            if let error = error?.takeRetainedValue() {
                throw SignatureError.signingFailed(CFErrorCopyDescription(error) as String? ?? "Unknown error")
            }
            throw SignatureError.signingFailed("Failed to create signature")
        }
        
        let signatureString = (signature as Data).base64EncodedString()
        
        let modelSignature = ModelSignature(
            signature: signatureString,
            info: modelInfo,
            publicKeyFingerprint: try getKeyFingerprint(privateKey: privateKey)
        )
        
        await telemetry.recordEvent(TelemetryEvent(
            name: "model_signed",
            description: "Model signed with identity \(signerIdentity)",
            severity: .info,
            metadata: [
                "signer": signerIdentity,
                "file_size": String(fileData.count),
                "algorithm": "RSA-PSS"
            ]
        ))
        
        return modelSignature
    }
    
    /// Verify a model signature
    public func verifySignature(
        for fileURL: URL,
        signature: ModelSignature,
        publicKey: SecKey? = nil
    ) async throws -> SignatureVerificationResult {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw SignatureError.fileNotFound(fileURL.path)
        }
        
        logger.info("Verifying signature for model at \(fileURL.path)")
        
        // Load current file data
        let currentData = try Data(contentsOf: fileURL)
        let currentHash = SHA256.hash(data: currentData)
        let currentHashData = Data(currentHash)
        
        // Verify file hasn't changed
        guard Int64(currentData.count) == signature.info.fileSize else {
            return SignatureVerificationResult(
                isValid: false,
                trustLevel: .untrusted,
                issues: [.fileSizeChanged],
                verifiedAt: Date(),
                signerIdentity: signature.info.signerIdentity
            )
        }
        
        // Get public key for verification
        let verificationKey: SecKey
        if let providedKey = publicKey {
            verificationKey = providedKey
        } else if let storedKey = try await getStoredPublicKey(for: signature.info.signerIdentity) {
            verificationKey = storedKey
        } else {
            return SignatureVerificationResult(
                isValid: false,
                trustLevel: .untrusted,
                issues: [.publicKeyNotFound],
                verifiedAt: Date(),
                signerIdentity: signature.info.signerIdentity
            )
        }
        
        // Reconstruct signed data
        let signatureData = try JSONEncoder().encode(signature.info)
        let combinedData = currentHashData + signatureData
        
        // Verify signature
        guard let signatureBytes = Data(base64Encoded: signature.signature) else {
            return SignatureVerificationResult(
                isValid: false,
                trustLevel: .untrusted,
                issues: [.invalidSignatureFormat],
                verifiedAt: Date(),
                signerIdentity: signature.info.signerIdentity
            )
        }
        
        var error: Unmanaged<CFError>?
        let isValidSignature = SecKeyVerifySignature(
            verificationKey,
            .rsaSignatureMessagePKCS1v15SHA256,
            combinedData as CFData,
            signatureBytes as CFData,
            &error
        )
        
        var issues: [SignatureVerificationIssue] = []
        
        if !isValidSignature {
            issues.append(.signatureVerificationFailed)
        }
        
        // Check if signer is trusted
        let trustLevel: TrustLevel
        if trustedSigners.contains(signature.info.signerIdentity) {
            trustLevel = .trusted
        } else if isValidSignature {
            trustLevel = .validButUntrusted
        } else {
            trustLevel = .untrusted
        }
        
        // Check signature age
        let ageInDays = Date().timeIntervalSince(signature.info.signedAt) / (24 * 60 * 60)
        if ageInDays > 365 { // Signatures older than 1 year are considered stale
            issues.append(.signatureTooOld)
        }
        
        // Verify key fingerprint if available
        if !signature.publicKeyFingerprint.isEmpty {
            let currentFingerprint = try getKeyFingerprint(publicKey: verificationKey)
            if currentFingerprint != signature.publicKeyFingerprint {
                issues.append(.keyFingerprintMismatch)
            }
        }
        
        let result = SignatureVerificationResult(
            isValid: isValidSignature && issues.isEmpty,
            trustLevel: trustLevel,
            issues: issues,
            verifiedAt: Date(),
            signerIdentity: signature.info.signerIdentity
        )
        
        await telemetry.recordEvent(TelemetryEvent(
            name: "signature_verified",
            description: "Model signature verification completed",
            severity: result.isValid ? .info : .warning,
            metadata: [
                "signer": signature.info.signerIdentity,
                "is_valid": String(result.isValid),
                "trust_level": result.trustLevel.rawValue,
                "issues_count": String(issues.count)
            ]
        ))
        
        return result
    }
    
    /// Store a trusted public key
    public func storeTrustedPublicKey(_ publicKey: SecKey, for identity: String) async throws {
        guard useKeychain else {
            throw SignatureError.keychainUnavailable
        }
        
        // Convert key to data for storage
        var error: Unmanaged<CFError>?
        guard let keyData = SecKeyCopyExternalRepresentation(publicKey, &error) as Data? else {
            throw SignatureError.keyConversionFailed
        }
        
        // Create keychain item
        let query: [String: Any] = [
            kSecClass as String: kSecClassKey,
            kSecAttrApplicationLabel as String: "EmbedKit-\(identity)",
            kSecAttrKeyType as String: kSecAttrKeyTypeRSA,
            kSecAttrKeyClass as String: kSecAttrKeyClassPublic,
            kSecValueData as String: keyData,
            kSecAttrLabel as String: "EmbedKit Model Signer: \(identity)"
        ]
        
        // Delete existing item if it exists
        SecItemDelete(query as CFDictionary)
        
        // Add new item
        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw SignatureError.keychainStorageFailed(status)
        }
        
        logger.info("Stored trusted public key for identity: \(identity)")
    }
    
    /// Get a stored public key
    private func getStoredPublicKey(for identity: String) async throws -> SecKey? {
        guard useKeychain else {
            return nil
        }
        
        let query: [String: Any] = [
            kSecClass as String: kSecClassKey,
            kSecAttrApplicationLabel as String: "EmbedKit-\(identity)",
            kSecAttrKeyType as String: kSecAttrKeyTypeRSA,
            kSecAttrKeyClass as String: kSecAttrKeyClassPublic,
            kSecReturnRef as String: true
        ]
        
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        
        guard status == errSecSuccess else {
            if status != errSecItemNotFound {
                logger.warning("Failed to retrieve public key for \(identity): \(status)")
            }
            return nil
        }
        
        return (item as! SecKey)
    }
    
    /// Generate key fingerprint for identification
    private func getKeyFingerprint(privateKey: SecKey) throws -> String {
        guard let publicKey = SecKeyCopyPublicKey(privateKey) else {
            throw SignatureError.keyConversionFailed
        }
        
        return try getKeyFingerprint(publicKey: publicKey)
    }
    
    private func getKeyFingerprint(publicKey: SecKey) throws -> String {
        var error: Unmanaged<CFError>?
        guard let keyData = SecKeyCopyExternalRepresentation(publicKey, &error) as Data? else {
            throw SignatureError.keyConversionFailed
        }
        
        let hash = SHA256.hash(data: keyData)
        return Data(hash).base64EncodedString()
    }
    
    /// Generate a new RSA key pair for signing
    public static func generateKeyPair(keySize: Int = 2048) throws -> (privateKey: SecKey, publicKey: SecKey) {
        let attributes: [String: Any] = [
            kSecAttrKeyType as String: kSecAttrKeyTypeRSA,
            kSecAttrKeySizeInBits as String: keySize,
            kSecPrivateKeyAttrs as String: [
                kSecAttrIsPermanent as String: false
            ]
        ]
        
        var error: Unmanaged<CFError>?
        guard let privateKey = SecKeyCreateRandomKey(attributes as CFDictionary, &error) else {
            if let error = error?.takeRetainedValue() {
                throw SignatureError.keyGenerationFailed(CFErrorCopyDescription(error) as String? ?? "Unknown error")
            }
            throw SignatureError.keyGenerationFailed("Failed to generate key pair")
        }
        
        guard let publicKey = SecKeyCopyPublicKey(privateKey) else {
            throw SignatureError.keyGenerationFailed("Failed to extract public key")
        }
        
        return (privateKey, publicKey)
    }
    
    /// Validate signature format without file verification
    public func validateSignatureFormat(_ signature: ModelSignature) async -> Bool {
        // Check if signature can be base64 decoded
        guard Data(base64Encoded: signature.signature) != nil else {
            return false
        }
        
        // Verify required fields
        guard !signature.info.signerIdentity.isEmpty,
              !signature.info.fileName.isEmpty,
              signature.info.fileSize > 0 else {
            return false
        }
        
        // Check if signature is not too old or in the future
        let now = Date()
        let daysSinceSigning = now.timeIntervalSince(signature.info.signedAt) / (24 * 60 * 60)
        
        return daysSinceSigning >= 0 && daysSinceSigning <= 365 * 5 // Max 5 years old
    }
}

// MARK: - Supporting Types

public struct ModelSignature: Codable, Sendable {
    public let signature: String
    public let info: ModelSignatureInfo
    public let publicKeyFingerprint: String
    
    public init(signature: String, info: ModelSignatureInfo, publicKeyFingerprint: String) {
        self.signature = signature
        self.info = info
        self.publicKeyFingerprint = publicKeyFingerprint
    }
}

public struct ModelSignatureInfo: Codable, Sendable {
    public let fileName: String
    public let fileSize: Int64
    public let hashAlgorithm: String
    public let signatureAlgorithm: String
    public let signerIdentity: String
    public let signedAt: Date
    public let version: String
    
    public init(fileName: String, fileSize: Int64, hashAlgorithm: String, signatureAlgorithm: String, signerIdentity: String, signedAt: Date, version: String) {
        self.fileName = fileName
        self.fileSize = fileSize
        self.hashAlgorithm = hashAlgorithm
        self.signatureAlgorithm = signatureAlgorithm
        self.signerIdentity = signerIdentity
        self.signedAt = signedAt
        self.version = version
    }
}

public struct SignatureVerificationResult: Sendable {
    public let isValid: Bool
    public let trustLevel: TrustLevel
    public let issues: [SignatureVerificationIssue]
    public let verifiedAt: Date
    public let signerIdentity: String
    
    public init(isValid: Bool, trustLevel: TrustLevel, issues: [SignatureVerificationIssue], verifiedAt: Date, signerIdentity: String) {
        self.isValid = isValid
        self.trustLevel = trustLevel
        self.issues = issues
        self.verifiedAt = verifiedAt
        self.signerIdentity = signerIdentity
    }
}

public enum TrustLevel: String, Sendable, CaseIterable {
    case trusted = "trusted"
    case validButUntrusted = "valid_but_untrusted"
    case untrusted = "untrusted"
}

public enum SignatureVerificationIssue: Sendable {
    case signatureVerificationFailed
    case publicKeyNotFound
    case invalidSignatureFormat
    case fileSizeChanged
    case signatureTooOld
    case keyFingerprintMismatch
}

public enum SignatureError: LocalizedError {
    case fileNotFound(String)
    case signingFailed(String)
    case keyGenerationFailed(String)
    case keyConversionFailed
    case keychainUnavailable
    case keychainStorageFailed(OSStatus)
    case invalidSignature
    case untrustedSigner(String)
    
    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .signingFailed(let message):
            return "Signing failed: \(message)"
        case .keyGenerationFailed(let message):
            return "Key generation failed: \(message)"
        case .keyConversionFailed:
            return "Failed to convert key format"
        case .keychainUnavailable:
            return "Keychain is not available"
        case .keychainStorageFailed(let status):
            return "Keychain storage failed with status: \(status)"
        case .invalidSignature:
            return "Invalid signature"
        case .untrustedSigner(let identity):
            return "Untrusted signer: \(identity)"
        }
    }
}

/// Utility for creating signed model packages
public struct SignedModelPackage {
    public let modelURL: URL
    public let signature: ModelSignature
    public let metadata: [String: String]
    
    public init(modelURL: URL, signature: ModelSignature, metadata: [String: String] = [:]) {
        self.modelURL = modelURL
        self.signature = signature
        self.metadata = metadata
    }
    
    /// Save the signed package to a directory
    public func save(to directory: URL) throws {
        let packageName = modelURL.deletingPathExtension().lastPathComponent
        let packageDirectory = directory.appendingPathComponent("\(packageName).embedkit")
        
        // Create package directory
        try FileManager.default.createDirectory(at: packageDirectory, withIntermediateDirectories: true)
        
        // Copy model file
        let modelDestination = packageDirectory.appendingPathComponent(modelURL.lastPathComponent)
        try FileManager.default.copyItem(at: modelURL, to: modelDestination)
        
        // Save signature
        let signatureData = try JSONEncoder().encode(signature)
        let signatureURL = packageDirectory.appendingPathComponent("signature.json")
        try signatureData.write(to: signatureURL)
        
        // Save metadata
        let packageMetadata = PackageMetadata(
            version: "1.0",
            modelFile: modelURL.lastPathComponent,
            signature: "signature.json",
            metadata: metadata
        )
        
        let metadataData = try JSONEncoder().encode(packageMetadata)
        let metadataURL = packageDirectory.appendingPathComponent("package.json")
        try metadataData.write(to: metadataURL)
    }
    
    /// Load a signed package from a directory
    public static func load(from packageDirectory: URL) throws -> SignedModelPackage {
        let metadataURL = packageDirectory.appendingPathComponent("package.json")
        let metadataData = try Data(contentsOf: metadataURL)
        let packageMetadata = try JSONDecoder().decode(PackageMetadata.self, from: metadataData)
        
        let signatureURL = packageDirectory.appendingPathComponent(packageMetadata.signature)
        let signatureData = try Data(contentsOf: signatureURL)
        let signature = try JSONDecoder().decode(ModelSignature.self, from: signatureData)
        
        let modelURL = packageDirectory.appendingPathComponent(packageMetadata.modelFile)
        
        return SignedModelPackage(
            modelURL: modelURL,
            signature: signature,
            metadata: packageMetadata.metadata
        )
    }
}

struct PackageMetadata: Codable {
    let version: String
    let modelFile: String
    let signature: String
    let metadata: [String: String]
}