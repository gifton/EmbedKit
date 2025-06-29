import Foundation

/// Protocol for command metadata
/// 
/// Swift 6 compliant - all conforming types must be Sendable
public protocol CommandMetadata: Sendable {
    associatedtype Value: Sendable
    var value: Value { get }
}

/// Immutable command metadata implementation
public struct ImmutableMetadata<T: Sendable>: CommandMetadata {
    public let value: T
    
    public init(value: T) {
        self.value = value
    }
}

/// Mutable command metadata implementation using actor isolation
/// 
/// Swift 6 compliant replacement for class with @unchecked Sendable
public actor MutableMetadata<T: Sendable>: CommandMetadata {
    private var _value: T
    
    public var value: T {
        get async { _value }
    }
    
    public init(value: T) {
        self._value = value
    }
    
    /// Update the metadata value
    public func update(_ newValue: T) {
        self._value = newValue
    }
    
    /// Update the value using a transform function
    public func update<U>(_ transform: (T) throws -> U) async rethrows where T == U {
        self._value = try transform(_value)
    }
}

/// Thread-safe reference type metadata wrapper
/// 
/// Uses actor isolation for thread safety without @unchecked Sendable
public actor ReferenceMetadata<T: AnyObject & Sendable>: CommandMetadata {
    private var _value: T
    
    public var value: T {
        get async { _value }
    }
    
    public init(value: T) {
        self._value = value
    }
    
    /// Replace the reference
    public func replace(with newValue: T) {
        self._value = newValue
    }
}

/// Atomic metadata for simple value types
/// 
/// Provides lock-free atomic operations for supported types
@available(macOS 13.0, iOS 16.0, watchOS 9.0, tvOS 16.0, *)
public final class AtomicMetadata<T: AtomicRepresentable>: CommandMetadata, Sendable {
    private let atomic: Atomic<T>
    
    public var value: T {
        atomic.load(ordering: .relaxed)
    }
    
    public init(value: T) {
        self.atomic = Atomic(value)
    }
    
    /// Atomically update the value
    public func store(_ newValue: T, ordering: AtomicStoreOrdering = .relaxed) {
        atomic.store(newValue, ordering: ordering)
    }
    
    /// Atomically exchange values
    @discardableResult
    public func exchange(_ newValue: T, ordering: AtomicUpdateOrdering = .relaxed) -> T {
        atomic.exchange(newValue, ordering: ordering)
    }
    
    /// Compare and exchange
    @discardableResult
    public func compareExchange(
        expected: T,
        desired: T,
        ordering: AtomicUpdateOrdering = .relaxed
    ) -> (exchanged: Bool, original: T) {
        atomic.compareExchange(
            expected: expected,
            desired: desired,
            ordering: ordering
        )
    }
}

/// Protocol for types that can be used with Atomic
public protocol AtomicRepresentable {
    // This would include Int, Bool, UInt, etc.
}

// Conform basic types to AtomicRepresentable
extension Int: AtomicRepresentable {}
extension UInt: AtomicRepresentable {}
extension Bool: AtomicRepresentable {}
extension Int32: AtomicRepresentable {}
extension Int64: AtomicRepresentable {}
extension UInt32: AtomicRepresentable {}
extension UInt64: AtomicRepresentable {}

/// Versioned metadata for tracking changes
public actor VersionedMetadata<T: Sendable & Equatable>: CommandMetadata {
    private var _value: T
    private var _version: UInt64 = 0
    private var history: [(version: UInt64, value: T, timestamp: Date)] = []
    private let maxHistorySize: Int
    
    public var value: T {
        get async { _value }
    }
    
    public var version: UInt64 {
        get async { _version }
    }
    
    public init(value: T, maxHistorySize: Int = 10) {
        self._value = value
        self.maxHistorySize = maxHistorySize
        self.history.append((version: 0, value: value, timestamp: Date()))
    }
    
    /// Update value and increment version
    public func update(_ newValue: T) {
        guard newValue != _value else { return }
        
        _version += 1
        _value = newValue
        
        history.append((version: _version, value: newValue, timestamp: Date()))
        
        // Trim history if needed
        if history.count > maxHistorySize {
            history.removeFirst(history.count - maxHistorySize)
        }
    }
    
    /// Get value at specific version
    public func value(at version: UInt64) -> T? {
        history.first { $0.version == version }?.value
    }
    
    /// Get full history
    public func getHistory() -> [(version: UInt64, value: T, timestamp: Date)] {
        history
    }
}

/// Cached metadata with expiration
public actor CachedMetadata<T: Sendable>: CommandMetadata {
    private var _value: T
    private var expiresAt: Date?
    private let ttl: TimeInterval?
    private let refreshHandler: (@Sendable () async throws -> T)?
    
    public var value: T {
        get async throws {
            // Check expiration
            if let expiresAt = expiresAt, Date() > expiresAt {
                // Refresh if handler provided
                if let handler = refreshHandler {
                    try await refresh(using: handler)
                }
            }
            return _value
        }
    }
    
    public init(
        value: T,
        ttl: TimeInterval? = nil,
        refreshHandler: (@Sendable () async throws -> T)? = nil
    ) {
        self._value = value
        self.ttl = ttl
        self.refreshHandler = refreshHandler
        
        if let ttl = ttl {
            self.expiresAt = Date().addingTimeInterval(ttl)
        }
    }
    
    /// Manually refresh the cached value
    public func refresh(using handler: @Sendable () async throws -> T) async throws {
        _value = try await handler()
        if let ttl = ttl {
            expiresAt = Date().addingTimeInterval(ttl)
        }
    }
    
    /// Update value and reset expiration
    public func update(_ newValue: T) {
        _value = newValue
        if let ttl = ttl {
            expiresAt = Date().addingTimeInterval(ttl)
        }
    }
    
    /// Check if cache is expired
    public var isExpired: Bool {
        get async {
            guard let expiresAt = expiresAt else { return false }
            return Date() > expiresAt
        }
    }
}

/// Observable metadata that notifies on changes
public actor ObservableMetadata<T: Sendable>: CommandMetadata {
    public typealias Observer = @Sendable (T, T) async -> Void
    
    private var _value: T
    private var observers: [UUID: Observer] = [:]
    
    public var value: T {
        get async { _value }
    }
    
    public init(value: T) {
        self._value = value
    }
    
    /// Update value and notify observers
    public func update(_ newValue: T) async {
        let oldValue = _value
        _value = newValue
        
        // Notify all observers
        await withTaskGroup(of: Void.self) { group in
            for observer in observers.values {
                group.addTask {
                    await observer(oldValue, newValue)
                }
            }
        }
    }
    
    /// Add an observer
    @discardableResult
    public func addObserver(_ observer: @escaping Observer) -> UUID {
        let id = UUID()
        observers[id] = observer
        return id
    }
    
    /// Remove an observer
    public func removeObserver(_ id: UUID) {
        observers.removeValue(forKey: id)
    }
}