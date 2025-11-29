// EmbedKit - Apple Embedding Model

import Foundation
import Logging

/// Apple embedding model orchestration: tokenization → CoreML inference → pooling → normalization.
/// - Important: Batch padding requires a PAD token when using `.paddingStrategy = .batch` with `embedBatch(…)`.
public actor AppleEmbeddingModel: EmbeddingModel {
    // MARK: - Identity
    public nonisolated let id: ModelID
    public nonisolated let dimensions: Int
    public nonisolated let device: ComputeDevice

    // MARK: - Dependencies
    private let backend: (any CoreMLProcessingBackend)?
    private let tokenizer: any Tokenizer
    private let configuration: EmbeddingConfiguration
    private let logger = Logger(label: "EmbedKit.AppleEmbeddingModel")
    private let profiler: Profiler?
    private var metalAccelerator: MetalAccelerator? = nil

    // MARK: - Metrics
    private var metricsData = MetricsData()
    private var stageAgg = StageAgg()
    private let tokenCache = TokenCache<String, TokenizedText>(capacity: 2048)

    // MARK: - Init
    public init(
        backend: (any CoreMLProcessingBackend)?,
        tokenizer: any Tokenizer,
        configuration: EmbeddingConfiguration = EmbeddingConfiguration(),
        id: ModelID? = nil,
        dimensions: Int = 384,
        device: ComputeDevice? = nil,
        profiler: Profiler? = nil
    ) {
        self.backend = backend
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.dimensions = dimensions
        self.device = device ?? configuration.inferenceDevice
        self.id = id ?? ModelID(provider: "apple", name: "text-embedding", version: "1.0.0", variant: "base")
        self.profiler = profiler
    }

    // MARK: - EmbeddingModel
    /// Compute an embedding for a single text with the configured pipeline.
    public func embed(_ text: String) async throws -> Embedding {
        guard let backend else { throw EmbedKitError.modelLoadFailed("Backend unavailable") }

        let tStart = CFAbsoluteTimeGetCurrent()
        // Tokenize with configured behavior
        let inputInfo = try await buildInput(for: text)
        let tTok1 = CFAbsoluteTimeGetCurrent()

        // Padding invariant: ids and mask must align
        guard inputInfo.tokenIDs.count == inputInfo.attentionMask.count else {
            throw EmbedKitError.invalidConfiguration("ids/mask length mismatch: ids=\(inputInfo.tokenIDs.count), mask=\(inputInfo.attentionMask.count)")
        }

        if await !backend.isLoaded {
            if let ml = backend as? CoreMLBackend {
                await ml.setAutoWorkloadHint(paddedLength: inputInfo.tokenIDs.count, batchSize: 1)
            }
            try await backend.load()
        }
        let out = try await backend.process(.init(tokenIDs: inputInfo.tokenIDs, attentionMask: inputInfo.attentionMask))
        let tInf1 = CFAbsoluteTimeGetCurrent()

        let (tokens, dim) = try inferTokensAndDim(from: out.shape, valuesCount: out.values.count)
        logger.debug("Output shape resolved", metadata: [
            "shape": .string("\(out.shape)"),
            "tokens": .string("\(tokens)"),
            "dim": .string("\(dim)")
        ])
        if dim != dimensions { throw EmbedKitError.dimensionMismatch(expected: dimensions, got: dim) }

        // Determine if GPU pooling should be used for this sequence
        let useGPUPooling = tokens * dim >= configuration.minElementsForGPU

        let pooled: [Float]
        switch configuration.poolingStrategy {
        case .mean:
            if useGPUPooling, let acc = await ensureMetal() {
                pooled = await acc.meanPool(
                    embeddings: out.values,
                    sequenceLength: tokens,
                    dimensions: dim,
                    mask: inputInfo.attentionMask
                )
            } else {
                pooled = PoolingHelpers.mean(sequence: out.values, tokens: tokens, dim: dim, mask: inputInfo.attentionMask)
            }
        case .max:
            if useGPUPooling, let acc = await ensureMetal() {
                pooled = await acc.maxPool(
                    embeddings: out.values,
                    sequenceLength: tokens,
                    dimensions: dim,
                    mask: inputInfo.attentionMask
                )
            } else {
                pooled = PoolingHelpers.max(sequence: out.values, tokens: tokens, dim: dim, mask: inputInfo.attentionMask)
            }
        case .cls:
            // CLS pooling is O(1) - just grab first token, no GPU benefit
            pooled = PoolingHelpers.cls(sequence: out.values, tokens: tokens, dim: dim)
        case .attention:
            // Attention pooling without weights falls back to mean
            if useGPUPooling, let acc = await ensureMetal() {
                pooled = await acc.meanPool(
                    embeddings: out.values,
                    sequenceLength: tokens,
                    dimensions: dim,
                    mask: inputInfo.attentionMask
                )
            } else {
                pooled = PoolingHelpers.mean(sequence: out.values, tokens: tokens, dim: dim, mask: inputInfo.attentionMask)
            }
        }
        let tPool0 = CFAbsoluteTimeGetCurrent()
        let vec = configuration.normalizeOutput ? PoolingHelpers.normalize(pooled) : pooled
        let tPool1 = CFAbsoluteTimeGetCurrent()

        let dt = CFAbsoluteTimeGetCurrent() - tStart
        metricsData.record(tokenCount: inputInfo.attentionMask.reduce(0, +), time: dt)
        // tokenization time = tTok1 - tStart, inference = tInf1 - tTok1, pooling = tPool1 - tPool0
        let tokDur = tTok1 - tStart
        let infDur = tInf1 - tTok1
        let poolDur = tPool1 - tPool0
        stageAgg.record(tokenization: tokDur, inference: infDur, pooling: poolDur, items: 1, batches: 1)
        profiler?.recordStage(model: id, items: 1, tokenization: tokDur, inference: infDur, pooling: poolDur, context: ["path": "single"])

        return Embedding(
            vector: vec,
            metadata: EmbeddingMetadata(
                modelID: id,
                tokenCount: inputInfo.attentionMask.reduce(0, +),
                processingTime: dt,
                normalized: configuration.normalizeOutput,
                poolingStrategy: configuration.poolingStrategy,
                truncated: inputInfo.originalLength > configuration.maxTokens,
                custom: [:]
            )
        )
    }

    /// Compute embeddings for a batch of texts. May perform micro‑batching by length to minimize padding.
    public func embedBatch(_ texts: [String], options: BatchOptions) async throws -> [Embedding] {
        guard let backend else { throw EmbedKitError.modelLoadFailed("Backend unavailable") }

        // Optional: length sort for better batching; keep output order stable
        let indices: [Int] = options.sortByLength ? texts.indices.sorted { texts[$0].count < texts[$1].count } : Array(texts.indices)

        // Tokenize all inputs and compute original lengths
        let tTokStart = CFAbsoluteTimeGetCurrent()
        struct Info { let idx: Int; let ids: [Int]; let mask: [Int]; let originalLen: Int }
        var infos: [Info] = []
        infos.reserveCapacity(texts.count)
        let conc = options.tokenizationConcurrency
        if conc > 1 {
            var pos = 0
            while pos < indices.count {
                let end = Swift.min(indices.count, pos + conc)
                let chunk = Array(indices[pos..<end])
                try await withThrowingTaskGroup(of: Info.self) { group in
                    for idx in chunk {
                        group.addTask {
                            if self.configuration.paddingStrategy == .batch {
                                let pre = TokenizerConfig(
                                    maxLength: Int.max,  // No truncation in pre-tokenization
                                    truncation: .end,
                                    padding: .none,
                                    addSpecialTokens: self.configuration.includeSpecialTokens
                                )
                                let tk = try await self.cachedEncode(prefix: "pre", text: texts[idx], config: pre)
                                return Info(idx: idx, ids: tk.ids, mask: tk.attentionMask, originalLen: tk.ids.count)
                            } else {
                                let built = try await self.buildInput(for: texts[idx])
                                return Info(idx: idx, ids: built.tokenIDs, mask: built.attentionMask, originalLen: built.originalLength)
                            }
                        }
                    }
                    for try await item in group { infos.append(item) }
                }
                pos = end
            }
            infos.sort { $0.idx < $1.idx }
        } else {
            for idx in indices {
                if configuration.paddingStrategy == .batch {
                    let pre = TokenizerConfig(
                        maxLength: Int.max,  // No truncation in pre-tokenization
                        truncation: .end,
                        padding: .none,
                        addSpecialTokens: configuration.includeSpecialTokens
                    )
                    let tk = try await self.cachedEncode(prefix: "pre", text: texts[idx], config: pre)
                    infos.append(Info(idx: idx, ids: tk.ids, mask: tk.attentionMask, originalLen: tk.ids.count))
                } else {
                    let built = try await self.buildInput(for: texts[idx])
                    infos.append(Info(idx: idx, ids: built.tokenIDs, mask: built.attentionMask, originalLen: built.originalLength))
                }
            }
        }

        // If requested, sort by tokenized length (more accurate than character count) for better bucketing.
        if options.sortByLength && configuration.paddingStrategy == .batch {
            infos.sort { (a, b) -> Bool in
                if a.ids.count == b.ids.count { return a.idx < b.idx }
                return a.ids.count < b.ids.count
            }
        }
        let tTokEnd = CFAbsoluteTimeGetCurrent()

        if await !backend.isLoaded {
            if let ml = backend as? CoreMLBackend {
                let bs = min(options.maxBatchSize, max(1, infos.count))
                if configuration.paddingStrategy == .batch {
                    let M = max(1, options.bucketSize)
                    let baseLen = max(1, infos.first?.ids.count ?? 1)
                    let padded = ((baseLen + M - 1) / M) * M
                    await ml.setAutoWorkloadHint(paddedLength: min(configuration.maxTokens, padded), batchSize: bs)
                } else {
                    let maxLen = min(configuration.maxTokens, infos.map { $0.ids.count }.max() ?? 1)
                    await ml.setAutoWorkloadHint(paddedLength: maxLen, batchSize: bs)
                }
            }
            try await backend.load()
        }
        let tStart = CFAbsoluteTimeGetCurrent()

        var results: [Embedding?] = Array(repeating: nil, count: texts.count)

        if configuration.paddingStrategy == .batch {
            // Micro-batching by length buckets
            let M = max(1, options.bucketSize)
            func bucketKey(_ len: Int) -> Int {
                // Ensure empty sequences are assigned at least one bucket of size M
                let rounded = ((max(1, len) + M - 1) / M) * M
                return min(configuration.maxTokens, rounded)
            }
            guard let pad = tokenizer.specialTokens.pad else {
                throw EmbedKitError.invalidConfiguration("Batch padding requires PAD token in vocabulary")
            }
            var start = 0
            var infTotal: TimeInterval = 0
            var poolTotal: TimeInterval = 0
            var processedCount = 0
            var batchCount = 0
            while start < infos.count {
                let key = bucketKey(infos[start].ids.count)
                logger.debug("Forming micro-batch", metadata: [
                    "bucketKey": .string("\(key)")
                ])
                var end = start
                var count = 0
                let maxCountBySize = options.maxBatchSize
                let maxCountByTokens = options.maxBatchTokens.map { max(1, $0 / key) } ?? Int.max
                let minDesired = options.minBatchSize ?? 1
                let maxAllowed = max(minDesired, min(maxCountBySize, maxCountByTokens))
                // Precompute padding ratio threshold (if any)
                let minPreLenForRatio: Int? = options.maxPaddingRatio.map { ratio in
                    let thr = Double(key) * (1.0 - max(0.0, min(1.0, ratio)))
                    return Int(ceil(thr))
                }
                var addedAny = false
                while end < infos.count && count < maxAllowed {
                    if bucketKey(infos[end].ids.count) != key { break }
                    if let thr = minPreLenForRatio, infos[end].ids.count < thr {
                        break
                    }
                    end += 1; count += 1; addedAny = true
                }
                if !addedAny {
                    end = min(infos.count, start + 1)
                    count = end - start
                }
                let slice = infos[start..<end]
                // Convert to Array to rebase indices to 0..<(count), avoiding ArraySlice index pitfalls
                let sliceArray = Array(slice)
                let targetLen = key
                var providers: [CoreMLInput] = []
                providers.reserveCapacity(slice.count)
                // Record micro-batch before inference
                profiler?.recordMicroBatch(model: id, batchSize: sliceArray.count, paddedLength: targetLen, device: device, context: [:])
                for item in slice {
                    let preLen = item.ids.count
                    var ids = item.ids
                    var mask = item.mask
                    if preLen > targetLen {
                        // Respect configured truncation behavior for batch path
                        switch configuration.truncationStrategy {
                        case .none:
                            throw EmbedKitError.inputTooLong(length: preLen, max: targetLen)
                        case .end:
                            ids = Array(ids.prefix(targetLen))
                            mask = Array(mask.prefix(targetLen))
                        case .start:
                            ids = Array(ids.suffix(targetLen))
                            mask = Array(mask.suffix(targetLen))
                        case .middle:
                            let keep = targetLen
                            let head = keep / 2
                            let tail = keep - head
                            ids = Array(ids.prefix(head) + ids.suffix(tail))
                            mask = Array(mask.prefix(head) + mask.suffix(tail))
                        }
                    } else if preLen < targetLen {
                        let diff = targetLen - preLen
                        ids += Array(repeating: pad.id, count: diff)
                        mask += Array(repeating: 0, count: diff)
                    }
                    // Padding invariant per item
                    guard ids.count == mask.count else {
                        throw EmbedKitError.invalidConfiguration("ids/mask length mismatch in batch item: ids=\(ids.count), mask=\(mask.count)")
                    }
                    providers.append(CoreMLInput(tokenIDs: ids, attentionMask: mask))
                }
                let tInf0 = CFAbsoluteTimeGetCurrent()
                let outs = try await backend.processBatch(providers)
                let tInf1 = CFAbsoluteTimeGetCurrent()
                infTotal += (tInf1 - tInf0)

                let tPoolStart = CFAbsoluteTimeGetCurrent()

                // Validate shapes and determine if batch GPU processing is beneficial
                guard !outs.isEmpty else { start = end; continue }
                let (firstTokens, firstDim) = try inferTokensAndDim(from: outs[0].shape, valuesCount: outs[0].values.count)
                if firstDim != dimensions { throw EmbedKitError.dimensionMismatch(expected: dimensions, got: firstDim) }

                logger.debug("Batch output shape resolved", metadata: [
                    "items": .string("\(outs.count)"),
                    "shape": .string("\(outs[0].shape)"),
                    "tokens": .string("\(firstTokens)"),
                    "dim": .string("\(firstDim)")
                ])

                // Determine if batch GPU operations should be used
                let totalElements = outs.count * firstTokens * firstDim
                let useBatchGPU = totalElements >= configuration.minElementsForGPU && outs.count > 1

                if useBatchGPU, let acc = await ensureMetal() {
                    // Metal 4 Tensor Operations: Process entire micro-batch in single GPU dispatch
                    // This is ~62% faster than per-item processing

                    // Concatenate all outputs into single flattened array
                    var flatEmbeddings: [Float] = []
                    flatEmbeddings.reserveCapacity(totalElements)
                    var flatMasks: [Int32] = []
                    flatMasks.reserveCapacity(outs.count * firstTokens)
                    var metaBatch: [(originalIndex: Int, tokenCount: Int, truncated: Bool)] = []
                    metaBatch.reserveCapacity(outs.count)

                    var allSameShape = true
                    for (i, out) in outs.enumerated() {
                        let (tokens, dim) = try inferTokensAndDim(from: out.shape, valuesCount: out.values.count)
                        if tokens != firstTokens || dim != firstDim {
                            allSameShape = false
                            break
                        }
                        flatEmbeddings.append(contentsOf: out.values)
                        flatMasks.append(contentsOf: providers[i].attentionMask.map { Int32($0) })
                        let tokenCount = providers[i].attentionMask.reduce(0, +)
                        metaBatch.append((sliceArray[i].idx, tokenCount, sliceArray[i].originalLen > configuration.maxTokens))
                    }

                    if allSameShape {
                        // Single GPU dispatch for entire micro-batch
                        let pooledNormalized = await acc.tensorPoolNormalize(
                            embeddings: flatEmbeddings,
                            batchSize: outs.count,
                            sequenceLength: firstTokens,
                            dimensions: firstDim,
                            masks: flatMasks,
                            strategy: configuration.poolingStrategy,
                            normalize: configuration.normalizeOutput
                        )

                        let tPoolEnd = CFAbsoluteTimeGetCurrent()
                        poolTotal += (tPoolEnd - tPoolStart)

                        for (k, vec) in pooledNormalized.enumerated() {
                            let meta = metaBatch[k]
                            let elapsed = (CFAbsoluteTimeGetCurrent() - tStart) / Double(texts.count)
                            metricsData.record(tokenCount: meta.tokenCount, time: elapsed)
                            results[meta.originalIndex] = Embedding(
                                vector: vec,
                                metadata: EmbeddingMetadata(
                                    modelID: id,
                                    tokenCount: meta.tokenCount,
                                    processingTime: elapsed,
                                    normalized: configuration.normalizeOutput,
                                    poolingStrategy: configuration.poolingStrategy,
                                    truncated: meta.truncated,
                                    custom: [:]
                                )
                            )
                        }
                    } else {
                        // Fall back to per-item processing if shapes differ
                        let itemMetas = sliceArray.map { BatchItemMeta(originalIndex: $0.idx, originalLen: $0.originalLen) }
                        try await processItemsIndividually(
                            outs: outs, providers: providers, itemMetas: itemMetas,
                            tStart: tStart, textsCount: texts.count, results: &results, poolTotal: &poolTotal
                        )
                    }
                } else {
                    // CPU or per-item GPU processing for small batches
                    let itemMetas = sliceArray.map { BatchItemMeta(originalIndex: $0.idx, originalLen: $0.originalLen) }
                    try await processItemsIndividually(
                        outs: outs, providers: providers, itemMetas: itemMetas,
                        tStart: tStart, textsCount: texts.count, results: &results, poolTotal: &poolTotal
                    )
                }
                start = end
                processedCount += slice.count
                batchCount += 1
            }
            // Record average per-item stage timings for the batch based on processed items
            let items = max(1, processedCount)
            let tokAvg = (tTokEnd - tTokStart) / Double(max(1, texts.count))
            let infAvg = infTotal / Double(items)
            let poolAvg = poolTotal / Double(items)
            stageAgg.record(tokenization: tokAvg, inference: infAvg, pooling: poolAvg, items: items, batches: max(1, batchCount))
            profiler?.recordStage(model: id, items: items, tokenization: tokAvg, inference: infAvg, pooling: poolAvg, context: ["path": "batch"])
            if batchCount > 0 {
                let avgBatch = Double(processedCount) / Double(batchCount)
                logger.debug("Batch summary", metadata: [
                    "items": .string("\(processedCount)"),
                    "batches": .string("\(batchCount)"),
                    "avgBatchSize": .string(String(format: "%.2f", avgBatch))
                ])
            }
        } else {
            // Single-batch path
            // Validate invariants for single-batch path and build providers
            var providers: [CoreMLInput] = []
            providers.reserveCapacity(infos.count)
            for it in infos {
                guard it.ids.count == it.mask.count else {
                    throw EmbedKitError.invalidConfiguration("ids/mask length mismatch: ids=\(it.ids.count), mask=\(it.mask.count)")
                }
                providers.append(CoreMLInput(tokenIDs: it.ids, attentionMask: it.mask))
            }
            let tInf0 = CFAbsoluteTimeGetCurrent()
            let outs = try await backend.processBatch(providers)
            let tInf1 = CFAbsoluteTimeGetCurrent()
            var poolSum: TimeInterval = 0
            let tPool0 = CFAbsoluteTimeGetCurrent()

            // Validate shapes and determine if batch GPU processing is beneficial
            guard !outs.isEmpty else {
                let items = max(1, texts.count)
                let tokAvg = (tTokEnd - tTokStart) / Double(items)
                stageAgg.record(tokenization: tokAvg, inference: 0, pooling: 0, items: items, batches: 1)
                return []
            }

            let (firstTokens, firstDim) = try inferTokensAndDim(from: outs[0].shape, valuesCount: outs[0].values.count)
            if firstDim != dimensions { throw EmbedKitError.dimensionMismatch(expected: dimensions, got: firstDim) }

            // Determine if batch GPU operations should be used
            let totalElements = outs.count * firstTokens * firstDim
            let useBatchGPU = totalElements >= configuration.minElementsForGPU && outs.count > 1

            if useBatchGPU, let acc = await ensureMetal() {
                // Metal 4 Tensor Operations: Process entire batch in single GPU dispatch

                // Concatenate all outputs into single flattened array
                var flatEmbeddings: [Float] = []
                flatEmbeddings.reserveCapacity(totalElements)
                var flatMasks: [Int32] = []
                flatMasks.reserveCapacity(outs.count * firstTokens)
                var metaBatch: [(originalIndex: Int, tokenCount: Int, truncated: Bool)] = []
                metaBatch.reserveCapacity(outs.count)

                var allSameShape = true
                for (i, out) in outs.enumerated() {
                    let (tokens, dim) = try inferTokensAndDim(from: out.shape, valuesCount: out.values.count)
                    if tokens != firstTokens || dim != firstDim {
                        allSameShape = false
                        break
                    }
                    flatEmbeddings.append(contentsOf: out.values)
                    flatMasks.append(contentsOf: providers[i].attentionMask.map { Int32($0) })
                    let tokenCount = providers[i].attentionMask.reduce(0, +)
                    metaBatch.append((infos[i].idx, tokenCount, infos[i].originalLen > configuration.maxTokens))
                }

                if allSameShape {
                    // Single GPU dispatch for entire batch
                    let pooledNormalized = await acc.tensorPoolNormalize(
                        embeddings: flatEmbeddings,
                        batchSize: outs.count,
                        sequenceLength: firstTokens,
                        dimensions: firstDim,
                        masks: flatMasks,
                        strategy: configuration.poolingStrategy,
                        normalize: configuration.normalizeOutput
                    )

                    let tPool1 = CFAbsoluteTimeGetCurrent()
                    poolSum = tPool1 - tPool0

                    for (k, vec) in pooledNormalized.enumerated() {
                        let meta = metaBatch[k]
                        let elapsed = (CFAbsoluteTimeGetCurrent() - tStart) / Double(texts.count)
                        metricsData.record(tokenCount: meta.tokenCount, time: elapsed)
                        results[meta.originalIndex] = Embedding(
                            vector: vec,
                            metadata: EmbeddingMetadata(
                                modelID: id,
                                tokenCount: meta.tokenCount,
                                processingTime: elapsed,
                                normalized: configuration.normalizeOutput,
                                poolingStrategy: configuration.poolingStrategy,
                                truncated: meta.truncated,
                                custom: [:]
                            )
                        )
                    }
                } else {
                    // Fall back to per-item processing if shapes differ
                    let itemMetas = infos.map { BatchItemMeta(originalIndex: $0.idx, originalLen: $0.originalLen) }
                    try await processItemsIndividually(
                        outs: outs, providers: providers, itemMetas: itemMetas,
                        tStart: tStart, textsCount: texts.count, results: &results, poolTotal: &poolSum
                    )
                }
            } else {
                // CPU or per-item GPU processing for small batches
                let itemMetas = infos.map { BatchItemMeta(originalIndex: $0.idx, originalLen: $0.originalLen) }
                try await processItemsIndividually(
                    outs: outs, providers: providers, itemMetas: itemMetas,
                    tStart: tStart, textsCount: texts.count, results: &results, poolTotal: &poolSum
                )
            }

            let items = max(1, texts.count)
            let tokAvg = (tTokEnd - tTokStart) / Double(items)
            let infAvg = (tInf1 - tInf0) / Double(items)
            let poolAvg = poolSum / Double(items)
            stageAgg.record(tokenization: tokAvg, inference: infAvg, pooling: poolAvg, items: items, batches: 1)
            profiler?.recordStage(model: id, items: items, tokenization: tokAvg, inference: infAvg, pooling: poolAvg, context: ["path": "single-batch"])
        }

        return results.compactMap { $0 }
    }

    public func warmup() async throws {
        try await backend?.load()
    }

    public func release() async throws {
        try await backend?.unload()
    }

    /// Proactively trim memory by clearing caches and optionally unloading the backend.
    /// - Parameter aggressive: When true, also unloads the backend to free model memory.
    public func trimMemory(aggressive: Bool = false) async {
        await tokenCache.reset()
        if aggressive {
            try? await backend?.unload()
        }
    }

    // MARK: - CoreML I/O override convenience
    /// Set CoreML input feature key overrides. If the backend is not CoreML-based, this is a no-op.
    public func setCoreMLInputKeyOverrides(token: String? = nil, mask: String? = nil, type: String? = nil, pos: String? = nil) async {
        if let ml = backend as? CoreMLBackend {
            await ml.setInputKeyOverrides(token: token, mask: mask, type: type, pos: pos)
        }
    }

    /// Set CoreML output feature key override. If the backend is not CoreML-based, this is a no-op.
    public func setCoreMLOutputKeyOverride(_ key: String?) async {
        if let ml = backend as? CoreMLBackend {
            await ml.setOutputKeyOverride(key)
        }
    }

    public var metrics: ModelMetrics {
        get async {
            let stats = await tokenCache.stats()
            return metricsData.snapshot(memoryUsage: currentMemoryUsage(), cacheHitRate: stats.hitRate)
        }
    }

    /// Reset aggregate performance counters.
    /// Reset aggregate performance counters and stage averages. Cache contents are preserved, but stats reset.
    public func resetMetrics() async throws {
        metricsData = MetricsData()
        stageAgg = StageAgg()
        await tokenCache.reset()
    }
    /// Average per‑stage timings (tokenization, inference, pooling) collected since the last reset.
    public var stageMetricsSnapshot: StageMetrics { stageAgg.snapshot() }

    // MARK: - Input Builder
    struct AppleInputInfo { let tokenIDs: [Int]; let attentionMask: [Int]; let originalLength: Int }

    /// Build CoreML input from text using the configured tokenizer and compute original length
    /// before truncation/padding for accurate truncation metadata.
    func buildInput(for text: String) async throws -> AppleInputInfo {
        let tk = TokenizerConfig(
            maxLength: configuration.maxTokens,
            truncation: configuration.truncationStrategy,
            padding: configuration.paddingStrategy,
            addSpecialTokens: configuration.includeSpecialTokens
        )

        // Compute original length without truncation/padding
        let pre = TokenizerConfig(
            maxLength: Int.max,  // No truncation
            truncation: .end,
            padding: .none,
            addSpecialTokens: configuration.includeSpecialTokens
        )

        let preTokenized = try await cachedEncode(prefix: "pre", text: text, config: pre)
        let originalLen = preTokenized.ids.count

        let tokenized = try await cachedEncode(prefix: "cfg", text: text, config: tk)
        return AppleInputInfo(tokenIDs: tokenized.ids, attentionMask: tokenized.attentionMask, originalLength: originalLen)
    }

    // Cached tokenization helper
    private func cachedEncode(prefix: String, text: String, config: TokenizerConfig) async throws -> TokenizedText {
        let key = "\(prefix)|\(configSignature(config))|\(text)"
        if let cached = await tokenCache.get(key) { return cached }
        let tk = try await tokenizer.encode(text, config: config)
        await tokenCache.put(key, tk)
        return tk
    }

    // Compact fingerprint of tokenization config to avoid cross-config cache collisions
    private func configSignature(_ c: TokenizerConfig) -> String {
        let t: String
        switch c.truncation {
        case .none: t = "n"
        case .end: t = "e"
        case .start: t = "s"
        case .middle: t = "m"
        }
        let p: String
        switch c.padding {
        case .none: p = "n"
        case .max: p = "x"
        case .batch: p = "b"
        }
        let a = c.addSpecialTokens ? "1" : "0"
        let r = c.returnOffsets ? "1" : "0"
        return "L\(c.maxLength)_T\(t)_P\(p)_A\(a)_R\(r)"
    }

    // MARK: - Metal accelerator
    private func ensureMetal() async -> MetalAccelerator? {
        if metalAccelerator == nil {
            metalAccelerator = await MetalAccelerator()
        }
        if let acc = metalAccelerator {
            if await acc.isAvailable { return acc }
        }
        return nil
    }

    // MARK: - Batch Processing Helpers

    /// Metadata for batch item processing
    private struct BatchItemMeta {
        let originalIndex: Int
        let originalLen: Int
    }

    /// Process outputs individually (fallback when batch GPU isn't beneficial or shapes vary)
    private func processItemsIndividually(
        outs: [CoreMLOutput],
        providers: [CoreMLInput],
        itemMetas: [BatchItemMeta],
        tStart: CFAbsoluteTime,
        textsCount: Int,
        results: inout [Embedding?],
        poolTotal: inout TimeInterval
    ) async throws {
        for (i, out) in outs.enumerated() {
            let (tokens, dim) = try inferTokensAndDim(from: out.shape, valuesCount: out.values.count)
            if dim != dimensions { throw EmbedKitError.dimensionMismatch(expected: dimensions, got: dim) }

            let useGPUPooling = tokens * dim >= configuration.minElementsForGPU
            let tPool0 = CFAbsoluteTimeGetCurrent()

            let pooled: [Float]
            switch configuration.poolingStrategy {
            case .mean:
                if useGPUPooling, let acc = await ensureMetal() {
                    pooled = await acc.meanPool(
                        embeddings: out.values,
                        sequenceLength: tokens,
                        dimensions: dim,
                        mask: providers[i].attentionMask
                    )
                } else {
                    pooled = PoolingHelpers.mean(sequence: out.values, tokens: tokens, dim: dim, mask: providers[i].attentionMask)
                }
            case .max:
                if useGPUPooling, let acc = await ensureMetal() {
                    pooled = await acc.maxPool(
                        embeddings: out.values,
                        sequenceLength: tokens,
                        dimensions: dim,
                        mask: providers[i].attentionMask
                    )
                } else {
                    pooled = PoolingHelpers.max(sequence: out.values, tokens: tokens, dim: dim, mask: providers[i].attentionMask)
                }
            case .cls:
                pooled = PoolingHelpers.cls(sequence: out.values, tokens: tokens, dim: dim)
            case .attention:
                // Attention pooling without weights falls back to mean
                if useGPUPooling, let acc = await ensureMetal() {
                    pooled = await acc.meanPool(
                        embeddings: out.values,
                        sequenceLength: tokens,
                        dimensions: dim,
                        mask: providers[i].attentionMask
                    )
                } else {
                    pooled = PoolingHelpers.mean(sequence: out.values, tokens: tokens, dim: dim, mask: providers[i].attentionMask)
                }
            }

            let vec = configuration.normalizeOutput ? PoolingHelpers.normalize(pooled) : pooled
            let tPool1 = CFAbsoluteTimeGetCurrent()
            poolTotal += (tPool1 - tPool0)

            let meta = itemMetas[i]
            let tokenCount = providers[i].attentionMask.reduce(0, +)
            let elapsed = (CFAbsoluteTimeGetCurrent() - tStart) / Double(textsCount)
            metricsData.record(tokenCount: tokenCount, time: elapsed)
            results[meta.originalIndex] = Embedding(
                vector: vec,
                metadata: EmbeddingMetadata(
                    modelID: id,
                    tokenCount: tokenCount,
                    processingTime: elapsed,
                    normalized: configuration.normalizeOutput,
                    poolingStrategy: configuration.poolingStrategy,
                    truncated: meta.originalLen > configuration.maxTokens,
                    custom: [:]
                )
            )
        }
    }

    // MARK: - Shape inference
    nonisolated func inferTokensAndDim(from shape: [Int], valuesCount: Int) throws -> (tokens: Int, dim: Int) {
        // Common layouts observed:
        // - 3D: [1, seq, dim]
        // - 2D: [seq, dim] or [1, dim] or sometimes [dim, seq]
        // - 1D: [dim] (already pooled)
        if shape.count == 3 {
            // e.g., [1, seq, dim]
            return (tokens: shape[1], dim: shape[2])
        } else if shape.count == 2 {
            let a = shape[0], b = shape[1]
            // Ensure the provided shape is consistent with valuesCount if possible
            if a * b == valuesCount {
                // Heuristics to disambiguate:
                // - If a == 1, prefer [1, dim]
                if a == 1 { return (tokens: 1, dim: b) }
                // - If b == dimensions, prefer [seq=a, dim=b]
                if b == dimensions { return (tokens: a, dim: b) }
                // - If a == dimensions, interpret as [dim, seq] and swap
                if a == dimensions { return (tokens: b, dim: a) }
                // - If b == 1, degenerate case: [seq, 1]
                if b == 1 { return (tokens: a, dim: 1) }
                // Default: assume [seq, dim]
                return (tokens: a, dim: b)
            }
        } else if shape.count == 1 {
            // Already pooled vector
            return (tokens: 1, dim: shape[0])
        }
        // Fallback to factorization using known dim when shape metadata is unreliable.
        if dimensions > 0, valuesCount % dimensions == 0 {
            return (tokens: valuesCount / dimensions, dim: dimensions)
        }
        throw EmbedKitError.invalidConfiguration("Unrecognized CoreML output shape: \(shape) with count=\(valuesCount)")
    }
}

// MARK: - Stage aggregation (tokenization, inference, pooling)
fileprivate struct StageAgg {
    private var n: Int = 0
    private var tokHistogram = LatencyHistogram()
    private var infHistogram = LatencyHistogram()
    private var poolHistogram = LatencyHistogram()
    private var itemSum: Int = 0
    private var batchSum: Int = 0

    mutating func record(tokenization: TimeInterval, inference: TimeInterval, pooling: TimeInterval, items: Int, batches: Int) {
        n &+= 1
        tokHistogram.record(tokenization)
        infHistogram.record(inference)
        poolHistogram.record(pooling)
        itemSum &+= max(0, items)
        batchSum &+= max(0, batches)
    }

    mutating func reset() {
        n = 0
        tokHistogram.reset()
        infHistogram.reset()
        poolHistogram.reset()
        itemSum = 0
        batchSum = 0
    }

    func snapshot() -> StageMetrics {
        let tokStats = tokHistogram.statistics
        let infStats = infHistogram.statistics
        let poolStats = poolHistogram.statistics
        let avgBatch = batchSum > 0 ? Double(itemSum) / Double(batchSum) : 1.0
        return StageMetrics(
            tokenizationAverage: tokStats.mean,
            inferenceAverage: infStats.mean,
            poolingAverage: poolStats.mean,
            samples: n,
            averageBatchSize: avgBatch,
            tokenizationStats: n > 0 ? tokStats : nil,
            inferenceStats: n > 0 ? infStats : nil,
            poolingStats: n > 0 ? poolStats : nil
        )
    }
}
