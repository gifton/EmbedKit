import Testing
@testable import EmbedKit

@Suite("Token Cache")
struct TokenCacheTestsSuite {
@Test
func tokenCache_lruEvictionOrder() async throws {
    let cache = TokenCache<String, Int>(capacity: 3)

    await cache.put("a", 1)
    await cache.put("b", 2)
    await cache.put("c", 3)

    // Access to update recency: a becomes MRU, b is LRU
    _ = await cache.get("a")

    // Insert d -> evict b
    await cache.put("d", 4)

    #expect(await cache.get("b") == nil)
    #expect(await cache.get("a") == 1)
    #expect(await cache.get("c") == 3)
    #expect(await cache.get("d") == 4)
}

@Test
func tokenCache_hitMissStats() async throws {
    let cache = TokenCache<String, Int>(capacity: 2)
    await cache.put("x", 10)
    _ = await cache.get("x") // hit
    _ = await cache.get("y") // miss
    let stats = await cache.stats()
    #expect(stats.hits == 1)
    #expect(stats.misses == 1)
    #expect(stats.total == 2)
    #expect(stats.hitRate >= 0.49)
}

@Test
func tokenCache_updateMovesToHead() async throws {
    let cache = TokenCache<String, Int>(capacity: 2)
    await cache.put("a", 1)
    await cache.put("b", 2)
    // Update a should make it MRU, so inserting c evicts b
    await cache.put("a", 11)
    await cache.put("c", 3)
    #expect(await cache.get("b") == nil)
    #expect(await cache.get("a") == 11)
    #expect(await cache.get("c") == 3)
}
}
