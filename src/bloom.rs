//! Minimal Bloom filter for deduplicating crawl URLs.

// • Bloom Filter Overview
//
//   - The module defines BloomFilter<const WORDS: usize>, a fixed-size bitset
//     backed by an array of WORDS 64-bit integers (words: [u64; WORDS]). The
//     const generic lets callers choose how many bits the filter tracks (WORDS
//     × 64).
//
//   Construction
//
//   - BloomFilter::new() initializes every word to zero, meaning no elements have
//     been seen.
//
//   Insertion Logic
//
//   - insert(&mut self, data: &[u8]) -> bool hashes the input with three
//     different seeds (BLOOM_HASH_SEEDS). Each hash picks a bit index in the
//     bitmap (idx = hash % bit_count, where bit_count = WORDS * 64).
//   - For each hashed bit, the function sets the bit if it was previously zero.
//     The inserted flag tracks whether any bit changed; returning false means all
//     bits were already set, so the value is probably a duplicate.
//   - If WORDS is zero, we can’t store any bits, so it immediately returns false.
//
//   Hash Function
//
//   - bloom_hash is a simple mixing function: XORs the seed with the data length,
//     then for each byte multiplies by a large constant, rotates left 13 bits,
//     multiplies again, and finally XORs the high bits into the low ones. This
//     produces a 64-bit pseudo-random value per seed.
//
//   Seeds
//
//   - BLOOM_HASH_SEEDS contains three well-spaced 64-bit values. Multiple seeds
//     reduce the false-positive probability compared to a single hash.
//
//   Usage
//
//   - Frontier keeps a BloomFilter to remember URLs it has already scheduled. On
//     every enqueue attempt it calls insert; returning false means “we’ve seen
//     this URL before,” so the task is rejected as a duplicate without touching
//     the queue.
//
//   This implementation avoids heap allocations, keeps hashing deterministic, and
//   fits well with the const-generic style of the rest of the "fastcrawl" project.

/// Fixed-size Bloom filter backed by a const-generic `u64` array.
pub(crate) struct BloomFilter<const WORDS: usize> {
    words: [u64; WORDS],
}

impl<const WORDS: usize> BloomFilter<WORDS> {
    pub const fn new() -> Self {
        Self {
            words: [0u64; WORDS],
        }
    }

    pub fn insert(&mut self, data: &[u8]) -> bool {
        if WORDS == 0 {
            return false;
        }
        let bit_count = WORDS * 64;
        let mut inserted = false;
        for &seed in BLOOM_HASH_SEEDS.iter() {
            let hash = bloom_hash(data, seed);
            let idx = (hash as usize) % bit_count;
            let word = idx / 64;
            let bit = idx % 64;
            let mask = 1u64 << bit;
            if self.words[word] & mask == 0 {
                inserted = true;
                self.words[word] |= mask;
            }
        }
        inserted
    }
}

const BLOOM_HASH_SEEDS: [u64; 3] = [
    0x517c_c1b7_2722_0a95,
    0x6d0f_27bd_ceb7_b067,
    0x9e37_79b1_85eb_ca87,
];

fn bloom_hash(data: &[u8], seed: u64) -> u64 {
    let mut hash = seed ^ data.len() as u64;
    for &byte in data {
        // wrapping_mul is Rust’s integer multiplication that
        // discards any overflow and just keeps the low bits,
        // “wrapping” around modulo 2⁶⁴.
        hash ^= (byte as u64).wrapping_mul(0x1000_0000_01b3);
        hash = hash.rotate_left(13).wrapping_mul(0xff51_afd7_ed55_8ccd);
    }
    hash ^ (hash >> 33)
}
