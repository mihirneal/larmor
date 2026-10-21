//! Bit-exact ports of the random generators the eval framework's numbers depend on.
//!
//! - [`RandomState`]: numpy's legacy `RandomState(seed)` (MT19937 with Knuth seeding). sklearn's `KFold` and
//!   `StratifiedKFold` shuffle through `check_random_state(seed)`, which builds one of these.
//! - [`Generator`]: numpy's `default_rng(seed)` (`SeedSequence` feeding `PCG64`), used for bootstrap indices.
//! - [`torch_randperm`]: `torch.randperm(n, generator=torch.Generator().manual_seed(seed))` on the CPU, used
//!   by `subsample_negatives`.

/// The Mersenne Twister MT19937, as numpy's `_mt19937` and ATen's `at::mt19937` implement it.
#[derive(Clone)]
pub struct Mt19937 {
    key: [u32; 624],
    pos: usize,
}

impl Mt19937 {
    const N: usize = 624;
    const M: usize = 397;

    /// Seeds like numpy `mt19937_seed` and ATen `mt19937::init_with_uint32` (Knuth's multiplier).
    pub fn new(seed: u32) -> Self {
        let mut key = [0u32; Self::N];
        let mut value = seed;
        for (pos, slot) in key.iter_mut().enumerate() {
            *slot = value;
            value = 1_812_433_253u32.wrapping_mul(value ^ (value >> 30)).wrapping_add(pos as u32 + 1);
        }
        Self { key, pos: Self::N }
    }

    fn regenerate(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        for i in 0..Self::N {
            let y = (self.key[i] & UPPER) | (self.key[(i + 1) % Self::N] & LOWER);
            let twist = if y & 1 == 1 { MATRIX_A } else { 0 };
            self.key[i] = self.key[(i + Self::M) % Self::N] ^ (y >> 1) ^ twist;
        }
        self.pos = 0;
    }

    /// The next tempered 32-bit output (`mt19937_next32`).
    pub fn next_u32(&mut self) -> u32 {
        if self.pos == Self::N {
            self.regenerate();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// Two 32-bit outputs, high word first (`mt19937_next64`).
    pub fn next_u64(&mut self) -> u64 {
        let high = u64::from(self.next_u32());
        (high << 32) | u64::from(self.next_u32())
    }
}

/// numpy's legacy `RandomState`, limited to what sklearn's splitters use.
#[derive(Clone)]
pub struct RandomState {
    mt: Mt19937,
}

impl RandomState {
    /// `np.random.RandomState(seed)` for an integer seed (`_legacy_seeding`).
    pub fn new(seed: u32) -> Self {
        Self { mt: Mt19937::new(seed) }
    }

    /// Uniform integer in `[0, max]` by masked rejection (numpy `random_interval`).
    pub fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mask = smallest_mask(max);
        if max <= u64::from(u32::MAX) {
            loop {
                let value = u64::from(self.mt.next_u32()) & mask;
                if value <= max {
                    return value;
                }
            }
        }
        loop {
            let value = self.mt.next_u64() & mask;
            if value <= max {
                return value;
            }
        }
    }

    /// In-place Fisher-Yates shuffle from the back (`RandomState.shuffle` on a 1-D array).
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.interval(i as u64) as usize;
            values.swap(i, j);
        }
    }

    /// The raw 32-bit stream, for tests.
    pub fn next_u32(&mut self) -> u32 {
        self.mt.next_u32()
    }
}

fn smallest_mask(max: u64) -> u64 {
    let mut mask = max;
    for shift in [1, 2, 4, 8, 16, 32] {
        mask |= mask >> shift;
    }
    mask
}

/// numpy's `SeedSequence(entropy)` with the default pool of four words and no spawn key.
#[derive(Clone, Debug)]
pub struct SeedSequence {
    pool: [u32; 4],
}

impl SeedSequence {
    const INIT_A: u32 = 0x43b0_d7e5;
    const MULT_A: u32 = 0x931e_8875;
    const INIT_B: u32 = 0x8b51_f9dd;
    const MULT_B: u32 = 0x58f3_8ded;
    const MIX_MULT_L: u32 = 0xca01_f9dd;
    const MIX_MULT_R: u32 = 0x4973_f715;
    const XSHIFT: u32 = 16;

    /// `SeedSequence(entropy)` for a non-negative integer.
    pub fn new(entropy: u128) -> Self {
        Self::from_words(&uint32_words(entropy))
    }

    /// `SeedSequence(entropy)` from the entropy's 32-bit words, lowest first (`mix_entropy`).
    pub fn from_words(words: &[u32]) -> Self {
        let mut hash_const = Self::INIT_A;
        let mut pool = [0u32; 4];
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = Self::hashmix(words.get(i).copied().unwrap_or(0), &mut hash_const);
        }
        for src in 0..pool.len() {
            for dst in 0..pool.len() {
                if src != dst {
                    let hashed = Self::hashmix(pool[src], &mut hash_const);
                    pool[dst] = Self::mix(pool[dst], hashed);
                }
            }
        }
        for &word in words.iter().skip(pool.len()) {
            for slot in pool.iter_mut() {
                let hashed = Self::hashmix(word, &mut hash_const);
                *slot = Self::mix(*slot, hashed);
            }
        }
        Self { pool }
    }

    fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
        let mut value = value ^ *hash_const;
        *hash_const = hash_const.wrapping_mul(Self::MULT_A);
        value = value.wrapping_mul(*hash_const);
        value ^ (value >> Self::XSHIFT)
    }

    fn mix(x: u32, y: u32) -> u32 {
        let result = Self::MIX_MULT_L.wrapping_mul(x).wrapping_sub(Self::MIX_MULT_R.wrapping_mul(y));
        result ^ (result >> Self::XSHIFT)
    }

    /// `generate_state(n_words, np.uint64)`: pairs of 32-bit words, low word first.
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        let mut hash_const = Self::INIT_B;
        let words: Vec<u32> = (0..2 * n_words)
            .map(|i| {
                let mut value = self.pool[i % self.pool.len()] ^ hash_const;
                hash_const = hash_const.wrapping_mul(Self::MULT_B);
                value = value.wrapping_mul(hash_const);
                value ^ (value >> Self::XSHIFT)
            })
            .collect();
        words.chunks(2).map(|pair| u64::from(pair[0]) | (u64::from(pair[1]) << 32)).collect()
    }
}

/// An integer split into 32-bit words, lowest first; zero is one zero word (`_int_to_uint32_array`).
fn uint32_words(mut value: u128) -> Vec<u32> {
    if value == 0 {
        return vec![0];
    }
    let mut words = Vec::new();
    while value > 0 {
        words.push(value as u32);
        value >>= 32;
    }
    words
}

/// numpy's `PCG64` bit generator (XSL-RR 128/64) with its buffered 32-bit draws.
#[derive(Clone, Debug)]
pub struct Pcg64 {
    state: u128,
    increment: u128,
    buffered_u32: Option<u32>,
}

impl Pcg64 {
    const MULTIPLIER: u128 = (2_549_297_995_355_413_924u128 << 64) | 4_865_540_595_714_422_341u128;

    /// `PCG64(seed)`.
    pub fn new(seed: u128) -> Self {
        Self::from_seed_sequence(&SeedSequence::new(seed))
    }

    /// `PCG64(seed_sequence)`: four words from `generate_state(4, np.uint64)`, then `pcg64_set_seed`.
    pub fn from_seed_sequence(seed_sequence: &SeedSequence) -> Self {
        let words = seed_sequence.generate_state_u64(4);
        let initial_state = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let sequence = (u128::from(words[2]) << 64) | u128::from(words[3]);
        let mut generator = Self { state: 0, increment: (sequence << 1) | 1, buffered_u32: None };
        generator.step();
        generator.state = generator.state.wrapping_add(initial_state);
        generator.step();
        generator
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(Self::MULTIPLIER).wrapping_add(self.increment);
    }

    /// `pcg64_next64`: step, then the XSL-RR output of the new state.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let xored = ((self.state >> 64) as u64) ^ (self.state as u64);
        xored.rotate_right((self.state >> 122) as u32)
    }

    /// `pcg64_next32`: the low half of a 64-bit draw, keeping the high half for the next call.
    pub fn next_u32(&mut self) -> u32 {
        if let Some(high) = self.buffered_u32.take() {
            return high;
        }
        let next = self.next_u64();
        self.buffered_u32 = Some((next >> 32) as u32);
        next as u32
    }
}

/// numpy's `Generator` on `PCG64`, limited to `integers`.
#[derive(Clone, Debug)]
pub struct Generator {
    bits: Pcg64,
}

impl Generator {
    /// `np.random.default_rng(seed)`.
    pub fn default_rng(seed: u128) -> Self {
        Self { bits: Pcg64::new(seed) }
    }

    /// `np.random.default_rng(SeedSequence(...))`, for seeds wider than 128 bits.
    pub fn from_seed_sequence(seed_sequence: &SeedSequence) -> Self {
        Self { bits: Pcg64::from_seed_sequence(seed_sequence) }
    }

    /// `rng.integers(low, high, size)` with the default int64 dtype and `endpoint=False`
    /// (`random_bounded_uint64_fill` with Lemire's method, `use_masked=False`).
    pub fn integers(&mut self, low: i64, high: i64, size: usize) -> Vec<i64> {
        assert!(high > low, "integers needs low < high");
        let range = (high as i128 - low as i128 - 1) as u64;
        let offset = low;
        (0..size).map(|_| offset.wrapping_add(self.bounded(range) as i64)).collect()
    }

    /// Uniform in `[0, range]`. Ranges that fit 32 bits draw 32-bit words, which matters for the stream.
    fn bounded(&mut self, range: u64) -> u64 {
        if range == 0 {
            0
        } else if range < u64::from(u32::MAX) {
            u64::from(self.lemire_u32(range as u32))
        } else if range == u64::from(u32::MAX) {
            u64::from(self.bits.next_u32())
        } else if range == u64::MAX {
            self.bits.next_u64()
        } else {
            self.lemire_u64(range)
        }
    }

    fn lemire_u32(&mut self, range: u32) -> u32 {
        let range_excl = range + 1;
        let mut m = u64::from(self.bits.next_u32()) * u64::from(range_excl);
        let mut leftover = m as u32;
        if leftover < range_excl {
            let threshold = (u32::MAX - range) % range_excl;
            while leftover < threshold {
                m = u64::from(self.bits.next_u32()) * u64::from(range_excl);
                leftover = m as u32;
            }
        }
        (m >> 32) as u32
    }

    fn lemire_u64(&mut self, range: u64) -> u64 {
        let range_excl = range + 1;
        let mut m = u128::from(self.bits.next_u64()) * u128::from(range_excl);
        let mut leftover = m as u64;
        if leftover < range_excl {
            let threshold = (u64::MAX - range) % range_excl;
            while leftover < threshold {
                m = u128::from(self.bits.next_u64()) * u128::from(range_excl);
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// The raw 64-bit stream (`bit_generator.random_raw`), for tests.
    pub fn next_raw(&mut self) -> u64 {
        self.bits.next_u64()
    }
}

/// `torch.randperm(n, generator=torch.Generator().manual_seed(seed))` on the CPU.
///
/// ATen's `randperm_cpu` for `n < u32::MAX / 20` swaps forward with `generator->random() % (n - i)`, where
/// `random()` is one MT19937 output seeded by `init_with_uint32`.
pub fn torch_randperm(n: usize, seed: u64) -> Vec<i64> {
    assert!(n < (u32::MAX / 20) as usize, "torch switches to a 64-bit draw for n >= 2^32 / 20");
    let mut mt = Mt19937::new(seed as u32);
    let mut order: Vec<i64> = (0..n as i64).collect();
    for i in 0..n.saturating_sub(1) {
        let z = (u64::from(mt.next_u32()) % (n - i) as u64) as usize;
        order.swap(i, z + i);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_the_reference_first_output() {
        // first output of std::mt19937 with the default seed 5489, from the C++ standard
        let mut mt = Mt19937::new(5489);
        for _ in 0..9999 {
            mt.next_u32();
        }
        assert_eq!(mt.next_u32(), 4_123_659_995);
    }
}
