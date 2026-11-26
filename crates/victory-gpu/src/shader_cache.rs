//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content, translator sources,
//! resource ABI and translation settings. Rebuilding unrelated components
//! preserves cache hits. Entries go through
//! `victory-pipeline`'s content-addressed cache, which writes them
//! atomically; an entry that does not read back is translated again.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway).

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, Target, UserData};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// Stable cache identity: the source fingerprint, encoding ABI and settings.
/// The legacy diagnostic override is retained for explicitly matched old
/// executables only. Normal builds never hash the whole EXE.
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let limit = victory_shader::emit::loop_limit();
        let rays = victory_shader::emit::rays();
        if let Some(other) = std::env::var_os("VICTORY_SHADER_BUILD_OF")
            .filter(|_| cfg!(feature = "diagnostics"))
        {
            let other = PathBuf::from(other);
            if let Ok(mut bytes) = std::fs::read(&other) {
                tracing::warn!(of = %other.display(), "explicit legacy shader cache override; translation must match");
                bytes.extend_from_slice(&limit.to_le_bytes());
                bytes.push(rays.into());
                return *blake3::hash(&bytes).as_bytes();
            }
            tracing::warn!(of = %other.display(), "legacy shader cache override unreadable; using source identity");
        }
        // Effective values for target, wave mode, heap clamp, optimisation
        // passes and raw CBVs still join key() below. Diagnostic patches
        // remain scoped to the matching containers in host_shader().
        let mut settings = Vec::new();
        for name in victory_shader::TRANSLATOR_SWITCHES.split(',') {
            if name.is_empty() || matches!(name,
                "VICTORY_SHADER_LOOP_LIMIT" | "VICTORY_RAYS" | "VICTORY_WAVE64_MODE"
                | "VICTORY_SHADER_OPT" | "VICTORY_PORT_CBV" | "VICTORY_SHADER_PATCH"
                | "VICTORY_WHY")
            {
                continue;
            }
            if let Some(value) = std::env::var_os(name) {
                settings.push((name, value));
            }
        }
        source_identity(victory_shader::TRANSLATOR_SOURCE_HASH, limit, rays, &settings)
    })
}

fn source_identity(
    source: &str,
    limit: u32,
    rays: bool,
    settings: &[(&str, std::ffi::OsString)],
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"VIctory-NVIDIA shader source identity v1");
    h.update(MAGIC);
    let mut field = |bytes: &[u8]| {
        h.update(&(bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    };
    field(source.as_bytes());
    field(&limit.to_le_bytes());
    field(&[rays.into()]);
    for (name, value) in settings {
        field(name.as_bytes());
        field(value.as_encoded_bytes());
    }
    *h.finalize().as_bytes()
}

/// The translation of a console shader container for a host that runs
/// `target`'s wave sizes, from the cache when there is one.
pub(crate) fn host_shader(container: &[u8], target: Target) -> Result<HostShader, String> {
    let translate =
        |c: &[u8]| victory_shader::emit::translate_for(c, target).map_err(|e| e.to_string());
    match CACHE.get() {
        Some(cache) => {
            let mut key = key(build(), target);
            // A container a diagnostic patch may apply to has entries of its
            // own, by the patches.
            let patches = victory_shader::native::diagnostic_patches();
            if patches.iter().any(|p| p.0 == container.len()) {
                let mut h = blake3::Hasher::new();
                h.update(&key);
                h.update(format!("{patches:?}").as_bytes());
                key = *h.finalize().as_bytes();
            }
            cached(cache, &key, container, translate)
        }
        None => translate(container),
    }
}

/// The part of an entry's key before the container: the build, and the
/// host's wave sizes (a translation for one host's waves is not another's).
fn key(build: &[u8; 32], target: Target) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(build);
    h.update(&target.key());
    let wave64 = victory_shader::emit::wave64_policy_key();
    if wave64 != 0 {
        h.update(b"wave64 policy");
        h.update(&[wave64]);
    }
    // Where heap indices are cut is part of the translation.
    h.update(&victory_shader::emit::heap_clamp().to_le_bytes());
    // The clean-up passes are part of the translation; none leaves the key
    // as it was before there were any.
    let passes = victory_shader::emit::shader_opt();
    if passes != 0 {
        h.update(b"clean-up passes");
        h.update(&passes.to_le_bytes());
    }
    if victory_shader::emit::constants_raw() {
        h.update(b"VICTORY_PORT_CBV=1");
    }
    *h.finalize().as_bytes()
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let shader = (self.translate)(&input[32..]).map_err(anyhow::Error::msg)?;
        Ok(encode(&shader))
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
    };
    for _ in 0..2 {
        let path = cache
            .get_or_convert(&converter, &input)
            .map_err(|e| format!("{e:#}"))?;
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let (counter, what) = if converter.ran.load(Ordering::Relaxed) {
                (&MISSES, "translated and cached")
            } else {
                (&HITS, "from the cache")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return Ok(shader);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    (converter.translate)(container)
}

const MAGIC: &[u8; 4] = b"VSH2";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, u32::from(s.stride_table));
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let stride_table = r.word()? != 0;
    r.0.is_empty().then_some(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        stride_table,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![4],
            stride_table: true,
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Failures are not stored.
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        assert!(cached(&cache, &build, b"other", fail).is_err());
    }

    #[test]
    fn source_identity_survives_unrelated_executable_rebuilds() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path().join("shaders")).unwrap();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        let first = source_identity("translator-a", 100, false, &[]);
        assert_eq!(cached(&cache, &first, b"synthetic", translate), Ok(shader()));
        // Change executable bytes without changing any translation input.
        let exe = dir.path().join("synthetic.exe");
        std::fs::write(&exe, b"build one").unwrap();
        std::fs::write(&exe, b"build two").unwrap();
        let second = source_identity("translator-a", 100, false, &[]);
        assert_eq!(first, second);
        assert_eq!(cached(&cache, &second, b"synthetic", translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        for changed in [
            source_identity("translator-b", 100, false, &[]),
            source_identity("translator-a", 101, false, &[]),
            source_identity("translator-a", 100, true, &[]),
            source_identity("translator-a", 100, false, &[("VICTORY_BUILT_BOUNDS", "0".into())]),
        ] {
            assert_ne!(first, changed);
            assert_eq!(cached(&cache, &changed, b"synthetic", translate), Ok(shader()));
        }
        assert_eq!(runs.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn keys_tell_hosts_wave_sizes_apart() {
        let build = [7u8; 32];
        let nvidia = Target {
            lanes_min: 32,
            lanes_max: 32,
        };
        let rdna = Target {
            lanes_min: 32,
            lanes_max: 64,
        };
        assert_eq!(key(&build, nvidia), key(&build, nvidia));
        assert_ne!(key(&build, nvidia), key(&build, rdna));
        assert_ne!(key(&build, nvidia), key(&[8; 32], nvidia));
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
        let mut foreign = bytes.clone();
        foreign[..4].copy_from_slice(b"VSH1");
        assert_eq!(decode(&foreign), None);
    }
}
