//! Source-based translation identity, adapted from DoubleGhosty/cool_fox
//! (GPL-3.0). Use BLAKE3, framed inputs and fail closed on unreadable files.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read translator source directory") {
        let entry = entry.expect("read translator source entry");
        let path = entry.path();
        if entry.file_type().expect("translator source type").is_dir() {
            files(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn field(hash: &mut blake3::Hasher, bytes: &[u8]) {
    hash.update(&(bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=../../Cargo.lock");
    println!("cargo:rerun-if-changed=../../Cargo.toml");
    let mut paths = Vec::new();
    files(&root.join("src"), &mut paths);
    paths.extend([root.join("build.rs"), root.join("Cargo.toml"),
        root.join("../../Cargo.lock"), root.join("../../Cargo.toml")]);
    paths.sort();
    let mut hash = blake3::Hasher::new();
    field(&mut hash, b"victory-shader source v1");
    let mut switches = BTreeSet::new();
    for path in paths {
        let rel = path.strip_prefix(&root).expect("relative source path")
            .to_string_lossy().replace('\\', "/");
        let bytes = std::fs::read(&path).expect("read translator fingerprint input");
        field(&mut hash, rel.as_bytes());
        field(&mut hash, &bytes);
        if path.extension().is_some_and(|ext| ext == "rs") && rel.starts_with("src/") {
            let source = String::from_utf8_lossy(&bytes);
            let mut rest = source.as_ref();
            while let Some(at) = rest.find("\"VICTORY_") {
                let tail = &rest[at + 1..];
                let end = tail.find(|c: char| {
                    !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
                }).unwrap_or(tail.len());
                if tail[end..].starts_with('"') {
                    switches.insert(tail[..end].to_owned());
                }
                rest = &tail[end..];
            }
        }
    }
    // Different compile-time features or targets may change emitted DXIL.
    let mut config: Vec<_> = std::env::vars().filter(|(name, _)| {
        name.starts_with("CARGO_FEATURE_") || name.starts_with("CARGO_CFG_")
            || name == "TARGET" || name == "CARGO_ENCODED_RUSTFLAGS"
    }).collect();
    config.sort();
    for (name, value) in config {
        field(&mut hash, name.as_bytes());
        field(&mut hash, value.as_bytes());
    }
    println!("cargo:rustc-env=VICTORY_SHADER_SOURCE_HASH={}", hash.finalize().to_hex());
    println!("cargo:rustc-env=VICTORY_SHADER_SWITCHES={}",
        switches.into_iter().collect::<Vec<_>>().join(","));
}
