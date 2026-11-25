//! Link settings for the Windows executable.

fn main() {
    // Threads created without a stack size take the executable's default.
    // Reserve 4 MiB rather than MSVC's 1 MiB so host drivers compiling large
    // shaders do not overflow, without burning ~555 MiB of `< 4 GiB` virtual
    // address space across 37+ threads like 16 MiB did (blocks 32-bit title
    // reservations on Windows + NVIDIA).
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/STACK:4194304");
    }
}
