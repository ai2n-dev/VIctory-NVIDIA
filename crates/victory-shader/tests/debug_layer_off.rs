//! Without `VICTORY_D3D12_DEBUG` a refusal is the bare error, as before.
//! Software adapter, and a shader that is not one (no title data).

#![cfg(all(windows, feature = "host"))]

use victory_shader::emit::host::Host;
use windows::Win32::Graphics::Direct3D12::*;

#[test]
fn a_refused_pipeline_carries_no_debug_layer_text() {
    // SAFETY: the only test of this process.
    unsafe { std::env::remove_var("VICTORY_D3D12_DEBUG") };
    let host = Host::new(true).expect("WARP");
    let desc = D3D12_ROOT_SIGNATURE_DESC::default();
    let mut blob = None;
    // SAFETY: the description is empty and the blob outlives the call.
    let root = unsafe {
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).unwrap();
        let blob = blob.unwrap();
        let bytes = std::slice::from_raw_parts(
            blob.GetBufferPointer().cast::<u8>(),
            blob.GetBufferSize(),
        );
        host.device.CreateRootSignature(0, bytes).unwrap()
    };
    let error = host
        .compute_pipeline_with(&root, b"not a shader")
        .expect_err("not a shader");
    assert!(!error.contains("debug layer"), "{error}");
}
