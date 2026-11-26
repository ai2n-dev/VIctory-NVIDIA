//! Mesh draws launched by the card: an amplification shader of ours
//! (`launch.hlsl`, embedded as `launch_code.rs`) in front of every translated
//! mesh shader (`VICTORY_MESH_LAUNCH`, on by default; `0`: the clamp of
//! `crate::indirect` before each record, as before).
//!
//! Here the record is read by the draw itself: the pipeline's amplification
//! stage loads its three words where the title's passes wrote them, bounds
//! them as the clamp does (65,535 groups, one-dimensional), charges the
//! list's budget as the clamp does (a record that does not fit runs empty),
//! and launches the mesh groups. The title's own barrier orders its passes
//! before the draw, as on the console; nothing of ours runs between draws.
//! A mesh dispatch the CPU knows (`DispatchMesh`) gives its count as a root
//! constant and is launched the same way, unbounded by the budget as
//! before.
//!
//! An indirect call with a count buffer or more than one record keeps the
//! clamp and runs on the same pipeline made without the amplification
//! stage (`HostPipeline::launch::plain`): the translated mesh shader declares
//! the payload either way (`victory_shader::emit::MESH_PAYLOAD_BYTES`), which
//! the driver takes with or without an amplification shader.

use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::host::HostList;

#[path = "launch_code.rs"]
mod launch_code;

/// The amplification shader's code (see `launch.hlsl`).
const CODE: &[u8] = launch_code::CODE;

/// The root constant that has the card read the record.
pub(crate) const DIRECT_NONE: u32 = u32::MAX;

/// Where the amplification shader binds: `b0`, `t0`, `u0` in this space.
const SPACE: u32 = victory_shader::emit::LAUNCH_SPACE;

/// Whether mesh pipelines get the amplification stage and indirect mesh
/// records are launched by it (`VICTORY_MESH_LAUNCH=0`: not).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MESH_LAUNCH").is_none_or(|v| v != "0"))
}

/// The amplification stage's code.
pub(crate) fn code() -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: CODE.as_ptr().cast(),
        BytecodeLength: CODE.len(),
    }
}

/// The bytes of [`code`], for the pipeline library's names.
pub(crate) fn bytes() -> &'static [u8] {
    CODE
}

/// The three root parameters the stage reads, in order: the direct group
/// count (one constant), the record (a root SRV), the list's budget (a
/// root UAV). Five root words.
pub(crate) fn params() -> [D3D12_ROOT_PARAMETER1; 3] {
    let descriptor = |kind| D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: 0,
                RegisterSpace: SPACE,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
    };
    [
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: SPACE,
                    Num32BitValues: 1,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
        },
        descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV),
        descriptor(D3D12_ROOT_PARAMETER_TYPE_UAV),
    ]
}

/// Mesh draws launched since the last line: records read by the card, and
/// dispatches whose count the CPU gave.
static LAUNCHED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Records the launch of one mesh draw, the mesh pipeline with its stage
/// bound and the stage's parameters at `roots`: `record` the record's
/// address on the card (the card reads it), or the group count the CPU
/// knows. False if the list has no budget or does not support mesh shaders.
pub(crate) fn record(list: &HostList, roots: [u32; 3], what: Launch) -> bool {
    let Ok(l6) = list.list.cast::<ID3D12GraphicsCommandList6>() else {
        return false;
    };
    let Some(budget) = list.budget() else {
        return false;
    };
    let (direct, record, budget) = match what {
        Launch::Record(address) => (DIRECT_NONE, address, budget),
        // The stage reads neither the record nor the budget; valid addresses
        // keep D3D12 root descriptor validation clean.
        Launch::Direct(x) => (x, budget, budget),
    };
    // SAFETY: an open list with the mesh pipeline and its root signature
    // set; the record and the budget are live buffers the list keeps.
    unsafe {
        list.list.SetGraphicsRoot32BitConstant(roots[0], direct, 0);
        list.list.SetGraphicsRootShaderResourceView(roots[1], record);
        list.list.SetGraphicsRootUnorderedAccessView(roots[2], budget);
        l6.DispatchMesh(1, 1, 1);
    }
    LAUNCHED[usize::from(matches!(what, Launch::Direct(_)))].fetch_add(1, Ordering::Relaxed);
    true
}

/// What one launch reads.
#[derive(Clone, Copy)]
pub(crate) enum Launch {
    /// The record at this address on the card.
    Record(u64),
    /// A group count known on the CPU.
    Direct(u32),
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    let [records, direct] = [0, 1].map(|i| LAUNCHED[i].swap(0, Ordering::Relaxed));
    if records + direct > 0 {
        tracing::info!(
            "mesh draws launched by the card per frame: {:.0} records read where the title wrote them, {:.0} counts \
             known on the CPU (VICTORY_MESH_LAUNCH=0: each record clamped by one)",
            records as f64 / frames,
            direct as f64 / frames
        );
    }
}
