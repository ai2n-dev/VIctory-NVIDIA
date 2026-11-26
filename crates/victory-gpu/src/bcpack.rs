//! Opt-in GPU BCPack decompression (`VICTORY_BCPACK_GPU=1`, off by default).
//!
//! Uses the clean-room SM 6.6 Wave32 HLSL compute decoders from
//! `POMXARK/bcpack-gpu-sdk` (MIT OR Apache-2.0) compiled on first use via
//! `dxcompiler.dll`, with CPU fallback to `victory_formats::bcpack::decode`.

use std::ffi::c_void;
use std::mem::{ManuallyDrop, size_of};
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use victory_formats::bcpack::{GpuTables, Tables};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, PCWSTR, s};

const BCN_RANS_HLSL: &str = include_str!("bcpack/bcpack_bcn_rans.hlsl");
const BCN_REBUILD_HLSL: &str = include_str!("bcpack/bcpack_bcn_rebuild.hlsl");
const BC7_RANS_HLSL: &str = include_str!("bcpack/bcpack_bc7_rans.hlsl");
const BC7_REBUILD_HLSL: &str = include_str!("bcpack/bcpack_bc7_rebuild.hlsl");
const BC6H_RANS_HLSL: &str = include_str!("bcpack/bcpack_bc6h_rans.hlsl");
const BC6H_REBUILD_HLSL: &str = include_str!("bcpack/bcpack_bc6h_rebuild.hlsl");

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BCPACK_GPU").is_some_and(|v| v == "1"))
}

fn min_gpu_bytes() -> usize {
    static MIN: OnceLock<usize> = OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("VICTORY_BCPACK_GPU_MIN_BYTES")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(64 << 10)
    })
}

/// Attempts GPU BCPack decompression into `dst` when `VICTORY_BCPACK_GPU=1`
/// and `dst.len()` reaches the threshold. Returns `None` if disabled, too
/// small, or no compatible `dxcompiler.dll` / WaveOps GPU context exists.
pub fn try_decode(
    bc: u8,
    src: &[u8],
    dst: &mut [u8],
    tables: &Tables,
) -> Option<Result<(), &'static str>> {
    if !enabled() || dst.len() < min_gpu_bytes() || src.is_empty() {
        return None;
    }
    static ENGINE: OnceLock<Option<Mutex<BcpackEngine>>> = OnceLock::new();
    let lock = ENGINE
        .get_or_init(|| {
            let host = crate::host::host()?;
            match BcpackEngine::new(&host.device, tables) {
                Ok(engine) => {
                    tracing::info!("BCPack GPU compute decoder ready (SM 6.6 Wave32)");
                    Some(Mutex::new(engine))
                }
                Err(err) => {
                    tracing::warn!(error = err, "BCPack GPU decoder unavailable; using CPU fallback");
                    None
                }
            }
        })
        .as_ref()?;
    let mut engine = lock.lock().unwrap_or_else(|e| e.into_inner());
    Some(engine.decode(bc, src, dst))
}

// --- Dynamic DXC compiler (dxcompiler.dll / IDxcCompiler3) ---

#[repr(C)]
struct Guid(u32, u16, u16, [u8; 8]);

const CLSID_DXC_COMPILER: Guid = Guid(
    0x73e2_2d93,
    0xe6ce,
    0x47f3,
    [0xb5, 0xbf, 0xf0, 0x66, 0x4f, 0x39, 0xc1, 0xb0],
);
const IID_IDXC_COMPILER3: Guid = Guid(
    0x228b_4687,
    0x5a6a,
    0x4730,
    [0x90, 0x0c, 0x97, 0x02, 0xb2, 0x20, 0x3f, 0x54],
);
const IID_IDXC_RESULT: Guid = Guid(
    0x5834_6cda,
    0xdde7,
    0x4497,
    [0x94, 0x61, 0x6f, 0x87, 0xaf, 0x5e, 0x06, 0x59],
);

#[repr(C)]
struct DxcBuffer {
    ptr: *const c_void,
    size: usize,
    encoding: u32,
}

type ComObject = *mut *const usize;
type CreateInstance =
    unsafe extern "system" fn(*const Guid, *const Guid, *mut *mut c_void) -> i32;

struct ComGuard(ComObject);

impl Drop for ComGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: COM `IUnknown::Release` is slot 2.
            unsafe {
                let release: unsafe extern "system" fn(ComObject) -> u32 =
                    std::mem::transmute_copy(&*(*self.0).add(2));
                release(self.0);
            }
        }
    }
}

unsafe fn com_method<F: Copy>(obj: ComObject, slot: usize) -> F {
    // SAFETY: caller matches slot to the COM vtable layout.
    unsafe { std::mem::transmute_copy(&*(*obj).add(slot)) }
}

fn dxc_dll_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("VICTORY_DXC_DLL") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    let kits = PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut versions: Vec<PathBuf> = std::fs::read_dir(&kits)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|e| e.path().join(r"x64\dxcompiler.dll"))
        .filter(|p| p.exists())
        .collect();
    versions.sort();
    if let Some(found) = versions.pop() {
        return Some(found);
    }
    let mut seeds = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        seeds.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        seeds.push(parent.to_path_buf());
    }
    for seed in seeds {
        for ancestor in seed.ancestors().take(6) {
            let candidate = ancestor.join(r"PortableBuildTools\xwin\bin\x64\dxcompiler.dll");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

struct DxcCompiler {
    compiler: ComGuard,
}

impl DxcCompiler {
    fn load() -> Result<Self, &'static str> {
        let path = dxc_dll_path();
        let wide: Vec<u16> = path
            .as_ref()
            .map_or_else(|| "dxcompiler.dll".into(), |p| p.as_os_str().to_owned())
            .to_string_lossy()
            .encode_utf16()
            .chain([0])
            .collect();
        // SAFETY: NUL-terminated path to `dxcompiler.dll`.
        let module = unsafe { LoadLibraryW(PCWSTR(wide.as_ptr())) }
            .map_err(|_| "dxcompiler.dll not found")?;
        // SAFETY: exported factory function on `dxcompiler.dll`.
        let proc = unsafe { GetProcAddress(module, s!("DxcCreateInstance")) }
            .ok_or("DxcCreateInstance missing in dxcompiler.dll")?;
        // SAFETY: matches `DxcCreateInstance` signature.
        let create: CreateInstance = unsafe { std::mem::transmute(proc) };
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid CLSID/IID pair.
        let hr = unsafe { create(&CLSID_DXC_COMPILER, &IID_IDXC_COMPILER3, &mut raw) };
        if hr < 0 || raw.is_null() {
            return Err("DxcCreateInstance(IDxcCompiler3) failed");
        }
        Ok(Self {
            compiler: ComGuard(raw.cast()),
        })
    }

    fn compile_cs_6_6(&self, source: &str) -> Result<Vec<u8>, String> {
        let w = |s: &str| s.encode_utf16().chain([0]).collect::<Vec<u16>>();
        let args_w = [
            w("-T"),
            w("cs_6_6"),
            w("-E"),
            w("CSMain"),
            w("-HV"),
            w("2021"),
            w("-O3"),
        ];
        let args: Vec<*const u16> = args_w.iter().map(|a| a.as_ptr()).collect();
        let buf = DxcBuffer {
            ptr: source.as_ptr().cast(),
            size: source.len(),
            encoding: 65001, // CP_UTF8
        };
        let mut res_raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `IDxcCompiler3::Compile` is slot 3.
        let hr = unsafe {
            let compile: unsafe extern "system" fn(
                ComObject,
                *const DxcBuffer,
                *const *const u16,
                u32,
                *mut c_void,
                *const Guid,
                *mut *mut c_void,
            ) -> i32 = com_method(self.compiler.0, 3);
            compile(
                self.compiler.0,
                &buf,
                args.as_ptr(),
                args.len() as u32,
                std::ptr::null_mut(),
                &IID_IDXC_RESULT,
                &mut res_raw,
            )
        };
        if hr < 0 || res_raw.is_null() {
            return Err(format!("IDxcCompiler3::Compile failed ({hr:#x})"));
        }
        let res = ComGuard(res_raw.cast());
        let mut status = 0i32;
        // SAFETY: `IDxcOperationResult::GetStatus` is slot 3.
        let hr = unsafe {
            let get_status: unsafe extern "system" fn(ComObject, *mut i32) -> i32 =
                com_method(res.0, 3);
            get_status(res.0, &mut status)
        };
        if hr < 0 || status < 0 {
            let mut err_raw: *mut c_void = std::ptr::null_mut();
            // SAFETY: `IDxcOperationResult::GetErrorBuffer` is slot 5.
            let msg = unsafe {
                let get_err: unsafe extern "system" fn(ComObject, *mut *mut c_void) -> i32 =
                    com_method(res.0, 5);
                if get_err(res.0, &mut err_raw) >= 0 && !err_raw.is_null() {
                    let err = ComGuard(err_raw.cast());
                    let ptr: unsafe extern "system" fn(ComObject) -> *const u8 =
                        com_method(err.0, 3);
                    let len: unsafe extern "system" fn(ComObject) -> usize = com_method(err.0, 4);
                    let p = ptr(err.0);
                    let n = len(err.0);
                    if !p.is_null() && n > 0 {
                        String::from_utf8_lossy(std::slice::from_raw_parts(p, n)).into_owned()
                    } else {
                        format!("status {status:#x}")
                    }
                } else {
                    format!("status {status:#x}")
                }
            };
            return Err(msg);
        }
        let mut blob_raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `IDxcOperationResult::GetResult` is slot 4.
        let hr = unsafe {
            let get_res: unsafe extern "system" fn(ComObject, *mut *mut c_void) -> i32 =
                com_method(res.0, 4);
            get_res(res.0, &mut blob_raw)
        };
        if hr < 0 || blob_raw.is_null() {
            return Err(format!("IDxcResult::GetResult failed ({hr:#x})"));
        }
        let blob = ComGuard(blob_raw.cast());
        // SAFETY: `IDxcBlob` GetBufferPointer (slot 3) and GetBufferSize (slot 4).
        unsafe {
            let ptr: unsafe extern "system" fn(ComObject) -> *const u8 = com_method(blob.0, 3);
            let len: unsafe extern "system" fn(ComObject) -> usize = com_method(blob.0, 4);
            let p = ptr(blob.0);
            let n = len(blob.0);
            if p.is_null() || n == 0 {
                return Err("empty DXIL blob".into());
            }
            Ok(std::slice::from_raw_parts(p, n).to_vec())
        }
    }
}

// --- D3D12 compute pipelines & reusable scratch ---

#[repr(C)]
#[derive(Clone, Copy)]
struct RansChunkDesc {
    stream_offset: u32,
    stream_len: u32,
    count: u32,
    out_row_offset: u32,
    mode: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Rans16ChunkDesc {
    stream_offset: u32,
    stream_len: u32,
    blocks: u32,
    out_block_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RebuildChunkDesc {
    in_row_offset: u32,
    blocks: u32,
    out_offset: u32,
    mode_or_pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ChunkStatus {
    code: u32,
    consumed: u32,
    cycles: u32,
    blocks_written: u32,
}

struct BcpackEngine {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    fence_value: u64,
    event: OwnedHandle,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,

    rans_sig: ID3D12RootSignature,
    bcn_rans_pso: ID3D12PipelineState,
    bc7_rans_pso: ID3D12PipelineState,
    bc6_rans_pso: ID3D12PipelineState,

    bcn_rebuild_sig: ID3D12RootSignature,
    bcn_rebuild_pso: ID3D12PipelineState,
    bc7_rebuild_sig: ID3D12RootSignature,
    bc7_rebuild_pso: ID3D12PipelineState,
    bc6_rebuild_sig: ID3D12RootSignature,
    bc6_rebuild_pso: ID3D12PipelineState,

    models_buf: ID3D12Resource,
    bcn_pairs_buf: ID3D12Resource,
    bcn_quads_buf: ID3D12Resource,
    bcn_dict_buf: ID3D12Resource,
    bc7_ctrl_buf: ID3D12Resource,
    bc7_dict_buf: ID3D12Resource,
    bc6_dict_buf: ID3D12Resource,

    streams: GrowBuf,
    chunks: GrowBuf,
    rebuild_chunks: GrowBuf,
    syms_default: Option<(ID3D12Resource, u64)>,
    hist_default: Option<(ID3D12Resource, u64)>,
    out_buf: ReadbackPair,
    status_buf: ReadbackPair,
}

#[derive(Default)]
struct GrowBuf {
    upload: Option<(ID3D12Resource, *mut u8, u64)>,
    default: Option<(ID3D12Resource, u64)>,
}

#[derive(Default)]
struct ReadbackPair {
    default: Option<(ID3D12Resource, u64)>,
    readback: Option<(ID3D12Resource, *const u8, u64)>,
}

// SAFETY: accessed exclusively under `ENGINE` mutex; D3D12 objects are free-threaded.
unsafe impl Send for BcpackEngine {}

impl Drop for BcpackEngine {
    fn drop(&mut self) {
        self.wait_idle();
    }
}

impl BcpackEngine {
    fn new(device: &ID3D12Device, tables: &Tables) -> Result<Self, &'static str> {
        let mut opts1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        // SAFETY: plain feature query on live device.
        let wave_ok = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut opts1).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS1>() as u32,
            )
        }
        .is_ok()
            && opts1.WaveOps.as_bool()
            && opts1.WaveLaneCountMin <= 32
            && opts1.WaveLaneCountMax >= 32;
        if !wave_ok {
            return Err("device lacks Wave32 compute support");
        }

        let compiler = DxcCompiler::load()?;
        let compile = |src: &str, name: &str| -> Result<Vec<u8>, &'static str> {
            compiler.compile_cs_6_6(src).map_err(|err| {
                tracing::warn!(shader = name, %err, "DXC BCPack shader compilation failed");
                "DXC compilation failed"
            })
        };
        let bcn_rans_dxil = compile(BCN_RANS_HLSL, "bcpack_bcn_rans")?;
        let bcn_rebuild_dxil = compile(BCN_REBUILD_HLSL, "bcpack_bcn_rebuild")?;
        let bc7_rans_dxil = compile(BC7_RANS_HLSL, "bcpack_bc7_rans")?;
        let bc7_rebuild_dxil = compile(BC7_REBUILD_HLSL, "bcpack_bc7_rebuild")?;
        let bc6_rans_dxil = compile(BC6H_RANS_HLSL, "bcpack_bc6h_rans")?;
        let bc6_rebuild_dxil = compile(BC6H_REBUILD_HLSL, "bcpack_bc6h_rebuild")?;

        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
            ..Default::default()
        };
        // SAFETY: plain queue, fence, event, allocator, list creation.
        let (queue, fence, event, alloc, list) = unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&queue_desc)
                .map_err(|_| "CreateCommandQueue COMPUTE failed")?;
            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(|_| "CreateFence failed")?;
            let event = OwnedHandle::from_raw_handle(
                CreateEventW(None, false, false, None)
                    .map_err(|_| "CreateEventW failed")?
                    .0,
            );
            let alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
                .map_err(|_| "CreateCommandAllocator COMPUTE failed")?;
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &alloc, None)
                .map_err(|_| "CreateCommandList COMPUTE failed")?;
            list.Close().map_err(|_| "Close COMPUTE list failed")?;
            (queue, fence, event, alloc, list)
        };

        let rans_params = [srv_param(0), srv_param(1), srv_param(2), uav_param(0), uav_param(1)];
        let rans_sig = create_root_sig(device, &rans_params)?;
        let bcn_rans_pso = create_pso(device, &rans_sig, &bcn_rans_dxil)?;
        let bc7_rans_pso = create_pso(device, &rans_sig, &bc7_rans_dxil)?;
        let bc6_rans_pso = create_pso(device, &rans_sig, &bc6_rans_dxil)?;

        let bcn_rebuild_sig = create_root_sig(
            device,
            &[
                cb_param(0, 4),
                srv_param(0),
                srv_param(1),
                srv_param(2),
                srv_param(3),
                srv_param(4),
                uav_param(0),
                uav_param(1),
            ],
        )?;
        let bcn_rebuild_pso = create_pso(device, &bcn_rebuild_sig, &bcn_rebuild_dxil)?;

        let bc7_rebuild_sig = create_root_sig(
            device,
            &[
                cb_param(0, 4),
                srv_param(0),
                srv_param(1),
                srv_param(2),
                srv_param(3),
                uav_param(0),
                uav_param(1),
            ],
        )?;
        let bc7_rebuild_pso = create_pso(device, &bc7_rebuild_sig, &bc7_rebuild_dxil)?;

        let bc6_rebuild_sig = create_root_sig(
            device,
            &[
                cb_param(0, 4),
                srv_param(0),
                srv_param(1),
                srv_param(2),
                uav_param(0),
                uav_param(1),
            ],
        )?;
        let bc6_rebuild_pso = create_pso(device, &bc6_rebuild_sig, &bc6_rebuild_dxil)?;

        let gpu: GpuTables = tables.gpu_tables();
        let mut engine = Self {
            device: device.clone(),
            queue,
            fence,
            fence_value: 0,
            event,
            alloc,
            list,
            rans_sig,
            bcn_rans_pso,
            bc7_rans_pso,
            bc6_rans_pso,
            bcn_rebuild_sig,
            bcn_rebuild_pso,
            bc7_rebuild_sig,
            bc7_rebuild_pso,
            bc6_rebuild_sig,
            bc6_rebuild_pso,
            models_buf: create_buf(device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON)?,
            bcn_pairs_buf: create_buf(device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON)?,
            bcn_quads_buf: create_buf(device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON)?,
            bcn_dict_buf: create_buf(device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON)?,
            bc7_ctrl_buf: create_buf(device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON)?,
            bc7_dict_buf: create_buf(device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON)?,
            bc6_dict_buf: create_buf(device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON)?,
            streams: GrowBuf::default(),
            chunks: GrowBuf::default(),
            rebuild_chunks: GrowBuf::default(),
            syms_default: None,
            hist_default: None,
            out_buf: ReadbackPair::default(),
            status_buf: ReadbackPair::default(),
        };
        engine.models_buf = engine.upload_static_srv(&gpu.models)?;
        engine.bcn_pairs_buf = engine.upload_static_srv(&gpu.bcn_pairs)?;
        engine.bcn_quads_buf = engine.upload_static_srv(&gpu.bcn_quads)?;
        engine.bcn_dict_buf = engine.upload_static_srv(&gpu.bcn_dict)?;
        engine.bc7_ctrl_buf = engine.upload_static_srv(&gpu.bc7_controls)?;
        engine.bc7_dict_buf = engine.upload_static_srv(&gpu.bc7_dict)?;
        engine.bc6_dict_buf = engine.upload_static_srv(&gpu.bc6_dict)?;
        Ok(engine)
    }

    fn upload_static_srv<T: Copy>(&mut self, data: &[T]) -> Result<ID3D12Resource, &'static str> {
        let bytes = std::mem::size_of_val(data).max(256) as u64;
        let (up, ptr) = create_mapped_buf(&self.device, D3D12_HEAP_TYPE_UPLOAD, bytes, D3D12_RESOURCE_STATE_GENERIC_READ)?;
        // SAFETY: `up` is mapped for `bytes`; `data` size <= `bytes`.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), ptr, std::mem::size_of_val(data));
        }
        let dst = create_buf(
            &self.device,
            D3D12_HEAP_TYPE_DEFAULT,
            bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        // SAFETY: allocator & list are idle; both resources are live.
        unsafe {
            self.alloc.Reset().map_err(|_| "alloc Reset failed")?;
            self.list.Reset(&self.alloc, None).map_err(|_| "list Reset failed")?;
            self.list.CopyBufferRegion(&dst, 0, &up, 0, bytes);
            let barrier = [transition(&dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)];
            self.list.ResourceBarrier(&barrier);
            drop_barriers(barrier);
            self.list.Close().map_err(|_| "list Close failed")?;
            let lists = [Some(self.list.cast::<ID3D12CommandList>().map_err(|_| "cast failed")?)];
            self.queue.ExecuteCommandLists(&lists);
        }
        self.signal_and_wait()?;
        Ok(dst)
    }

    fn signal_and_wait(&mut self) -> Result<(), &'static str> {
        self.fence_value += 1;
        let target = self.fence_value;
        // SAFETY: queue, fence, and event belong to `self`.
        unsafe {
            self.queue.Signal(&self.fence, target).map_err(|_| "Signal failed")?;
            if self.fence.GetCompletedValue() < target {
                self.fence
                    .SetEventOnCompletion(target, HANDLE(self.event.as_raw_handle()))
                    .map_err(|_| "SetEventOnCompletion failed")?;
                WaitForSingleObject(HANDLE(self.event.as_raw_handle()), INFINITE);
            }
        }
        Ok(())
    }

    fn wait_idle(&self) {
        // SAFETY: plain fence query & event wait on live handles.
        unsafe {
            if self.fence_value != 0 && self.fence.GetCompletedValue() < self.fence_value {
                let _ = self
                    .fence
                    .SetEventOnCompletion(self.fence_value, HANDLE(self.event.as_raw_handle()));
                WaitForSingleObject(HANDLE(self.event.as_raw_handle()), 5_000);
            }
        }
    }

    fn decode(&mut self, bc: u8, src: &[u8], dst: &mut [u8]) -> Result<(), &'static str> {
        let (block_bytes, rows_per_block, hist_words, sym_words) = match bc {
            1 => (8usize, 1usize, 6usize, 6usize),
            3 => (16, 2, 6, 6),
            4 => (8, 1, 6, 6),
            5 => (16, 2, 6, 6),
            6 => (16, 2, 15, 8),
            7 => (16, 2, 11, 8),
            _ => return Err("unsupported BC format"),
        };
        if dst.is_empty() || !dst.len().is_multiple_of(block_bytes) {
            return Err("destination is not a whole number of blocks");
        }
        let blocks = dst.len() / block_bytes;
        let sym_rows = blocks * rows_per_block;
        let stream_bytes = src.len().next_multiple_of(4) + 256;
        let chunk_count = 2usize; // 1 real chunk + 1 inactive Wave32 pair chunk

        let (stream_up, stream_up_ptr, stream_def) = self.streams.ensure(&self.device, stream_bytes as u64)?;
        // SAFETY: `stream_up_ptr` covers `stream_bytes` >= `src.len() + 256`.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), stream_up_ptr, src.len());
            std::ptr::write_bytes(stream_up_ptr.add(src.len()), 0, stream_bytes - src.len());
        }

        let syms_bytes = ((sym_rows * sym_words * 4).max(256)) as u64;
        let syms_def = ensure_uav(&self.device, &mut self.syms_default, syms_bytes)?;
        let hist_bytes = ((256 * hist_words * 4).max(256)) as u64;
        let hist_def = ensure_uav(&self.device, &mut self.hist_default, hist_bytes)?;
        let status_bytes = (chunk_count * size_of::<ChunkStatus>()).max(256) as u64;
        let (status_def, status_rb, status_rb_ptr) = self.status_buf.ensure(&self.device, status_bytes)?;
        let out_bytes = dst.len().max(256) as u64;
        let (out_def, out_rb, out_rb_ptr) = self.out_buf.ensure(&self.device, out_bytes)?;
        let rb_bytes = (chunk_count * size_of::<RebuildChunkDesc>()).max(256) as u64;
        let (rb_up, rb_up_ptr, rb_def) = self.rebuild_chunks.ensure(&self.device, rb_bytes)?;

        let is_bcn = matches!(bc, 1 | 3 | 4 | 5);
        let (rans_up, rans_up_ptr, rans_def, rc_bytes) = if is_bcn {
            let rc_bytes = (chunk_count * size_of::<RansChunkDesc>()).max(256) as u64;
            let (u, p, d) = self.chunks.ensure(&self.device, rc_bytes)?;
            let mode = match bc {
                3 => 0u32,
                4 => 1,
                5 => 2,
                _ => 3,
            };
            let descs = [
                RansChunkDesc {
                    stream_offset: 0,
                    stream_len: src.len() as u32,
                    count: sym_rows as u32,
                    out_row_offset: 0,
                    mode,
                    _pad: [0; 3],
                },
                RansChunkDesc {
                    stream_offset: 0,
                    stream_len: 0,
                    count: 0,
                    out_row_offset: 0,
                    mode,
                    _pad: [0; 3],
                },
            ];
            // SAFETY: `p` covers `rc_bytes` >= 2 descriptors.
            unsafe { std::ptr::copy_nonoverlapping(descs.as_ptr().cast::<u8>(), p, size_of_val(&descs)) };
            let r_descs = [RebuildChunkDesc {
                in_row_offset: 0,
                blocks: blocks as u32,
                out_offset: 0,
                mode_or_pad: mode,
            }];
            // SAFETY: `rb_up_ptr` covers `rb_bytes`.
            unsafe { std::ptr::copy_nonoverlapping(r_descs.as_ptr().cast::<u8>(), rb_up_ptr, size_of_val(&r_descs)) };
            (u, p, d, rc_bytes)
        } else {
            let rc_bytes = (chunk_count * size_of::<Rans16ChunkDesc>()).max(256) as u64;
            let (u, p, d) = self.chunks.ensure(&self.device, rc_bytes)?;
            let descs = [
                Rans16ChunkDesc {
                    stream_offset: 0,
                    stream_len: src.len() as u32,
                    blocks: sym_rows as u32,
                    out_block_offset: 0,
                },
                Rans16ChunkDesc {
                    stream_offset: 0,
                    stream_len: 0,
                    blocks: 0,
                    out_block_offset: 0,
                },
            ];
            // SAFETY: `p` covers `rc_bytes` >= 2 descriptors.
            unsafe { std::ptr::copy_nonoverlapping(descs.as_ptr().cast::<u8>(), p, size_of_val(&descs)) };
            let r_descs = [RebuildChunkDesc {
                in_row_offset: 0,
                blocks: blocks as u32,
                out_offset: 0,
                mode_or_pad: 0,
            }];
            // SAFETY: `rb_up_ptr` covers `rb_bytes`.
            unsafe { std::ptr::copy_nonoverlapping(r_descs.as_ptr().cast::<u8>(), rb_up_ptr, size_of_val(&r_descs)) };
            (u, p, d, rc_bytes)
        };
        let _ = rans_up_ptr;

        // SAFETY: allocator & list are idle; all buffers and pipelines are live.
        unsafe {
            self.alloc.Reset().map_err(|_| "alloc Reset failed")?;
            self.list.Reset(&self.alloc, None).map_err(|_| "list Reset failed")?;

            let to_copy = [
                transition(&stream_def, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                transition(&rans_def, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                transition(&rb_def, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                transition(&syms_def, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                transition(&hist_def, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                transition(&status_def, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                transition(&out_def, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ];
            self.list.ResourceBarrier(&to_copy);
            drop_barriers(to_copy);

            self.list.CopyBufferRegion(&stream_def, 0, &stream_up, 0, stream_bytes as u64);
            self.list.CopyBufferRegion(&rans_def, 0, &rans_up, 0, rc_bytes);
            self.list.CopyBufferRegion(&rb_def, 0, &rb_up, 0, rb_bytes);

            let to_srv = [
                transition(&stream_def, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition(&rans_def, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition(&rb_def, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            ];
            self.list.ResourceBarrier(&to_srv);
            drop_barriers(to_srv);

            let rans_pso = match bc {
                1 | 3 | 4 | 5 => &self.bcn_rans_pso,
                6 => &self.bc6_rans_pso,
                _ => &self.bc7_rans_pso,
            };
            self.list.SetComputeRootSignature(&self.rans_sig);
            self.list.SetPipelineState(rans_pso);
            self.list.SetComputeRootShaderResourceView(0, self.models_buf.GetGPUVirtualAddress());
            self.list.SetComputeRootShaderResourceView(1, stream_def.GetGPUVirtualAddress());
            self.list.SetComputeRootShaderResourceView(2, rans_def.GetGPUVirtualAddress());
            self.list.SetComputeRootUnorderedAccessView(3, syms_def.GetGPUVirtualAddress());
            self.list.SetComputeRootUnorderedAccessView(4, status_def.GetGPUVirtualAddress());
            self.list.Dispatch(1, 1, 1);

            let mid = [
                transition(&syms_def, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                uav_barrier(),
            ];
            self.list.ResourceBarrier(&mid);
            drop_barriers(mid);

            let num_chunks = [1u32, 0, 0, 0];
            match bc {
                1 | 3 | 4 | 5 => {
                    self.list.SetComputeRootSignature(&self.bcn_rebuild_sig);
                    self.list.SetPipelineState(&self.bcn_rebuild_pso);
                    self.list.SetComputeRoot32BitConstants(0, 4, num_chunks.as_ptr().cast(), 0);
                    self.list.SetComputeRootShaderResourceView(1, syms_def.GetGPUVirtualAddress());
                    self.list.SetComputeRootShaderResourceView(2, rb_def.GetGPUVirtualAddress());
                    self.list.SetComputeRootShaderResourceView(3, self.bcn_pairs_buf.GetGPUVirtualAddress());
                    self.list.SetComputeRootShaderResourceView(4, self.bcn_quads_buf.GetGPUVirtualAddress());
                    self.list.SetComputeRootShaderResourceView(5, self.bcn_dict_buf.GetGPUVirtualAddress());
                    self.list.SetComputeRootUnorderedAccessView(6, out_def.GetGPUVirtualAddress());
                    self.list.SetComputeRootUnorderedAccessView(7, hist_def.GetGPUVirtualAddress());
                }
                6 => {
                    self.list.SetComputeRootSignature(&self.bc6_rebuild_sig);
                    self.list.SetPipelineState(&self.bc6_rebuild_pso);
                    self.list.SetComputeRoot32BitConstants(0, 4, num_chunks.as_ptr().cast(), 0);
                    self.list.SetComputeRootShaderResourceView(1, syms_def.GetGPUVirtualAddress());
                    self.list.SetComputeRootShaderResourceView(2, rb_def.GetGPUVirtualAddress());
                    self.list.SetComputeRootShaderResourceView(3, self.bc6_dict_buf.GetGPUVirtualAddress());
                    self.list.SetComputeRootUnorderedAccessView(4, out_def.GetGPUVirtualAddress());
                    self.list.SetComputeRootUnorderedAccessView(5, hist_def.GetGPUVirtualAddress());
                }
                _ => {
                    self.list.SetComputeRootSignature(&self.bc7_rebuild_sig);
                    self.list.SetPipelineState(&self.bc7_rebuild_pso);
                    self.list.SetComputeRoot32BitConstants(0, 4, num_chunks.as_ptr().cast(), 0);
                    self.list.SetComputeRootShaderResourceView(1, syms_def.GetGPUVirtualAddress());
                    self.list.SetComputeRootShaderResourceView(2, rb_def.GetGPUVirtualAddress());
                    self.list.SetComputeRootShaderResourceView(3, self.bc7_ctrl_buf.GetGPUVirtualAddress());
                    self.list.SetComputeRootShaderResourceView(4, self.bc7_dict_buf.GetGPUVirtualAddress());
                    self.list.SetComputeRootUnorderedAccessView(5, out_def.GetGPUVirtualAddress());
                    self.list.SetComputeRootUnorderedAccessView(6, hist_def.GetGPUVirtualAddress());
                }
            }
            self.list.Dispatch(1, 1, 1);

            let to_rb = [
                transition(&syms_def, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
                transition(&hist_def, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
                transition(&status_def, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(&out_def, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            ];
            self.list.ResourceBarrier(&to_rb);
            drop_barriers(to_rb);

            self.list.CopyBufferRegion(&out_rb, 0, &out_def, 0, dst.len() as u64);
            self.list.CopyBufferRegion(&status_rb, 0, &status_def, 0, size_of::<ChunkStatus>() as u64);

            let to_common = [
                transition(&status_def, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                transition(&out_def, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
            ];
            self.list.ResourceBarrier(&to_common);
            drop_barriers(to_common);

            self.list.Close().map_err(|_| "list Close failed")?;
            let lists = [Some(self.list.cast::<ID3D12CommandList>().map_err(|_| "cast failed")?)];
            self.queue.ExecuteCommandLists(&lists);
        }
        self.signal_and_wait()?;

        // SAFETY: compute fence completed; readback mappings cover status and `dst.len()`.
        let status = unsafe { *(status_rb_ptr.cast::<ChunkStatus>()) };
        if status.code != 0 {
            return Err("BCPack GPU rANS stream error");
        }
        // SAFETY: `out_rb_ptr` covers `dst.len()` bytes written by `CopyBufferRegion`.
        unsafe {
            std::ptr::copy_nonoverlapping(out_rb_ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }
}

impl GrowBuf {
    fn ensure(
        &mut self,
        device: &ID3D12Device,
        bytes: u64,
    ) -> Result<(ID3D12Resource, *mut u8, ID3D12Resource), &'static str> {
        let need = bytes.max(256).next_multiple_of(4096);
        let (up, ptr) = match &self.upload {
            Some((r, p, cap)) if *cap >= need => (r.clone(), *p),
            _ => {
                let (r, p) = create_mapped_buf(device, D3D12_HEAP_TYPE_UPLOAD, need, D3D12_RESOURCE_STATE_GENERIC_READ)?;
                self.upload = Some((r.clone(), p, need));
                (r, p)
            }
        };
        let def = match &self.default {
            Some((r, cap)) if *cap >= need => r.clone(),
            _ => {
                let r = create_buf(
                    device,
                    D3D12_HEAP_TYPE_DEFAULT,
                    need,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )?;
                self.default = Some((r.clone(), need));
                r
            }
        };
        Ok((up, ptr, def))
    }
}

impl ReadbackPair {
    fn ensure(
        &mut self,
        device: &ID3D12Device,
        bytes: u64,
    ) -> Result<(ID3D12Resource, ID3D12Resource, *const u8), &'static str> {
        let need = bytes.max(256).next_multiple_of(4096);
        let def = ensure_uav(device, &mut self.default, need)?;
        let (rb, ptr) = match &self.readback {
            Some((r, p, cap)) if *cap >= need => (r.clone(), *p),
            _ => {
                let (r, p) = create_mapped_buf(device, D3D12_HEAP_TYPE_READBACK, need, D3D12_RESOURCE_STATE_COPY_DEST)?;
                self.readback = Some((r.clone(), p.cast_const(), need));
                (r, p.cast_const())
            }
        };
        Ok((def, rb, ptr))
    }
}

fn ensure_uav(
    device: &ID3D12Device,
    slot: &mut Option<(ID3D12Resource, u64)>,
    bytes: u64,
) -> Result<ID3D12Resource, &'static str> {
    let need = bytes.max(256).next_multiple_of(4096);
    if let Some((r, cap)) = slot
        && *cap >= need
    {
        return Ok(r.clone());
    }
    let r = create_buf(
        device,
        D3D12_HEAP_TYPE_DEFAULT,
        need,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    )?;
    *slot = Some((r.clone(), need));
    Ok(r)
}

fn create_buf(
    device: &ID3D12Device,
    heap_type: D3D12_HEAP_TYPE,
    bytes: u64,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource, &'static str> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: heap_type,
        ..Default::default()
    };
    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: valid buffer creation parameters.
    unsafe {
        device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut res)
            .map_err(|_| "CreateCommittedResource failed")?;
    }
    res.ok_or("CreateCommittedResource returned null")
}

fn create_mapped_buf(
    device: &ID3D12Device,
    heap_type: D3D12_HEAP_TYPE,
    bytes: u64,
    state: D3D12_RESOURCE_STATES,
) -> Result<(ID3D12Resource, *mut u8), &'static str> {
    let buf = create_buf(device, heap_type, bytes, D3D12_RESOURCE_FLAG_NONE, state)?;
    let mut ptr = std::ptr::null_mut();
    // SAFETY: persistent map of upload/readback committed resource.
    unsafe { buf.Map(0, None, Some(&mut ptr)) }.map_err(|_| "Map failed")?;
    Ok((buf, ptr.cast()))
}

fn cb_param(reg: u32, dwords: u32) -> D3D12_ROOT_PARAMETER1 {
    D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: reg,
                RegisterSpace: 0,
                Num32BitValues: dwords,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

fn srv_param(reg: u32) -> D3D12_ROOT_PARAMETER1 {
    D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: reg,
                RegisterSpace: 0,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

fn uav_param(reg: u32) -> D3D12_ROOT_PARAMETER1 {
    D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: reg,
                RegisterSpace: 0,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

fn create_root_sig(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
) -> Result<ID3D12RootSignature, &'static str> {
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                ..Default::default()
            },
        },
    };
    let mut blob = None;
    // SAFETY: valid versioned root signature description.
    unsafe {
        D3D12SerializeVersionedRootSignature(&desc, &mut blob, None)
            .map_err(|_| "D3D12SerializeVersionedRootSignature failed")?;
        let blob = blob.ok_or("root sig blob null")?;
        let bytes = std::slice::from_raw_parts(
            blob.GetBufferPointer().cast::<u8>(),
            blob.GetBufferSize(),
        );
        device
            .CreateRootSignature(0, bytes)
            .map_err(|_| "CreateRootSignature failed")
    }
}

fn create_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    dxil: &[u8],
) -> Result<ID3D12PipelineState, &'static str> {
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root_sig.clone())),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: dxil.as_ptr().cast(),
            BytecodeLength: dxil.len(),
        },
        ..Default::default()
    };
    // SAFETY: live root signature and valid DXIL bytecode.
    let res = unsafe { device.CreateComputePipelineState(&desc) };
    drop(ManuallyDrop::into_inner(desc.pRootSignature));
    res.map_err(|_| "CreateComputePipelineState failed")
}

fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: ManuallyDrop::new(None),
            }),
        },
    }
}

fn drop_barriers(barriers: impl IntoIterator<Item = D3D12_RESOURCE_BARRIER>) {
    for b in barriers {
        if b.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION {
            // SAFETY: constructed with `ManuallyDrop`.
            let t = unsafe { ManuallyDrop::into_inner(b.Anonymous.Transition) };
            let _ = ManuallyDrop::into_inner(t.pResource);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dxc_compiles_all_six_bcpack_kernels() {
        let Ok(compiler) = DxcCompiler::load() else {
            eprintln!("dxcompiler.dll unavailable; skipping DXC SM 6.6 compile test");
            return;
        };
        for (name, src) in [
            ("bcpack_bcn_rans", BCN_RANS_HLSL),
            ("bcpack_bcn_rebuild", BCN_REBUILD_HLSL),
            ("bcpack_bc7_rans", BC7_RANS_HLSL),
            ("bcpack_bc7_rebuild", BC7_REBUILD_HLSL),
            ("bcpack_bc6h_rans", BC6H_RANS_HLSL),
            ("bcpack_bc6h_rebuild", BC6H_REBUILD_HLSL),
        ] {
            let dxil = compiler
                .compile_cs_6_6(src)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(dxil.starts_with(b"DXBC"), "{name} must produce DXBC container");
        }
    }

    #[test]
    fn warp_bcpack_decode_matches_cpu_for_all_formats() {
        let Some(host) = crate::host::Host::new_warp() else {
            return;
        };
        let tables = Tables::builtin().unwrap();
        let Ok(mut engine) = BcpackEngine::new(&host.device, &tables) else {
            eprintln!("WARP lacks DXC or Wave32 support; skipping GPU BCPack test");
            return;
        };
        let zero_stream_12 = [0x10u8, 0x00].repeat(12);
        let zero_stream_16 = [0x10u8, 0x00].repeat(16);
        for (bc, block_size, src) in [
            (1u8, 8usize, zero_stream_12.as_slice()),
            (3, 16, zero_stream_12.as_slice()),
            (4, 8, zero_stream_12.as_slice()),
            (5, 16, zero_stream_12.as_slice()),
            (6, 16, zero_stream_16.as_slice()),
            (7, 16, zero_stream_16.as_slice()),
        ] {
            let mut cpu_dst = vec![0u8; block_size];
            let cpu_res = victory_formats::bcpack::decode(bc, src, &mut cpu_dst, &tables);
            let mut gpu_dst = vec![0u8; block_size];
            let gpu_res = engine.decode(bc, src, &mut gpu_dst);
            assert_eq!(cpu_res.is_ok(), gpu_res.is_ok(), "bc={bc} status mismatch");
            if cpu_res.is_ok() {
                assert_eq!(gpu_dst, cpu_dst, "bc={bc} decoded output mismatch");
            }
        }
    }
}
