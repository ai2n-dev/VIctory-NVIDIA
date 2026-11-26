//! Textures shaders write that the title's own code then reads on the CPU.
//!
//! Adapted from `realblastex/flamingo` (`1f70cac`). On the console, CPU and GPU
//! share memory, so what a compute/UAV pass stores into a texture is visible in
//! title memory for CPU sampling (e.g., ocean height/displacement maps). On PC,
//! it stays in the texture's host copy.
//!
//! For textures of the kinds in `VICTORY_TEXTURE_BACK` (comma-separated
//! `<width>x<height>f<format>`; `0` for none; by default the kinds read by
//! title CPU code), a list in which shaders write one copies it into a readback
//! buffer when closing, and once the list has run the texels are tiled back into
//! the texture's title memory in the console's layout.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

const DEFAULT_KINDS: &str = "256x256f41,256x256f16,64x64f16,128x128f36";

fn parse_kinds(spec: &str) -> Vec<(u64, u32, u32)> {
    spec.split(',')
        .filter_map(|k| {
            let (size, format) = k.trim().split_once('f')?;
            let (w, h) = size.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
        })
        .collect()
}

/// The kinds copied back: `(width, height, format)`.
fn kinds() -> &'static [(u64, u32, u32)] {
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        let list = std::env::var("VICTORY_TEXTURE_BACK").unwrap_or_else(|_| DEFAULT_KINDS.to_owned());
        let kinds = parse_kinds(&list);
        if !kinds.is_empty() {
            tracing::info!(
                ?kinds,
                "textures of these kinds written via UAV are copied back into title memory (VICTORY_TEXTURE_BACK)"
            );
        }
        kinds
    })
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty() && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// A texture's copy into a persistently mapped readback buffer and where its
/// texels go once the list has run.
pub(crate) struct Back {
    _buffer: ID3D12Resource,
    mapped: usize,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    /// The texture's title memory and its layout's byte size.
    at: usize,
    size: usize,
}

/// The padded linear range that land() passes to the layout walker.
fn footprint_span(
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    rows: u32,
) -> Option<(usize, usize, usize)> {
    let pitch = footprint.Footprint.RowPitch as usize;
    let depth = footprint.Footprint.Depth as usize;
    if pitch == 0 || rows == 0 || depth == 0 {
        return None;
    }
    let depth_pitch = pitch.checked_mul(rows as usize)?;
    let len = depth_pitch.checked_mul(depth)?;
    let end = usize::try_from(footprint.Offset).ok()?.checked_add(len)?;
    Some((depth_pitch, len, end))
}

const SCRATCH_KEEP: usize = 8 << 20;
thread_local! {
    static TILED: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn with_tiled_scratch<R>(
    scratch: &mut Vec<u8>,
    size: usize,
    f: impl FnOnce(&mut [u8]) -> R,
) -> R {
    if scratch.len() < size {
        scratch.resize(size, 0);
    }
    let result = f(&mut scratch[..size]);
    // Large, one-off allocations must not become permanent commit.
    if scratch.capacity() > SCRATCH_KEEP {
        *scratch = Vec::new();
    }
    result
}

/// Records into `list` the copy of `texture` (title resource `resource`'s
/// host copy, already in `COPY_SOURCE` state) into a readback buffer.
pub(crate) fn record(
    list: &ID3D12GraphicsCommandList,
    resource: This,
    texture: &ID3D12Resource,
) -> Option<Back> {
    let host = crate::host::host()?;
    let (layout, at, size) = crate::resource::title_texture(resource)?;
    // SAFETY: `texture` is a live D3D12 texture resource.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels.max(1));
    let slices = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        1
    } else {
        u32::from(desc.DepthOrArraySize.max(1))
    };
    let subresources = mips.checked_mul(slices)?;
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: output slices are sized to `n == subresources`.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        )
    };
    // Adapt cool_fox 67028e21 (GPL-3.0): GetCopyableFootprints may
    // omit the padding after the last row from its total. Our linear
    // slices include that padding, so the allocation must cover it.
    if total == u64::MAX {
        return None;
    }
    for (footprint, &rows) in footprints.iter().zip(&rows) {
        let (_, _, end) = footprint_span(footprint, rows)?;
        total = total.max(end as u64);
    }
    let (buffer, mapped) = host.readback_buffer(total)?;
    if mapped.is_null() {
        return None;
    }
    crate::host::name(&buffer, || format!("texback {size:#x}"));
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer kept alive by `buffer` inside `Back`.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer to `texture` kept alive by the caller/list.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        // SAFETY: `list` is open and `texture` is in `COPY_SOURCE` state.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    COPIED.fetch_add(1, Ordering::Relaxed);
    Some(Back {
        _buffer: buffer,
        mapped: mapped as usize,
        footprints,
        rows,
        mips,
        layout,
        at,
        size,
    })
}

/// Textures copied back since the start.
pub(crate) static COPIED: AtomicU64 = AtomicU64::new(0);

/// Once the list that recorded `back` has run: tiles its texels into the
/// texture's title memory in the console's layout.
pub(crate) fn land(back: &Back) {
    let data = back.mapped as *const u8;
    if data.is_null() {
        return;
    }
    TILED.with_borrow_mut(|scratch| {
        with_tiled_scratch(scratch, back.size, |tiled| {
            // copy_mapped writes every byte, including zeros for unreadable
            // pages. The diagnostic no-copy mode must keep its old zero input.
            #[cfg(feature = "diagnostics")]
            if crate::copystat::mode() == 2 {
                tiled.fill(0);
            }
            crate::host::copy_mapped(tiled.as_mut_ptr(), back.at as *const u8, back.size, false);
            for (i, (footprint, &rows)) in back.footprints.iter().zip(&back.rows).enumerate() {
                let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
                let Some((depth_pitch, len, _)) = footprint_span(footprint, rows) else {
                    return;
                };
                let pitch = footprint.Footprint.RowPitch as usize;
                // SAFETY: record allocated the complete padded span and the
                // list has completed before landing.
                let linear =
                    unsafe { std::slice::from_raw_parts(data.add(footprint.Offset as usize), len) };
                if let Err(e) = back.layout.tile(tiled, mip, slice, linear, pitch, depth_pitch) {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled")
                    });
                    return;
                }
            }
            crate::watch::writing(back.at, back.size);
            crate::host::copy_mapped(back.at as *mut u8, tiled.as_ptr(), back.size, true);
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readback_span_includes_last_row_padding_and_checks_overflow() {
        let mut f = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 512,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                RowPitch: 256,
                Depth: 3,
                ..Default::default()
            },
        };
        assert_eq!(footprint_span(&f, 5), Some((1280, 3840, 4352)));
        assert_eq!(footprint_span(&f, 0), None);
        f.Offset = u64::MAX;
        assert_eq!(footprint_span(&f, 5), None);
        f.Offset = 0;
        f.Footprint.RowPitch = u32::MAX;
        f.Footprint.Depth = u32::MAX;
        assert_eq!(footprint_span(&f, u32::MAX), None);
    }

    #[test]
    fn tiled_scratch_reuses_storage_and_releases_large_allocations() {
        let mut scratch = Vec::new();
        with_tiled_scratch(&mut scratch, 4096, |b| b.fill(0x5a));
        let ptr = scratch.as_ptr();
        with_tiled_scratch(&mut scratch, 1024, |b| {
            assert_eq!(b, &[0x5a; 1024]);
            b.fill(0xa5);
        });
        assert_eq!(scratch.as_ptr(), ptr);
        assert_eq!(scratch.len(), 4096);
        assert_eq!(scratch[1024], 0x5a);
        with_tiled_scratch(&mut scratch, SCRATCH_KEEP + 1, |b| {
            assert_eq!(b.len(), SCRATCH_KEEP + 1);
        });
        assert_eq!(scratch.capacity(), 0);
    }

    #[test]
    fn parse_kinds_parses_defaults_and_disable() {
        assert_eq!(
            parse_kinds(DEFAULT_KINDS),
            [(256, 256, 41), (256, 256, 16), (64, 64, 16), (128, 128, 36)]
        );
        assert!(parse_kinds("0").is_empty());
        assert!(parse_kinds("").is_empty());
    }

    #[test]
    fn record_and_land_copies_uav_texture_into_tiled_title_memory() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let resource = crate::resource::test_texture(256, 256, 41);
        assert!(wanted(resource));
        let copy = crate::resource::host_copy(resource).expect("test texture has no host copy");
        // SAFETY: `copy.texture` is a live D3D12 resource.
        let desc = unsafe { copy.texture.GetDesc() };

        let list = host.new_list().expect("host command list");
        let mut footprints = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default()];
        let mut rows = [0u32];
        let mut total = 0u64;
        // SAFETY: output slices have length 1 matching `num_subresources`.
        unsafe {
            host.device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(footprints.as_mut_ptr()),
                Some(rows.as_mut_ptr()),
                None,
                Some(&mut total),
            );
        }
        let (upload, offset, ptr) = list.upload(total).expect("upload buffer");
        let pitch = footprints[0].Footprint.RowPitch as usize;
        let expected_texel = 0x3F80_1234u32;
        for y in 0..256usize {
            // SAFETY: inside the persistently mapped upload buffer row.
            let row_ptr = unsafe { ptr.add(footprints[0].Offset as usize + y * pitch) }.cast::<u32>();
            for x in 0..256usize {
                // SAFETY: `x < 256` fits within the row footprint.
                unsafe {
                    row_ptr
                        .add(x)
                        .write(expected_texel.wrapping_add((y * 256 + x) as u32));
                }
            }
        }
        footprints[0].Offset += offset;
        crate::list::ensure_state(&list, resource, 0x400); // COPY_DEST
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer kept alive by `copy`.
            pResource: unsafe { std::mem::transmute_copy(&copy.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer kept alive by `list`'s `uploads`.
            pResource: unsafe { std::mem::transmute_copy(&upload) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprints[0],
            },
        };
        // SAFETY: `list` is open and destinations are valid.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        list.texture_written(resource);
        list.close();
        host.execute(&[&list]);
        list.reset();

        let (layout, at, size) = crate::resource::title_texture(resource).unwrap();
        // SAFETY: `at..at + size` is committed by `test_texture`.
        let backing = unsafe { std::slice::from_raw_parts(at as *const u8, size) };
        let mut untiled = vec![0u8; 256 * 256 * 4];
        layout
            .detile(backing, 0, 0, &mut untiled, 256 * 4, 256 * 256 * 4)
            .expect("detile landed texture");
        for i in 0..(256 * 256) {
            let got = u32::from_le_bytes(untiled[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(got, expected_texel.wrapping_add(i as u32), "texel {i}");
        }
        type Release = extern "system" fn(This) -> u32;
        crate::object::slot::<Release>(resource, 0x10)(resource);
    }
}
