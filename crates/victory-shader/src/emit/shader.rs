//! A DXIL vertex, pixel, compute or mesh shader under construction: the module,
//! the `dx.op` declarations it calls, its signatures and bound resources,
//! and the metadata and container that wrap it up (SM 6.6, validator 1.6).

use super::bitcode::{Attr, Md, Module, Ty, V};
use super::container::{self, Binding, Psv, SigElement};

/// Where the shader reads its user data: `b0` in this space.
pub const USER_DATA_SPACE: u32 = 0;
/// Where immediate samplers are bound: `s<n>` in this space.
pub const SAMPLER_SPACE: u32 = 0;
/// Where root CBVs and SRVs are bound: `b<root parameter index>` and
/// `t<root parameter index>` in this space.
pub const ROOT_CBV_SPACE: u32 = 1;
/// The bytes of payload every translated mesh shader declares and never
/// reads. The host may put an amplification shader of its own in front of
/// it (the launch of an indirect mesh record, `victory-gpu`'s `launch`),
/// whose payload must have the size the mesh shader declares: a mesh
/// shader that declares none is refused with one (NVIDIA Blackwell drivers
/// return `E_INVALIDARG`), and one that declares a payload is still taken
/// without an amplification shader.
pub const MESH_PAYLOAD_BYTES: u32 = 4;
/// The register space the host's amplification shader binds in (`b0`,
/// `t0`, `u0`): no translated shader binds there.
pub const LAUNCH_SPACE: u32 = 9;
/// Where the host's runtime stride table is bound: `t0` in this space
/// (one `u32` element stride per view-heap slot).
pub const STRIDE_SPACE: u32 = 2;
/// Root SRV slot for the host ray-tracing acceleration structure
/// (`RTAccelerationStructure` for DXR Tier 1.1 inline ray queries).
pub const SLOT_TLAS: u8 = 252;

#[cfg(test)]
thread_local! {
    /// Per-thread override for [`constants_raw`] in isolated unit tests.
    pub(crate) static CONSTANTS_RAW_HERE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Opt-in (`VICTORY_PORT_CBV=1`, off by default): lower constant buffers as raw
/// buffer SRVs (table/heap CBVs as raw views, root CBVs as root SRVs) so the
/// host can bind 16-byte-aligned title buffers directly without copying.
/// Adapted from AvatarManz 3920dc26 (GPL-3.0).
pub fn constants_raw() -> bool {
    #[cfg(test)]
    if let Some(on) = CONSTANTS_RAW_HERE.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PORT_CBV").is_some_and(|v| v == "1"))
}

/// `ShaderFlags` bits (metadata) and their `SFI0` feature bits.
pub(crate) mod flags {
    pub const DOUBLES: u64 = 1 << 2;
    pub const RAW_BUFFERS: u64 = 1 << 4;
    pub const DOUBLE_EXT: u64 = 1 << 6;
    pub const STENCIL_REF: u64 = 1 << 11;
    pub const TYPED_UAV_LOADS: u64 = 1 << 13;
    /// UAVs in a vertex, geometry or mesh program.
    pub const UAVS_EVERY_STAGE: u64 = 1 << 16;
    pub const ROVS: u64 = 1 << 18;
    pub const WAVE_OPS: u64 = 1 << 19;
    pub const BARYCENTRICS: u64 = 1 << 22;
    pub const INT64: u64 = 1 << 20;
    /// AllocateRayQuery (DXR inline ray tracing, Tier 1.1).
    pub const RAYTRACING_1_1: u64 = 1 << 25;
    pub const ATOMIC_TYPED64: u64 = 1 << 27;
    pub const ATOMIC_INT64_GROUP_SHARED: u64 = 1 << 28;
    pub const ATOMIC_HEAP64: u64 = 1 << 32;
    pub const RESOURCE_HEAP: u64 = 1 << 30;
    pub const SAMPLER_HEAP: u64 = 1 << 31;
}

pub(crate) struct Types {
    pub void: Ty,
    pub i1: Ty,
    pub i32: Ty,
    pub f32: Ty,
    pub handle: Ty,
    pub props: Ty,
    pub bind: Ty,
    pub cbuf_ret: Ty,
    pub res_ret: Ty,
}

/// The vertices a rectangle's geometry shader emits.
pub(crate) const RECTANGLE_VERTICES: u32 = 6;

/// What a mesh shader declares beyond its thread-group size.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Mesh {
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// 1 lines, 2 triangles.
    pub topology: u8,
}

pub(crate) struct Shader {
    pub m: Module,
    /// Test builds: the property words of each handle annotation, in order.
    #[cfg(test)]
    pub annotations: Vec<(u32, u32)>,
    pub t: Types,
    pub pixel: bool,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    /// A mesh shader's per-primitive outputs.
    pub prims: Vec<SigElement>,
    /// A mesh shader (`threads` is its thread-group size).
    pub mesh: Option<Mesh>,
    /// A geometry shader from triangles to two-triangle strips
    /// (`rectangle`).
    pub rectangles: bool,
    /// Dwords of user data the shader reads (0: no user-data buffer).
    pub user_data: u32,
    /// Immediate samplers, bound as `s0..` in [`SAMPLER_SPACE`].
    pub samplers: u32,
    /// Bit n: immediate sampler n is a comparison sampler.
    pub cmp_samplers: u64,
    /// Root CBVs read, by root parameter index, with their handles.
    pub root_cbvs: Vec<(u8, V)>,
    /// Root SRVs (raw buffers) read, likewise.
    pub root_srvs: Vec<(u8, V)>,
    /// Where reads through a handle (a root view's) may start at most.
    pub limits: Vec<(V, V)>,
    /// Root UAVs (raw buffers) read or written, likewise.
    pub root_uavs: Vec<(u8, V)>,
    /// Whether they are globally coherent.
    pub root_uavs_shared: bool,
    /// Handle of the runtime stride table (`t0`, [`STRIDE_SPACE`]), if read.
    pub stride_table: Option<V>,
    pub flags: u64,
    /// A POPS pixel shader whose heap UAV writes must be rasterizer-ordered.
    pub ordered: bool,
    /// Heap handles made with a non-uniform index.
    pub nonuniform_handles: u32,
    /// A compute shader with this thread-group size.
    pub threads: Option<[u32; 3]>,
    /// The wave size a compute shader requires (`WaveSize`).
    pub wave_size: Option<u32>,
    /// Thread-group shared memory, in dwords.
    pub lds_dwords: u32,
    lds: Option<V>,
    main: V,
    pub entry: usize,
}

impl Shader {
    pub fn new(pixel: bool) -> Self {
        let mut m = Module::default();
        let void = m.void();
        let i1 = m.int(1);
        let i8 = m.int(8);
        let i32 = m.int(32);
        let f32 = m.float();
        let i8p = m.ptr(i8);
        let handle = m.named_struct("dx.types.Handle", vec![i8p]);
        let props = m.named_struct("dx.types.ResourceProperties", vec![i32, i32]);
        let bind = m.named_struct("dx.types.ResBind", vec![i32, i32, i32, i8]);
        let cbuf_ret = m.named_struct("dx.types.CBufRet.i32", vec![i32, i32, i32, i32]);
        let res_ret = m.named_struct("dx.types.ResRet.f32", vec![f32, f32, f32, f32, i32]);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        let main = m.function("main", main_ty, None, true);
        let entry = m.new_block();
        Shader {
            m,
            #[cfg(test)]
            annotations: Vec::new(),
            t: Types {
                void,
                i1,
                i32,
                f32,
                handle,
                props,
                bind,
                cbuf_ret,
                res_ret,
            },
            pixel,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh: None,
            rectangles: false,
            user_data: 0,
            samplers: 0,
            cmp_samplers: 0,
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            limits: Vec::new(),
            root_uavs: Vec::new(),
            root_uavs_shared: false,
            stride_table: None,
            flags: 0,
            ordered: false,
            nonuniform_handles: 0,
            threads: None,
            wave_size: None,
            lds_dwords: 0,
            lds: None,
            main,
            entry,
        }
    }

    /// The single LDS allocation in address space 3.
    pub fn lds(&mut self, wide: bool) -> V {
        if let Some(g) = self.lds {
            return g;
        }
        let arr = if wide {
            let i64t = self.m.int(64);
            self.m.array(self.lds_dwords.div_ceil(2), i64t)
        } else {
            self.m.array(self.lds_dwords, self.t.i32)
        };
        let g = self.m.global("lds", arr, 3);
        if wide {
            self.m.align_global_to_eight(g);
        }
        self.lds = Some(g);
        g
    }

    /// Calls `dx.op.<name>` (class and overload) with `opcode` and `args`.
    pub fn op(
        &mut self,
        block: usize,
        name: &str,
        attr: Attr,
        ret: Ty,
        opcode: u32,
        args: &[V],
    ) -> V {
        // A raw buffer load through a limited handle.
        const RAW_LOAD: u32 = 139;
        if opcode == RAW_LOAD
            && let Some(&(_, limit)) = self.limits.iter().find(|l| l.0 == args[0])
        {
            let i32t = self.t.i32;
            let at = self.op(block, "binary.i32", Attr::ReadNone, i32t, 40, &[args[1], limit]);
            let mut args = args.to_vec();
            args[1] = at;
            return self.op_as_is(block, name, attr, ret, opcode, &args);
        }
        self.op_as_is(block, name, attr, ret, opcode, args)
    }

    fn op_as_is(
        &mut self,
        block: usize,
        name: &str,
        attr: Attr,
        ret: Ty,
        opcode: u32,
        args: &[V],
    ) -> V {
        let mut params = vec![self.t.i32];
        params.extend(args.iter().map(|&a| self.m.type_of(a)));
        let fty = self.m.func_ty(ret, params);
        self.m.ptr(fty);
        let f = self
            .m
            .function(&format!("dx.op.{name}"), fty, Some(attr), false);
        let mut all = vec![self.m.i32(opcode)];
        all.extend(args);
        self.m.call(block, f, all)
    }

    /// The handle of the user-data constant buffer, made in the entry block.
    pub fn user_data_handle(&mut self) -> V {
        let bind = [
            self.m.i32(0),
            self.m.i32(0),
            self.m.i32(USER_DATA_SPACE),
            self.m.i8(2), // CBV
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let block = self.entry;
        let (i, f) = (self.m.i32(0), self.m.bool(false));
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, i, f],
        );
        // The size is known in `finish`, which fixes this placeholder.
        self.annotate(block, h, 13, u32::MAX)
    }

    /// The handle of the root CBV of root parameter `slot`, made in the
    /// entry block.
    pub fn root_cbv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_cbvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(2)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 13, 65536);
        self.root_cbvs.push((slot, h));
        h
    }

    /// The handle of the root SRV (a raw buffer) of root parameter `slot`.
    pub fn root_srv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_srvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_srvs.push((slot, h));
        h
    }

    /// Handle for the host DXR top-level acceleration structure ([`SLOT_TLAS`]).
    #[allow(dead_code)]
    pub fn tlas_handle(&mut self) -> V {
        if let Some(&(_, h)) = self.root_srvs.iter().find(|r| r.0 == SLOT_TLAS) {
            return h;
        }
        let r = self.m.i32(u32::from(SLOT_TLAS));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        // DXIL resource kind 16: RTAccelerationStructure.
        let h = self.annotate(block, h, 16, 0);
        self.flags |= flags::RAYTRACING_1_1;
        self.root_srvs.push((SLOT_TLAS, h));
        h
    }

    /// Allocates an inline ray-query handle with constant ray `flags`.
    #[allow(dead_code)]
    pub fn ray_query(&mut self, block: usize, flags: u32) -> V {
        let f = self.m.i32(flags);
        self.op(block, "allocateRayQuery", Attr::NoUnwind, self.t.i32, 178, &[f])
    }

    /// Begins an inline ray trace against the host TLAS ([`SLOT_TLAS`]).
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub fn trace_ray_inline(
        &mut self,
        block: usize,
        rq: V,
        flags: V,
        mask: V,
        origin: [V; 3],
        tmin: V,
        dir: [V; 3],
        tmax: V,
    ) {
        let h = self.tlas_handle();
        let args = [
            rq, h, flags, mask, origin[0], origin[1], origin[2], tmin, dir[0], dir[1], dir[2],
            tmax,
        ];
        let void = self.t.void;
        self.op(block, "rayQuery_TraceRayInline", Attr::NoUnwind, void, 179, &args);
    }

    /// Advances an inline ray query (`RayQuery::Proceed`).
    #[allow(dead_code)]
    pub fn ray_query_proceed(&mut self, block: usize, rq: V) -> V {
        let i1 = self.t.i1;
        self.op(block, "rayQuery_Proceed.i1", Attr::NoUnwind, i1, 180, &[rq])
    }

    /// Returns committed hit status for an inline ray query (`0`=none, `1`=triangle).
    #[allow(dead_code)]
    pub fn ray_query_committed_status(&mut self, block: usize, rq: V) -> V {
        let i32t = self.t.i32;
        self.op(
            block,
            "rayQuery_StateScalar.i32",
            Attr::ReadOnly,
            i32t,
            184,
            &[rq],
        )
    }

    /// Returns committed hit distance `T` for an inline ray query.
    #[allow(dead_code)]
    pub fn ray_query_committed_t(&mut self, block: usize, rq: V) -> V {
        let f32t = self.t.f32;
        self.op(
            block,
            "rayQuery_StateScalar.f32",
            Attr::ReadOnly,
            f32t,
            200,
            &[rq],
        )
    }

    /// The handle of the root UAV (a raw buffer) of root parameter `slot`.
    pub fn root_uav_handle(&mut self, slot: u8, shared: bool) -> V {
        if let Some(&(_, h)) = self.root_uavs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(1)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        self.root_uavs_shared |= shared;
        let shared = if shared { 0x4000 } else { 0 };
        let h = self.annotate(block, h, 0x1000 | 11 | shared, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_uavs.push((slot, h));
        h
    }

    /// The handle of the host's runtime stride table (`t0` in
    /// [`STRIDE_SPACE`], a raw buffer of `u32` byte strides indexed by
    /// view-heap slot).
    pub fn stride_table_handle(&mut self) -> V {
        if let Some(h) = self.stride_table {
            return h;
        }
        let z = self.m.i32(0);
        let bind = [z, z, self.m.i32(STRIDE_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, z, f],
        );
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.stride_table = Some(h);
        h
    }

    /// The handle of immediate sampler `n` (a comparison sampler when its
    /// bit in `cmp_samplers` is set).
    pub fn sampler_handle(&mut self, block: usize, n: u32) -> V {
        let bind = [
            self.m.i32(n),
            self.m.i32(n),
            self.m.i32(SAMPLER_SPACE),
            self.m.i8(3), // sampler
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let (i, f) = (self.m.i32(n), self.m.bool(false));
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, i, f],
        );
        let cmp = if self.cmp_samplers >> n & 1 != 0 {
            0x8000
        } else {
            0
        };
        self.annotate(block, h, 14 | cmp, 0)
    }

    pub fn annotate(&mut self, block: usize, h: V, props0: u32, props1: u32) -> V {
        #[cfg(test)]
        self.annotations.push((props0, props1));
        let p = [self.m.i32(props0), self.m.i32(props1)];
        let p = self.m.aggregate(self.t.props, &p);
        self.op(
            block,
            "annotateHandle",
            Attr::ReadNone,
            self.t.handle,
            216,
            &[h, p],
        )
    }

    /// A descriptor heap handle, annotated with `props`; `nonuniform` when
    /// the index may differ between lanes (`NonUniformResourceIndex`).
    pub fn heap_handle(
        &mut self,
        block: usize,
        index: V,
        sampler: bool,
        nonuniform: bool,
        props: (u32, u32),
    ) -> V {
        // An index past the heap is a fault that stops the host GPU.
        let index = if !sampler && super::limits_indices() {
            let last = self.m.i32(super::heap_clamp());
            let i32t = self.t.i32;
            self.op(block, "binary.i32", Attr::ReadNone, i32t, 40, &[index, last])
        } else {
            index
        };
        self.handle_at(block, index, sampler, nonuniform, props)
    }

    /// A handle of the heap slot at `index`, which is known to be in it.
    pub fn handle_at(
        &mut self,
        block: usize,
        index: V,
        sampler: bool,
        nonuniform: bool,
        props: (u32, u32),
    ) -> V {
        self.flags |= if sampler {
            flags::SAMPLER_HEAP
        } else {
            flags::RESOURCE_HEAP
        };
        // (bit 12 of word 0 is UAV; bit 13 marks it rasterizer-ordered)
        let props = if self.ordered && !sampler && props.0 & 0x1000 != 0 {
            self.flags |= flags::ROVS;
            (props.0 | 0x2000, props.1)
        } else {
            props
        };
        self.nonuniform_handles += u32::from(nonuniform);
        let (s, n) = (self.m.bool(sampler), self.m.bool(nonuniform));
        let h = self.op(
            block,
            "createHandleFromHeap",
            Attr::ReadNone,
            self.t.handle,
            218,
            &[index, s, n],
        );
        self.annotate(block, h, props.0, props.1)
    }

    /// DXIL counts declared UAV ranges, not UAVs accessed through the heap.
    fn update_stage_uav_flag(&mut self) {
        let compute = self.threads.is_some() && self.mesh.is_none() && !self.rectangles;
        let declared = !self.root_uavs.is_empty();
        if declared && !self.pixel && !compute {
            self.flags |= flags::UAVS_EVERY_STAGE;
        }
    }

    /// The finished container.
    pub fn finish(mut self) -> Vec<u8> {
        self.update_stage_uav_flag();
        let ud_size = (self.user_data * 4).next_multiple_of(16);
        self.fix_user_data_size(ud_size);
        // The clean-up, before anything is declared about the code: a
        // feature whose last use it removed must not be declared (the
        // validator compares the flags with what the code uses).
        let passes = super::shader_opt() & super::bitcode::opt::MODULE;
        if passes != 0 {
            self.m.optimise(passes);
            let used = self.m.usage();
            for (flag, used) in [
                (flags::TYPED_UAV_LOADS, used.typed_uav_loads),
                (flags::ATOMIC_TYPED64 | flags::ATOMIC_HEAP64, used.atomic64),
            ] {
                if !used {
                    self.flags &= !flag;
                }
            }
        }
        // The flags of what lowering made, as the module writes it: what only
        // unread values or never-taken edges used is not written.
        let uses = self.m.uses();
        for (flag, used) in [
            (flags::RESOURCE_HEAP, uses.heap),
            (flags::SAMPLER_HEAP, uses.sampler_heap),
            (flags::ROVS, uses.rovs),
            (flags::WAVE_OPS, uses.wave),
            (flags::INT64, uses.int64),
        ] {
            self.flags = if used { self.flags | flag } else { self.flags & !flag };
        }
        let mut resources = None;
        let mut bindings = Vec::new();
        let i32c = |s: &mut Self, v: u32| {
            let c = s.m.i32(v);
            s.m.md_value(c)
        };
        if self.user_data > 0
            || self.samplers > 0
            || !self.root_cbvs.is_empty()
            || !self.root_srvs.is_empty()
            || !self.root_uavs.is_empty()
            || self.stride_table.is_some()
        {
            let empty = self.m.md_str("");
            let mut cbvs = None;
            let mut cbv_list = Vec::new();
            if self.user_data > 0 {
                let arr = self.m.array(self.user_data.next_multiple_of(4), self.t.i32);
                let ty = self.m.named_struct("UserData", vec![arr]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let ops = vec![
                    Some(i32c(&mut self, 0)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, USER_DATA_SPACE)),
                    Some(i32c(&mut self, 0)),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, ud_size)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: USER_DATA_SPACE,
                    register: 0,
                    dxil_kind: 13,
                });
            }
            let blob = self.m.array(16384, self.t.i32);
            let ty = self.m.named_struct("RootCbv", vec![blob]);
            let p = self.m.ptr(ty);
            let u = self.m.undef(p);
            for (slot, _) in self.root_cbvs.clone() {
                let ops = vec![
                    Some(i32c(&mut self, cbv_list.len() as u32)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, ROOT_CBV_SPACE)),
                    Some(i32c(&mut self, u32::from(slot))),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, 65536)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: ROOT_CBV_SPACE,
                    register: u32::from(slot),
                    dxil_kind: 13,
                });
            }
            if !cbv_list.is_empty() {
                cbvs = Some(self.m.md_node(cbv_list));
            }
            let mut srvs = None;
            if !self.root_srvs.is_empty() || self.stride_table.is_some() {
                let mut specs: Vec<(u32, u32)> = self
                    .root_srvs
                    .iter()
                    .map(|&(slot, _)| (ROOT_CBV_SPACE, u32::from(slot)))
                    .collect();
                if self.stride_table.is_some() {
                    specs.push((STRIDE_SPACE, 0));
                }
                let ty = self
                    .m
                    .named_struct("struct.ByteAddressBuffer", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let rtas = self
                    .m
                    .named_struct("struct.RaytracingAccelerationStructure", vec![self.t.i32]);
                let rtas = self.m.ptr(rtas);
                let rtas = self.m.undef(rtas);
                let mut list = Vec::new();
                for (n, (space, reg)) in specs.into_iter().enumerate() {
                    let is_tlas = space == ROOT_CBV_SPACE && reg == u32::from(SLOT_TLAS);
                    // 16: RTAccelerationStructure; 11: raw ByteAddressBuffer.
                    let (var, shape) = if is_tlas { (rtas, 16) } else { (u, 11) };
                    let ops = vec![
                        Some(i32c(&mut self, n as u32)),
                        Some(self.m.md_value(var)),
                        Some(empty),
                        Some(i32c(&mut self, space)),
                        Some(i32c(&mut self, reg)),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, shape)),
                        Some(i32c(&mut self, 0)),
                        // RTAccelerationStructure element metadata: tag 0, type u32 (5).
                        if is_tlas {
                            let tag = i32c(&mut self, 0);
                            let u32k = i32c(&mut self, 5);
                            Some(self.m.md_node(vec![Some(tag), Some(u32k)]))
                        } else {
                            None
                        },
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 4,
                        space,
                        register: reg,
                        dxil_kind: shape,
                    });
                }
                srvs = Some(self.m.md_node(list));
            }
            let mut uavs = None;
            if !self.root_uavs.is_empty() {
                let shared = self.m.bool(self.root_uavs_shared);
                let shared_md = self.m.md_value(shared);
                let ty = self
                    .m
                    .named_struct("struct.RWByteAddressBuffer", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let f = self.m.bool(false);
                let f_md = self.m.md_value(f);
                let mut list = Vec::new();
                for (n, &(slot, _)) in self.root_uavs.clone().iter().enumerate() {
                    let reg = u32::from(slot);
                    let ops = vec![
                        Some(i32c(&mut self, n as u32)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, ROOT_CBV_SPACE)),
                        Some(i32c(&mut self, reg)),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, 11)),
                        Some(shared_md),
                        Some(f_md),
                        Some(f_md),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 7,
                        space: ROOT_CBV_SPACE,
                        register: reg,
                        dxil_kind: 11,
                    });
                }
                uavs = Some(self.m.md_node(list));
            }
            let mut samplers = None;
            if self.samplers > 0 {
                let ty = self.m.named_struct("struct.SamplerState", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let mut list = Vec::new();
                for n in 0..self.samplers {
                    let kind = (self.cmp_samplers >> n & 1) as u32;
                    let ops = vec![
                        Some(i32c(&mut self, n)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, SAMPLER_SPACE)),
                        Some(i32c(&mut self, n)),
                        Some(i32c(&mut self, 1)),
                        // Sampler kind: 1 comparison.
                        Some(i32c(&mut self, kind)),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 1,
                        space: SAMPLER_SPACE,
                        register: n,
                        dxil_kind: 14,
                    });
                }
                samplers = Some(self.m.md_node(list));
            }
            resources = Some(self.m.md_node(vec![srvs, uavs, cbvs, samplers]));
            // PSV0 lists them as the module does: CBVs, samplers, SRVs, UAVs.
            bindings.sort_by_key(|b| match b.kind {
                2 => 0,
                1 => 1,
                4 => 2,
                _ => 3,
            });
        }
        if let Some(r) = resources {
            self.m.named_md("dx.resources", vec![r]);
        }

        let graphics = self.threads.is_none() || self.mesh.is_some();
        if graphics {
            // shortcut: every output is declared to depend on every input; the
            // exact sets need a dependence pass. Drivers read this state.
            let scalars = |els: &[SigElement]| {
                els.iter()
                    .filter_map(|e| e.row.map(|r| r * 4 + u32::from(e.col + e.cols)))
                    .max()
                    .unwrap_or(0)
            };
            let (ni, no) = (scalars(&self.inputs), scalars(&self.outputs));
            let mut view = vec![ni, no];
            // Mesh shaders have no inputs: then the primitive outputs' count.
            if self.mesh.is_some() {
                view.push(scalars(&self.prims));
            }
            for _ in 0..ni {
                for k in 0..no.div_ceil(32) {
                    let bits = (no - 32 * k).min(32);
                    view.push(if bits == 32 {
                        u32::MAX
                    } else {
                        (1 << bits) - 1
                    });
                }
            }
            // Geometry shaders list four streams; ours use the first.
            if self.rectangles {
                view.extend([0; 3]);
            }
            let arr = self.m.array(view.len() as u32, self.t.i32);
            let elems: Vec<V> = view.iter().map(|&v| self.m.i32(v)).collect();
            let arr = self.m.aggregate(arr, &elems);
            let arr = self.m.md_value(arr);
            let view_node = self.m.md_node(vec![Some(arr)]);
            self.m.named_md("dx.viewIdState", vec![view_node]);
        }

        let sigs = if graphics {
            let ins = self.signature_md(self.inputs.clone());
            let outs = self.signature_md(self.outputs.clone());
            let prims = self.signature_md(self.prims.clone());
            Some(self.m.md_node(vec![ins, outs, prims]))
        } else {
            None
        };
        // A pixel shader with an SV_StencilRef output (as DXC collects it).
        if self.pixel && self.outputs.iter().any(|o| o.kind == 20) {
            self.flags |= flags::STENCIL_REF;
        }
        // Entry properties: shader flags (tag 0), thread-group size (4),
        // wave size (11).
        let mut props = Vec::new();
        if self.flags != 0 {
            props.push(i32c(&mut self, 0));
            let i64t = self.m.int(64);
            let f = self.m.int_const(i64t, self.flags as i64);
            props.push(self.m.md_value(f));
        }
        if let Some(threads) = self.threads {
            let t = threads.map(|n| Some(i32c(&mut self, n)));
            let t = self.m.md_node(t.to_vec());
            match self.mesh {
                // Mesh state (tag 9): threads, maximum vertices and
                // primitives, topology, payload bytes.
                Some(mesh) => {
                    props.push(i32c(&mut self, 9));
                    let state = [
                        mesh.max_vertices,
                        mesh.max_primitives,
                        mesh.topology.into(),
                        MESH_PAYLOAD_BYTES,
                    ]
                    .map(|n| Some(i32c(&mut self, n)));
                    let mut node = vec![Some(t)];
                    node.extend(state);
                    props.push(self.m.md_node(node));
                }
                None => {
                    props.push(i32c(&mut self, 4));
                    props.push(t);
                }
            }
        }
        // Geometry state (tag 1): triangles in, six vertices at most, stream
        // 0, triangle strips out, one instance.
        if self.rectangles {
            props.push(i32c(&mut self, 1));
            let state = [3, RECTANGLE_VERTICES, 1, 5, 1].map(|n| Some(i32c(&mut self, n)));
            props.push(self.m.md_node(state.to_vec()));
        }
        if let Some(n) = self.wave_size {
            props.push(i32c(&mut self, 11));
            let n = Some(i32c(&mut self, n));
            props.push(self.m.md_node(vec![n]));
        }
        let props =
            (!props.is_empty()).then(|| self.m.md_node(props.into_iter().map(Some).collect()));
        let main = self.m.md_value(self.main);
        let name = self.m.md_str("main");
        let entry = self
            .m
            .md_node(vec![Some(main), Some(name), sigs, resources, props]);

        let ident = self.m.md_str("victory");
        let ident = self.m.md_node(vec![Some(ident)]);
        self.m.named_md("llvm.ident", vec![ident]);
        let one = i32c(&mut self, 1);
        let six = i32c(&mut self, 6);
        let version = self.m.md_node(vec![Some(one), Some(six)]);
        self.m.named_md("dx.version", vec![version]);
        self.m.named_md("dx.valver", vec![version]);
        // Program kinds: 0 pixel, 1 vertex, 2 geometry, 5 compute, 13 mesh.
        let (name, kind) = match (self.threads, self.pixel) {
            _ if self.mesh.is_some() => ("ms", 13u32),
            _ if self.rectangles => ("gs", 2),
            (Some(_), _) => ("cs", 5u32),
            (None, true) => ("ps", 0),
            (None, false) => ("vs", 1),
        };
        let name = self.m.md_str(name);
        let sm = self.m.md_node(vec![Some(name), Some(six), Some(six)]);
        self.m.named_md("dx.shaderModel", vec![sm]);
        self.m.named_md("dx.entryPoints", vec![entry]);

        let bitcode = self.m.write();
        let mut dxil = Vec::with_capacity(24 + bitcode.len());
        dxil.extend((kind << 16 | 0x66).to_le_bytes());
        dxil.extend((((24 + bitcode.len()) / 4) as u32).to_le_bytes());
        dxil.extend(b"DXIL");
        dxil.extend(0x106u32.to_le_bytes());
        dxil.extend(16u32.to_le_bytes());
        dxil.extend((bitcode.len() as u32).to_le_bytes());
        dxil.extend(&bitcode);

        let mut features = 0u64;
        if self.flags & flags::UAVS_EVERY_STAGE != 0 {
            features |= 0x4;
        }
        if self.flags & flags::DOUBLES != 0 {
            features |= 0x1;
        }
        if self.flags & flags::DOUBLE_EXT != 0 {
            features |= 0x20;
        }
        if self.flags & flags::UAVS_EVERY_STAGE != 0 {
            features |= 0x4;
        }
        if self.flags & flags::STENCIL_REF != 0 {
            features |= 0x200;
        }
        if self.flags & flags::TYPED_UAV_LOADS != 0 {
            features |= 0x800;
        }
        if self.flags & flags::ROVS != 0 {
            features |= 0x1000;
        }
        if self.flags & flags::WAVE_OPS != 0 {
            features |= 0x4000;
        }
        if self.flags & flags::INT64 != 0 {
            features |= 0x8000;
        }
        if self.flags & flags::BARYCENTRICS != 0 {
            features |= 0x2_0000;
        }
        if self.flags & flags::RESOURCE_HEAP != 0 {
            features |= 0x200_0000;
        }
        if self.flags & flags::SAMPLER_HEAP != 0 {
            features |= 0x400_0000;
        }
        if self.flags & flags::ATOMIC_TYPED64 != 0 {
            features |= 0x40_0000;
        }
        if self.flags & flags::ATOMIC_INT64_GROUP_SHARED != 0 {
            features |= 0x80_0000;
        }
        if self.flags & flags::ATOMIC_HEAP64 != 0 {
            features |= 0x1000_0000;
        }
        // D3D_SHADER_FEATURE_RAYTRACING_TIER_1_1.
        if self.flags & flags::RAYTRACING_1_1 != 0 {
            features |= 0x0010_0000;
        }
        let sfi0 = features.to_le_bytes();
        let isg1 = container::signature(&self.inputs, true);
        let osg1 = container::signature(&self.outputs, false);
        let flag = if self.pixel {
            self.outputs.iter().any(|o| (17..=19).contains(&o.kind))
        } else {
            self.outputs.iter().any(|o| o.kind == 3)
        };
        let psv0 = container::psv(&Psv {
            stage: kind as u8,
            flag: flag && self.threads.is_none(),
            inputs: &self.inputs,
            outputs: &self.outputs,
            prims: &self.prims,
            mesh: self.mesh.map(|m| (m, self.lds_dwords * 4)),
            rectangles: self.rectangles,
            bindings: &bindings,
            threads: self.threads.unwrap_or_default(),
            wave_size: self.wave_size,
        });
        let hash = [0u8; 20];
        let psg1 = container::signature(&self.prims, false);
        let mut parts: Vec<(&[u8; 4], &[u8])> =
            vec![(b"SFI0", &sfi0), (b"ISG1", &isg1), (b"OSG1", &osg1)];
        if self.mesh.is_some() {
            parts.push((b"PSG1", &psg1));
        }
        parts.extend([(b"PSV0", &psv0[..]), (b"HASH", &hash), (b"DXIL", &dxil)]);
        container::build(&parts)
    }

    /// The user-data buffer's annotation carries its size, known only now.
    fn fix_user_data_size(&mut self, size: u32) {
        let old = [self.m.i32(13), self.m.i32(u32::MAX)];
        let old = self.m.aggregate(self.t.props, &old);
        let new = [self.m.i32(13), self.m.i32(size)];
        let new = self.m.aggregate(self.t.props, &new);
        self.m.replace_const(old, new);
    }

    fn signature_md(&mut self, elements: Vec<SigElement>) -> Option<Md> {
        if elements.is_empty() {
            return None;
        }
        let mut list = Vec::new();
        for (id, e) in elements.iter().enumerate() {
            let c32 = |s: &mut Self, v: u32| {
                let c = s.m.i32(v);
                Some(s.m.md_value(c))
            };
            let c8 = |s: &mut Self, v: u8| {
                let c = s.m.i8(v);
                Some(s.m.md_value(c))
            };
            let idx = c32(self, e.index);
            let indices = Some(self.m.md_node(vec![idx]));
            let usage = if e.usage != 0 {
                let (tag, mask) = (c32(self, 3), c32(self, u32::from(e.usage)));
                Some(self.m.md_node(vec![tag, mask]))
            } else {
                None
            };
            let ops = vec![
                c32(self, id as u32),
                Some(self.m.md_str(&e.name)),
                c8(self, e.comp),
                c8(self, e.kind),
                indices,
                c8(self, e.interp),
                c32(self, 1),
                c8(self, e.cols),
                c32(self, e.row.unwrap_or(u32::MAX)),
                c8(self, if e.row.is_some() { e.col } else { u8::MAX }),
                usage,
            ];
            list.push(Some(self.m.md_node(ops)));
        }
        Some(self.m.md_node(list))
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::super::bitcode::Term;
    use super::*;
    use crate::emit::validator::Validator;

    #[test]
    fn uavs_outside_pixel_and_compute_programs_are_declared() {
        let mut validator = Validator::load(Validator::default_path().as_deref()).ok();
        for (kind, pixel, threads, heap) in [
            ("vs", false, None, false),
            ("ps", true, None, false),
            ("cs", false, Some([64, 1, 1]), false),
            ("vs_heap", false, None, true),
        ] {
            let mut s = Shader::new(pixel);
            s.threads = threads;
            let b = s.entry;
            let h = if heap {
                let index = s.m.i32(0);
                s.heap_handle(b, index, false, false, (11 | 0x1000, 0))
            } else {
                s.root_uav_handle(1, true)
            };
            let (or, zero, one) = (s.m.i32(2), s.m.i32(0), s.m.i32(1));
            let iu = s.m.undef(s.t.i32);
            let i32t = s.t.i32;
            s.op(b, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[h, or, zero, iu, iu, one]);
            s.m.terminate(b, Term::Ret);
            let mut dxil = s.finish();
            if let Some(v) = validator.as_mut() {
                v.validate(&mut dxil)
                    .unwrap_or_else(|err| panic!("{kind} shader failed validation: {err}"));
            }
        }
    }

    #[test]
    fn ordered_pixel_marks_heap_uavs_as_rovs() {
        let mut validator = Validator::load(Validator::default_path().as_deref()).ok();
        let sfi0 = |dxil: &[u8]| -> u64 {
            let c = crate::dxbc::Container::parse(dxil).unwrap();
            let part = c.part(crate::dxbc::FourCC::SFI0).unwrap();
            u64::from_le_bytes(part.data[..8].try_into().unwrap())
        };
        // Live heap UAV handle in an ordered PS: annotated with bit 13 (ROV) and declares SFI0 0x1000.
        let mut s = Shader::new(true);
        s.ordered = true;
        let b = s.entry;
        let index = s.m.i32(0);
        let h = s.heap_handle(b, index, false, false, (11 | 0x1000, 0));
        assert_ne!(s.annotations[0].0 & 0x2000, 0);
        let (or, zero, one) = (s.m.i32(2), s.m.i32(0), s.m.i32(1));
        let iu = s.m.undef(s.t.i32);
        let i32t = s.t.i32;
        s.op(b, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[h, or, zero, iu, iu, one]);
        s.m.terminate(b, Term::Ret);
        let mut live = s.finish();
        assert_ne!(sfi0(&live) & 0x1000, 0);
        if let Some(v) = validator.as_mut() {
            v.validate(&mut live)
                .unwrap_or_else(|err| panic!("ordered PS live ROV failed validation: {err}"));
        }

        // Dead heap UAV handle in an ordered PS: DCE strips annotateHandle, so SFI0 must not declare ROVS.
        let mut s = Shader::new(true);
        s.ordered = true;
        let b = s.entry;
        let index = s.m.i32(0);
        let _dead = s.heap_handle(b, index, false, false, (11 | 0x1000, 0));
        s.m.terminate(b, Term::Ret);
        let mut dead = s.finish();
        assert_eq!(sfi0(&dead) & 0x1000, 0);
        if let Some(v) = validator.as_mut() {
            v.validate(&mut dead)
                .unwrap_or_else(|err| panic!("ordered PS dead ROV failed validation: {err}"));
        }
    }
}

#[cfg(test)]
mod stage_uav_tests {
    use super::*;

    #[test]
    fn declared_uavs_require_graphics_stage_flag() {
        for (pixel, threads, mesh, rectangles, declared, want) in [
            (false, None, false, false, true, true),
            (true, None, false, false, true, false),
            (false, Some([1, 1, 1]), false, false, true, false),
            (false, Some([1, 1, 1]), true, false, true, true),
            (false, None, false, true, true, true),
            (false, None, false, false, false, false),
        ] {
            let mut shader = Shader::new(pixel);
            shader.threads = threads;
            shader.rectangles = rectangles;
            if mesh {
                shader.mesh = Some(Mesh {
                    max_vertices: 3,
                    max_primitives: 1,
                    topology: 2,
                });
            }
            if declared {
                shader.root_uav_handle(0, false);
            } else {
                shader.flags |= flags::RESOURCE_HEAP;
            }
            shader.update_stage_uav_flag();
            assert_eq!(shader.flags & flags::UAVS_EVERY_STAGE != 0, want);
        }
    }

    #[test]
    fn vertex_root_uav_sets_container_feature_bit() {
        let mut shader = Shader::new(false);
        shader.root_uav_handle(0, false);
        shader.m.terminate(shader.entry, super::super::bitcode::Term::Ret);
        let dxil = shader.finish();
        let first = u32::from_le_bytes(dxil[32..36].try_into().unwrap()) as usize;
        assert_eq!(&dxil[first..first + 4], b"SFI0");
        let features = u64::from_le_bytes(dxil[first + 8..first + 16].try_into().unwrap());
        assert_ne!(features & 0x4, 0);
    }
}
