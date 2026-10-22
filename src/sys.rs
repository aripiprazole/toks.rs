//! Raw bindings to `toks.h` (abi 0.4, toks 0.3.2): one item per declaration, same names, same values.
//!
//! The header is the source of truth for every contract (lengths, lifetimes, error codes); read it alongside
//! these declarations. The safe API in the crate root is built on them.

#![allow(non_camel_case_types, non_snake_case)]

use core::ffi::{c_char, c_void};
use core::marker::{PhantomData, PhantomPinned};

pub const TOKS_ABI_MAJOR: u32 = 0;
pub const TOKS_ABI_MINOR: u32 = 4;

pub const TOKS_VERSION_MAJOR: u32 = 0;
pub const TOKS_VERSION_MINOR: u32 = 3;
pub const TOKS_VERSION_PATCH: u32 = 2;
pub const TOKS_VERSION: &str = "0.3.2";

// ---- errors ----------------------------------------------------------------------------------------------

pub const TOKS_E_OPEN: i64 = -1;
pub const TOKS_E_FORMAT: i64 = -2;
pub const TOKS_E_UNSUPPORTED: i64 = -3;
pub const TOKS_E_TIER: i64 = -5;
pub const TOKS_E_SCRATCH: i64 = -6;
pub const TOKS_E_ID: i64 = -7;
pub const TOKS_E_CAP: i64 = -8;
pub const TOKS_E_LIMIT: i64 = -9;
pub const TOKS_E_ARG: i64 = -10;
pub const TOKS_E_NOMEM: i64 = -11;

// ---- limits ----------------------------------------------------------------------------------------------

pub const TOKS_MAX_TEXT: u64 = 1 << 29;
pub const TOKS_MAX_TOKEN_BYTES: u32 = 65535;
pub const TOKS_MAX_ADDED_BYTES: u32 = 255;
pub const TOKS_MAX_IDS: u32 = (1 << 21) - 1;
pub const TOKS_MAX_SOURCE_BYTES: u64 = 256 << 20;

// ---- types -----------------------------------------------------------------------------------------------

/// Opaque; created by `toks_load*`, freed by `toks_unload`.
#[repr(C)]
pub struct toks_ctx {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct toks_diag {
    pub code: i64,
    pub what: [c_char; 248],
}

pub const TOKS_TIER_AUTO: u32 = 0;
pub const TOKS_TIER_SCALAR: u32 = 1;
pub const TOKS_TIER_NEON: u32 = 2;
pub const TOKS_TIER_AVX2: u32 = 3;
pub const TOKS_TIER_AVX512: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct toks_load_opts {
    pub size: u32,
    pub tier: u32,
    pub flags: u32,
    pub rsv: u32,
    pub diag: *mut toks_diag,
}

// ---- scratch ---------------------------------------------------------------------------------------------

pub const TOKS_SCRATCH_MEMO_SET: u32 = 1 << 20;

pub const fn TOKS_SCRATCH_MEMO_MIB(m: u32) -> u32 {
    TOKS_SCRATCH_MEMO_SET | (m & 0xFFF)
}

pub const fn TOKS_SCRATCH_CACHE_MIB(n: u32) -> u32 {
    (n & 0xFF) << 12
}

// ---- encode ----------------------------------------------------------------------------------------------

pub const TOKS_ADDED_ALL: u32 = 0;
pub const TOKS_ADDED_NONSPECIAL: u32 = 1;
pub const TOKS_ADDED_NONE: u32 = 2;
pub const TOKS_ADDED_MASK: u32 = 3;
pub const TOKS_NO_POSTPROCESS: u32 = 4;
pub const TOKS_CONTINUATION: u32 = 8;
pub const TOKS_NO_TRUNCATE: u32 = 16;
pub const TOKS_NO_PAD: u32 = 32;

// ---- decode ----------------------------------------------------------------------------------------------

pub const TOKS_SKIP_SPECIAL: u32 = 1;
pub const TOKS_DECODE_RAW: u32 = 2;

// ---- vocabulary ------------------------------------------------------------------------------------------

pub const TOKS_ID_ADDED: u32 = 1;
pub const TOKS_ID_SPECIAL: u32 = 2;
pub const TOKS_ID_BYTE: u32 = 4;
pub const TOKS_ID_LSTRIP: u32 = 8;
pub const TOKS_ID_RSTRIP: u32 = 16;
pub const TOKS_ID_SINGLE_WORD: u32 = 32;
pub const TOKS_ID_NORMALIZED: u32 = 64;

// ---- stream decode ---------------------------------------------------------------------------------------

/// A copy shares the original's hold (`toks_stream_hold`): never push to both.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct toks_stream {
    pub opaque: [u64; 8],
}

// ---- info ------------------------------------------------------------------------------------------------

pub const TOKS_ALGO_BPE_BYTELEVEL: u32 = 1;
pub const TOKS_ALGO_BPE_SPM: u32 = 2;
pub const TOKS_ALGO_UNIGRAM: u32 = 3;
pub const TOKS_ALGO_WORDPIECE: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct toks_info {
    pub size: u32,
    pub abi_major: u32,
    pub abi_minor: u32,
    pub algorithm: u32,
    pub tier: u32,
    pub n_ids: u32,
    pub n_added: u32,
    pub paths: u32,
    pub cpu_features: u64,
    pub max_text: u64,
    pub control_isolation: u32,
    pub rsv: u32,
    pub source_sha256: [u8; 32],
    pub image_sha256: [u8; 32],
    pub name: [c_char; 64],
    pub trunc_on: u32,
    pub trunc_max: u32,
    pub trunc_stride: u32,
    pub pad_on: u32,
    pub pad_fixed: u32,
    pub pad_id: u32,
    pub pad_type_id: u32,
    pub pad_len: u32,
    pub pad_multiple: u32,
    pub pad_left: u32,
    pub n_template_prefix: u32,
    pub n_template_suffix: u32,
    pub seq_type_id: u32,
    pub rsv2: u32,
}

pub const TOKS_PATH_SCAN: u32 = 1;
pub const TOKS_PATH_NORMALIZE: u32 = 2;

// ---- toks_par --------------------------------------------------------------------------------------------

/// Opaque; created by `toks_par_create`, freed by `toks_par_destroy`.
#[repr(C)]
pub struct toks_par {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct toks_par_item {
    pub text: *const c_void,
    pub len: u64,
    pub out: *mut u32,
    pub cap: u64,
    pub n: i64,
}

pub const TOKS_PAR_HAS_INFO: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct toks_par_info {
    pub size: u32,
    pub threads: u32,
    pub fast: u32,
    pub last: u32,
    pub ns_per_mib: u64,
    pub wake_ns: u64,
    pub join_ns: u64,
    pub min_bytes: u64,
}

// the layouts toks.h fixes (toks_info: abi 0.3's size is offsetof(toks_info, trunc_on) == 184)
const _: () = {
    use core::mem::{offset_of, size_of};
    assert!(size_of::<toks_diag>() == 256);
    assert!(size_of::<toks_load_opts>() == 24);
    assert!(size_of::<toks_stream>() == 64);
    assert!(offset_of!(toks_info, trunc_on) == 184);
    assert!(size_of::<toks_info>() == 240);
    assert!(size_of::<toks_par_item>() == 40);
    assert!(size_of::<toks_par_info>() == 48);
};

unsafe extern "C" {
    // ---- loading
    pub fn toks_load(out: *mut *mut toks_ctx, path: *const c_char, opts: *const toks_load_opts) -> i64;
    pub fn toks_load_mem_copy(
        out: *mut *mut toks_ctx,
        data: *const c_void,
        len: u64,
        opts: *const toks_load_opts,
    ) -> i64;
    pub fn toks_unload(ctx: *mut toks_ctx);

    // ---- scratch
    pub fn toks_scratch_bytes(ctx: *const toks_ctx, max_len: u64, flags: u32) -> u64;
    pub fn toks_scratch_init(ctx: *const toks_ctx, scr: *mut c_void, bytes: u64, flags: u32) -> i64;

    // ---- encode
    pub fn toks_encode(
        ctx: *const toks_ctx,
        text: *const c_void,
        len: u64,
        flags: u32,
        out: *mut u32,
        cap: u64,
        scr: *mut c_void,
    ) -> i64;
    pub fn toks_pieces(
        ctx: *const toks_ctx,
        text: *const c_void,
        len: u64,
        flags: u32,
        ends: *mut u32,
        cap: u64,
        scr: *mut c_void,
    ) -> i64;
    pub fn toks_template(
        ctx: *const toks_ctx,
        ids: *mut u32,
        type_ids: *mut u32,
        cap: u64,
        n_prefix: *mut u32,
    ) -> i64;
    pub fn toks_encode_bound(ctx: *const toks_ctx, len: u64) -> u64;

    // ---- split planning
    pub fn toks_split_points(
        ctx: *const toks_ctx,
        text: *const c_void,
        len: u64,
        flags: u32,
        n_want: u32,
        offs: *mut u64,
        cap: u64,
        scr: *mut c_void,
    ) -> i64;

    // ---- decode
    pub fn toks_decode(ctx: *const toks_ctx, ids: *const u32, n: u64, flags: u32, out: *mut u8, cap: u64) -> i64;
    pub fn toks_token(ctx: *const toks_ctx, id: u32, len: *mut u64) -> *const u8;

    // ---- vocabulary
    pub fn toks_token_to_id(ctx: *const toks_ctx, s: *const c_void, len: u64) -> i64;
    pub fn toks_id_flags(ctx: *const toks_ctx, id: u32) -> i64;
    pub fn toks_added(
        ctx: *const toks_ctx,
        i: u32,
        content: *mut *const c_void,
        len: *mut u64,
        id: *mut u32,
    ) -> i64;

    // ---- stream decode
    pub fn toks_stream_init(ctx: *const toks_ctx, st: *mut toks_stream, flags: u32);
    pub fn toks_stream_bound(ctx: *const toks_ctx, n_ids: u64) -> u64;
    pub fn toks_stream_push(
        ctx: *const toks_ctx,
        st: *mut toks_stream,
        ids: *const u32,
        n: u64,
        out: *mut u8,
        cap: u64,
    ) -> i64;
    pub fn toks_stream_flush(ctx: *const toks_ctx, st: *mut toks_stream, out: *mut u8, cap: u64) -> i64;
    pub fn toks_stream_hold(ctx: *const toks_ctx, st: *mut toks_stream, hold: *mut c_void, cap: u64) -> i64;

    // ---- info
    pub fn toks_get_info(ctx: *const toks_ctx, out: *mut toks_info) -> i64;
    pub fn toks_version() -> *const c_char;

    // ---- toks_par
    pub fn toks_par_create(out: *mut *mut toks_par, ctx: *const toks_ctx, n_threads: u32, scratch_flags: u32) -> i64;
    pub fn toks_par_destroy(par: *mut toks_par);
    pub fn toks_par_encode_batch(par: *mut toks_par, items: *mut toks_par_item, n_items: u64, flags: u32) -> i64;
    pub fn toks_par_encode(
        par: *mut toks_par,
        text: *const c_void,
        len: u64,
        flags: u32,
        out: *mut u32,
        cap: u64,
    ) -> i64;
    pub fn toks_par_get_info(par: *const toks_par, out: *mut toks_par_info) -> i64;
}
