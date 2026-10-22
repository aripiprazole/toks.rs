//! Rust bindings for [toks](https://github.com/actual-computer/toks): exact Hugging Face `tokenizers` and
//! tiktoken ids, as fast as the hardware allows.
//!
//! ```no_run
//! use toks::{EncodeFlags, Tokenizer};
//!
//! let tok = Tokenizer::from_file("tokenizer.json")?;
//! let ids = tok.encode("hello world", EncodeFlags::DEFAULT)?;
//! assert_eq!(tok.decode(&ids, false)?, "hello world");
//! # Ok::<(), toks::Error>(())
//! ```
//!
//! A [`Tokenizer`] is read-only after load: share it across threads (`&Tokenizer`, `Arc<Tokenizer>`). Encoding
//! needs a per-thread scratch: [`Tokenizer::encode`] takes one from the tokenizer's own pool, a [`Scratch`] is
//! one you keep (no lock, warm caches), and a [`Pool`] spreads a batch or one big text over worker threads.
//! The C API itself is [`sys`].

#[cfg(not(target_pointer_width = "64"))]
compile_error!("toks supports arm64 and x86_64 targets only");

mod error;
mod flags;
mod par;
mod stream;
pub mod sys;

use std::ffi::{CString, c_void};
use std::path::Path;
use std::ptr::{self, NonNull};
use std::sync::{Mutex, PoisonError};
use std::{mem, slice};

use crate::error::check;
pub use crate::error::{Error, ErrorKind, Result};
pub use crate::flags::{DecodeFlags, EncodeFlags, IdFlags, ScratchFlags};
pub use crate::par::{Pool, PoolInfo};
pub use crate::stream::DecodeStream;

/// The most bytes one encode call takes (`TOKS_MAX_TEXT`).
pub const MAX_TEXT: usize = sys::TOKS_MAX_TEXT as usize;

/// The abi these bindings are written against (`TOKS_ABI_MAJOR`, `TOKS_ABI_MINOR`).
pub const ABI: (u32, u32) = (sys::TOKS_ABI_MAJOR, sys::TOKS_ABI_MINOR);

/// `TOKS_VERSION` of the linked library.
pub fn version() -> &'static str {
    // SAFETY: toks_version returns a static NUL-terminated string
    unsafe { std::ffi::CStr::from_ptr(sys::toks_version()) }.to_str().unwrap_or("")
}

/// A kernel tier (`TOKS_TIER_*`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Tier {
    /// The fastest the machine supports; also honours the `TOKS_TIER` environment variable.
    #[default]
    Auto,
    /// Portable C.
    Scalar,
    /// arm64 NEON.
    Neon,
    /// x86-64 AVX2.
    Avx2,
    /// x86-64 AVX-512.
    Avx512,
    Other(u32),
}

impl Tier {
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            sys::TOKS_TIER_AUTO => Self::Auto,
            sys::TOKS_TIER_SCALAR => Self::Scalar,
            sys::TOKS_TIER_NEON => Self::Neon,
            sys::TOKS_TIER_AVX2 => Self::Avx2,
            sys::TOKS_TIER_AVX512 => Self::Avx512,
            other => Self::Other(other),
        }
    }

    pub fn raw(self) -> u32 {
        match self {
            Self::Auto => sys::TOKS_TIER_AUTO,
            Self::Scalar => sys::TOKS_TIER_SCALAR,
            Self::Neon => sys::TOKS_TIER_NEON,
            Self::Avx2 => sys::TOKS_TIER_AVX2,
            Self::Avx512 => sys::TOKS_TIER_AVX512,
            Self::Other(raw) => raw,
        }
    }
}

/// The tokenizer's model (`TOKS_ALGO_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Algorithm {
    /// Byte-level BPE (gpt-2 alphabet).
    BpeByteLevel,
    /// Sentencepiece-style BPE (metaspace, byte fallback).
    BpeSpm,
    Unigram,
    WordPiece,
    Other(u32),
}

impl Algorithm {
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            sys::TOKS_ALGO_BPE_BYTELEVEL => Self::BpeByteLevel,
            sys::TOKS_ALGO_BPE_SPM => Self::BpeSpm,
            sys::TOKS_ALGO_UNIGRAM => Self::Unigram,
            sys::TOKS_ALGO_WORDPIECE => Self::WordPiece,
            other => Self::Other(other),
        }
    }
}

/// What `toks_get_info` reports about a loaded tokenizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Info {
    pub abi: (u32, u32),
    pub algorithm: Algorithm,
    /// The tier in use.
    pub tier: Tier,
    /// Every id is below this.
    pub n_ids: u32,
    pub n_added: u32,
    /// `TOKS_PATH_*` bits: the stages running on compiled fast paths.
    pub paths: u32,
    pub cpu_features: u64,
    pub max_text: u64,
    /// Control-token isolation is certified for this tokenizer.
    pub control_isolation: bool,
    pub source_sha256: [u8; 32],
    pub image_sha256: [u8; 32],
    pub name: String,
    /// The file's truncation, which encode applies unless [`EncodeFlags::NO_TRUNCATE`].
    pub truncation: Option<Truncation>,
    /// The file's padding, which encode applies unless [`EncodeFlags::NO_PAD`].
    pub padding: Option<Padding>,
    /// The template's ids before the text's.
    pub template_prefix: u32,
    /// The template's ids after the text's.
    pub template_suffix: u32,
    /// The type id of the text's own ids.
    pub seq_type_id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Truncation {
    /// The template's ids count in it.
    pub max_length: u32,
    pub stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Padding {
    /// `Some(length)` for Fixed; `None` for BatchLongest, which pads a batch (the caller's step).
    pub fixed_length: Option<u32>,
    /// `pad_to_multiple_of`.
    pub multiple: Option<u32>,
    pub id: u32,
    pub type_id: u32,
    pub left: bool,
}

impl Info {
    fn from_raw(raw: &sys::toks_info) -> Self {
        Self {
            abi: (raw.abi_major, raw.abi_minor),
            algorithm: Algorithm::from_raw(raw.algorithm),
            tier: Tier::from_raw(raw.tier),
            n_ids: raw.n_ids,
            n_added: raw.n_added,
            paths: raw.paths,
            cpu_features: raw.cpu_features,
            max_text: raw.max_text,
            control_isolation: raw.control_isolation != 0,
            source_sha256: raw.source_sha256,
            image_sha256: raw.image_sha256,
            name: c_chars(&raw.name),
            truncation: (raw.trunc_on != 0)
                .then_some(Truncation { max_length: raw.trunc_max, stride: raw.trunc_stride }),
            padding: (raw.pad_on != 0).then_some(Padding {
                fixed_length: (raw.pad_fixed != 0).then_some(raw.pad_len),
                multiple: (raw.pad_multiple != 0).then_some(raw.pad_multiple),
                id: raw.pad_id,
                type_id: raw.pad_type_id,
                left: raw.pad_left != 0,
            }),
            template_prefix: raw.n_template_prefix,
            template_suffix: raw.n_template_suffix,
            seq_type_id: raw.seq_type_id,
        }
    }
}

/// The post-processor's single-sequence template (`toks_template`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Template {
    /// The ids before the text's, then the ids after them.
    pub ids: Vec<u32>,
    pub type_ids: Vec<u32>,
    /// How many of `ids` come before the text's.
    pub n_prefix: usize,
}

impl Template {
    pub fn prefix(&self) -> &[u32] {
        &self.ids[..self.n_prefix]
    }

    pub fn suffix(&self) -> &[u32] {
        &self.ids[self.n_prefix..]
    }
}

/// One added token (`toks_added`), in id order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddedToken<'a> {
    pub id: u32,
    /// The content as the file writes it (for the last listing of this id).
    pub content: &'a [u8],
    /// `ADDED`, `SPECIAL` as this listing says, `BYTE`, and the listing's options.
    pub flags: IdFlags,
}

/// The added tokens of a [`Tokenizer`], hf's `get_added_tokens_decoder()`.
pub struct AddedTokens<'a> {
    tok: &'a Tokenizer,
    i: u32,
}

impl<'a> Iterator for AddedTokens<'a> {
    type Item = AddedToken<'a>;

    fn next(&mut self) -> Option<AddedToken<'a>> {
        let (mut content, mut len, mut id) = (ptr::null(), 0u64, 0u32);
        // SAFETY: valid out pointers; TOKS_E_ARG past the last one
        let r = unsafe { sys::toks_added(self.tok.ctx(), self.i, &mut content, &mut len, &mut id) };
        if r < 0 {
            return None;
        }
        self.i += 1;
        let content = if len == 0 {
            &[][..]
        } else {
            // SAFETY: content[0, len) lives as long as the context
            unsafe { slice::from_raw_parts(content.cast::<u8>(), len as usize) }
        };
        Some(AddedToken { id, content, flags: IdFlags::from_bits_retain(r as u32) })
    }
}

/// Load options.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadOptions {
    pub tier: Tier,
    /// The flags of the scratches the tokenizer makes: its own pool's ([`Tokenizer::encode`]),
    /// [`Tokenizer::scratch`]'s and [`Tokenizer::pool`]'s.
    pub scratch_flags: ScratchFlags,
}

/// A loaded tokenizer (`toks_ctx`).
pub struct Tokenizer {
    ctx: NonNull<sys::toks_ctx>,
    info: Info,
    scratch_flags: ScratchFlags,
    slots: Mutex<Vec<Slot>>,
}

// SAFETY: a context is read-only after load and may be shared by any number of threads (toks.h); the slots
// are behind a mutex
unsafe impl Send for Tokenizer {}
unsafe impl Sync for Tokenizer {}

impl Drop for Tokenizer {
    fn drop(&mut self) {
        // SAFETY: the context came from toks_load*, and every borrower (Scratch, DecodeStream, Pool) is gone
        unsafe { sys::toks_unload(self.ctx.as_ptr()) }
    }
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer").field("name", &self.info.name).field("n_ids", &self.info.n_ids).finish()
    }
}

impl Tokenizer {
    /// A `tokenizer.json`, a tiktoken model's `tiktoken.model` / `qwen.tiktoken` (its other files beside it), or
    /// a model directory holding one of them.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_file_with(path, LoadOptions::default())
    }

    pub fn from_file_with(path: impl AsRef<Path>, opts: LoadOptions) -> Result<Self> {
        let path = path_to_c(path.as_ref())?;
        // SAFETY: path is NUL-terminated; out and opts are valid
        Self::load(opts, |out, o| unsafe { sys::toks_load(out, path.as_ptr(), o) })
    }

    /// The bytes of a tokenizer file, copied into memory the tokenizer owns.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Self::from_bytes_with(data, LoadOptions::default())
    }

    pub fn from_bytes_with(data: &[u8], opts: LoadOptions) -> Result<Self> {
        // SAFETY: data[0, len) is valid; out and opts are valid
        Self::load(opts, |out, o| unsafe { sys::toks_load_mem_copy(out, data.as_ptr().cast(), data.len() as u64, o) })
    }

    fn load(
        opts: LoadOptions,
        load: impl FnOnce(*mut *mut sys::toks_ctx, *const sys::toks_load_opts) -> i64,
    ) -> Result<Self> {
        let mut diag = sys::toks_diag { code: 0, what: [0; 248] };
        let raw_opts = sys::toks_load_opts {
            size: mem::size_of::<sys::toks_load_opts>() as u32,
            tier: opts.tier.raw(),
            flags: 0,
            rsv: 0,
            diag: &mut diag,
        };
        let mut ctx = ptr::null_mut();
        let r = load(&mut ctx, &raw_opts);
        if r < 0 {
            let what = c_chars(&diag.what);
            return Err(Error::new(r, (!what.is_empty()).then_some(what)));
        }
        let ctx = NonNull::new(ctx).ok_or_else(|| Error::new(sys::TOKS_E_ARG, "no context".to_string()))?;

        // SAFETY: toks_info is integers and arrays only
        let mut raw: sys::toks_info = unsafe { mem::zeroed() };
        raw.size = mem::size_of::<sys::toks_info>() as u32;
        // SAFETY: ctx is live, raw is sizeof(toks_info) bytes
        let r = unsafe { sys::toks_get_info(ctx.as_ptr(), &mut raw) };
        if r < 0 || raw.abi_major != ABI.0 || raw.abi_minor < ABI.1 {
            // SAFETY: ctx is ours and unused
            unsafe { sys::toks_unload(ctx.as_ptr()) };
            let detail = format!("linked toks {} does not speak abi {}.{}", version(), ABI.0, ABI.1);
            return Err(Error::new(if r < 0 { r } else { sys::TOKS_E_UNSUPPORTED }, detail));
        }
        Ok(Self { ctx, info: Info::from_raw(&raw), scratch_flags: opts.scratch_flags, slots: Mutex::new(Vec::new()) })
    }

    /// The `toks_ctx`, for calls into [`sys`]; it lives as long as `self`.
    pub fn as_ptr(&self) -> *const sys::toks_ctx {
        self.ctx.as_ptr()
    }

    fn ctx(&self) -> *const sys::toks_ctx {
        self.ctx.as_ptr()
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    /// The most ids an encode of `len` bytes can return, under any flags.
    pub fn encode_bound(&self, len: usize) -> usize {
        // SAFETY: ctx is live
        unsafe { sys::toks_encode_bound(self.ctx(), len as u64) as usize }
    }

    /// A first output size for an encode of `len` bytes: the bound where it is tight, else room for the text's
    /// bytes and the file's padding (a longer result costs one more pass).
    fn ids_guess(&self, len: usize) -> usize {
        let mut guess = len.saturating_add(64);
        if let Some(pad) = self.info.padding {
            guess = guess.max(pad.fixed_length.unwrap_or(0) as usize);
            if let Some(m) = pad.multiple {
                guess = guess.next_multiple_of(m as usize);
            }
        }
        guess.min(self.encode_bound(len))
    }

    /// The ids of `text` (exactly hf `encode(text)` under [`EncodeFlags::DEFAULT`]), with a scratch from the
    /// tokenizer's pool.
    pub fn encode(&self, text: impl AsRef<[u8]>, flags: EncodeFlags) -> Result<Vec<u32>> {
        self.with_slot(|slot| slot.run_vec(self, self.scratch_flags, sys::toks_encode, text.as_ref(), flags))
    }

    /// Encodes into `out` and returns the total number of ids `n`: `out[..min(n, out.len())]` holds the exact
    /// prefix, so `n > out.len()` means `out` was short. [`encode_bound`](Self::encode_bound) ids are never short.
    pub fn encode_into(&self, text: impl AsRef<[u8]>, flags: EncodeFlags, out: &mut [u32]) -> Result<usize> {
        self.with_slot(|slot| {
            slot.reserve(self, text.as_ref().len(), self.scratch_flags)?;
            slot.run(self, sys::toks_encode, text.as_ref(), flags, out)
        })
    }

    /// The pieces the model sees, as end offsets into `text` (or its normalized form, where a normalizer such
    /// as NFC changed it). A view for inspection: to cut text, use [`split_points`](Self::split_points).
    pub fn pieces(&self, text: impl AsRef<[u8]>, flags: EncodeFlags) -> Result<Vec<u32>> {
        self.with_slot(|slot| slot.run_vec(self, self.scratch_flags, sys::toks_pieces, text.as_ref(), flags))
    }

    /// Up to `n_want - 1` certified cuts of `text`, strictly increasing offsets in `(0, len)`. Encoding the
    /// parts (every part after the first with [`EncodeFlags::CONTINUATION`]) under `flags` with
    /// [`EncodeFlags::NO_POSTPROCESS`] and concatenating gives the whole text's ids. A tokenizer family
    /// without certified rules gives none.
    pub fn split_points(&self, text: impl AsRef<[u8]>, flags: EncodeFlags, n_want: u32) -> Result<Vec<usize>> {
        let text = text.as_ref();
        let mut offs = vec![0u64; (n_want.saturating_sub(1) as usize).min(text.len().saturating_sub(1))];
        // SAFETY: text and offs are valid for their lengths; scr is reserved and may be NULL
        let n = check(unsafe {
            sys::toks_split_points(
                self.ctx(),
                text.as_ptr().cast(),
                text.len() as u64,
                flags.bits(),
                n_want,
                offs.as_mut_ptr(),
                offs.len() as u64,
                ptr::null_mut(),
            )
        })?;
        Ok(offs[..n.min(offs.len())].iter().map(|&o| o as usize).collect())
    }

    /// A scratch of your own, for texts of up to `max_len` bytes to begin with.
    pub fn scratch(&self, max_len: usize) -> Result<Scratch<'_>> {
        Scratch::new(self, max_len, self.scratch_flags)
    }

    /// A pool of at most `n_threads` participants, the caller included (0: a couple, at most 4).
    pub fn pool(&self, n_threads: u32) -> Result<Pool<'_>> {
        Pool::new(self, n_threads, self.scratch_flags)
    }

    /// The text of `ids` (hf `decode`). Bytes that are not UTF-8 come out as U+FFFD, as in hf.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let flags = if skip_special_tokens { DecodeFlags::SKIP_SPECIAL } else { DecodeFlags::DEFAULT };
        let bytes = self.decode_bytes(ids, flags)?;
        Ok(String::from_utf8(bytes).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned()))
    }

    /// The bytes of `ids`; with [`DecodeFlags::RAW`], never repaired (tiktoken's `decode_bytes`).
    pub fn decode_bytes(&self, ids: &[u32], flags: DecodeFlags) -> Result<Vec<u8>> {
        let mut out = vec![0u8; ids.len().saturating_mul(8).saturating_add(16)];
        for _ in 0..2 {
            let n = self.decode_into(ids, flags, &mut out)?;
            if n <= out.len() {
                out.truncate(n);
                return Ok(out);
            }
            out.resize(n, 0);
        }
        Err(Error::new(sys::TOKS_E_CAP, "decode: the output grew between two passes".to_string()))
    }

    /// Decodes into `out` and returns the total number of bytes `n`, `out[..min(n, out.len())]` the exact prefix.
    pub fn decode_into(&self, ids: &[u32], flags: DecodeFlags, out: &mut [u8]) -> Result<usize> {
        // SAFETY: ids and out are valid for their lengths and do not overlap
        check(unsafe {
            sys::toks_decode(self.ctx(), ids.as_ptr(), ids.len() as u64, flags.bits(), out.as_mut_ptr(), out.len() as u64)
        })
    }

    /// An incremental decoder (hf `DecodeStream`).
    pub fn decode_stream(&self, skip_special_tokens: bool) -> DecodeStream<'_> {
        DecodeStream::new(self, skip_special_tokens)
    }

    /// The raw bytes of one id, or `None` if it has no string.
    pub fn token(&self, id: u32) -> Option<&[u8]> {
        let mut len = 0u64;
        // SAFETY: ctx is live, len is a valid out pointer
        let p = unsafe { sys::toks_token(self.ctx(), id, &mut len) };
        // SAFETY: p[0, len) lives as long as the context
        (!p.is_null()).then(|| unsafe { slice::from_raw_parts(p, len as usize) })
    }

    /// The id whose string is these bytes: the bytes an id decodes to ([`token`](Self::token)'s), not hf's
    /// vocabulary spelling (`" hello"`, not `"Ġhello"`). A sentencepiece piece is as written (`"▁hello"`), an
    /// added token by its content.
    pub fn token_to_id(&self, token: impl AsRef<[u8]>) -> Option<u32> {
        let token = token.as_ref();
        // SAFETY: token is valid for its length
        let r = unsafe { sys::toks_token_to_id(self.ctx(), token.as_ptr().cast(), token.len() as u64) };
        (r >= 0).then_some(r as u32)
    }

    /// An id's `ADDED` / `SPECIAL` / `BYTE` flags, or `None` beyond the table.
    pub fn id_flags(&self, id: u32) -> Option<IdFlags> {
        // SAFETY: ctx is live
        let r = unsafe { sys::toks_id_flags(self.ctx(), id) };
        (r >= 0).then_some(IdFlags::from_bits_retain(r as u32))
    }

    pub fn added_tokens(&self) -> AddedTokens<'_> {
        AddedTokens { tok: self, i: 0 }
    }

    pub fn template(&self) -> Result<Template> {
        let mut n_prefix = 0u32;
        // SAFETY: ids NULL with cap 0 sizes the template
        let n = check(unsafe { sys::toks_template(self.ctx(), ptr::null_mut(), ptr::null_mut(), 0, &mut n_prefix) })?;
        let (mut ids, mut type_ids) = (vec![0u32; n], vec![0u32; n]);
        // SAFETY: ids and type_ids hold n entries
        check(unsafe {
            sys::toks_template(self.ctx(), ids.as_mut_ptr(), type_ids.as_mut_ptr(), n as u64, &mut n_prefix)
        })?;
        Ok(Template { ids, type_ids, n_prefix: n_prefix as usize })
    }

    fn with_slot<R>(&self, f: impl FnOnce(&mut Slot) -> Result<R>) -> Result<R> {
        let mut slot = self.slots.lock().unwrap_or_else(PoisonError::into_inner).pop().unwrap_or_default();
        let r = f(&mut slot);
        slot.trim();
        self.slots.lock().unwrap_or_else(PoisonError::into_inner).push(slot);
        r
    }
}

/// A scratch of your own (`toks_scratch_init`): bound to its tokenizer, used by one call at a time, grown
/// on demand for longer texts. Reusing one keeps its caches warm: a text sent again is answered from its memo.
pub struct Scratch<'t> {
    tok: &'t Tokenizer,
    flags: ScratchFlags,
    slot: Slot,
}

impl<'t> Scratch<'t> {
    pub fn new(tok: &'t Tokenizer, max_len: usize, flags: ScratchFlags) -> Result<Self> {
        check_len(max_len)?;
        let mut slot = Slot::default();
        slot.init(tok, max_len, flags)?;
        Ok(Self { tok, flags, slot })
    }

    pub fn tokenizer(&self) -> &'t Tokenizer {
        self.tok
    }

    /// The longest text the scratch holds now.
    pub fn max_len(&self) -> usize {
        self.slot.max_len
    }

    /// As [`Tokenizer::encode`].
    pub fn encode(&mut self, text: impl AsRef<[u8]>, flags: EncodeFlags) -> Result<Vec<u32>> {
        self.slot.run_vec(self.tok, self.flags, sys::toks_encode, text.as_ref(), flags)
    }

    /// As [`Tokenizer::encode_into`].
    pub fn encode_into(&mut self, text: impl AsRef<[u8]>, flags: EncodeFlags, out: &mut [u32]) -> Result<usize> {
        self.slot.reserve(self.tok, text.as_ref().len(), self.flags)?;
        self.slot.run(self.tok, sys::toks_encode, text.as_ref(), flags, out)
    }

    /// As [`Tokenizer::pieces`].
    pub fn pieces(&mut self, text: impl AsRef<[u8]>, flags: EncodeFlags) -> Result<Vec<u32>> {
        self.slot.run_vec(self.tok, self.flags, sys::toks_pieces, text.as_ref(), flags)
    }
}

impl std::fmt::Debug for Scratch<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scratch").field("max_len", &self.slot.max_len).field("flags", &self.flags).finish()
    }
}

const DEFAULT_TEXT: usize = 64 << 10; // a pool slot's first scratch holds texts of up to 64 KiB
const KEEP_TEXT: usize = 1 << 20; // a pool slot keeps a scratch for texts of up to 1 MiB between calls
const KEEP_IDS: usize = 1 << 20; // ... and staging for up to 1 Mi ids

type EncodeFn = unsafe extern "C" fn(*const sys::toks_ctx, *const c_void, u64, u32, *mut u32, u64, *mut c_void) -> i64;

/// A scratch and the staging its whole-result calls copy out of.
#[derive(Default)]
struct Slot {
    scr: Vec<u8>,
    max_len: usize,
    ids: Vec<u32>,
}

impl Slot {
    fn init(&mut self, tok: &Tokenizer, max_len: usize, flags: ScratchFlags) -> Result<()> {
        self.scr = Vec::new();
        // SAFETY: ctx is live
        let bytes = unsafe { sys::toks_scratch_bytes(tok.ctx(), max_len as u64, flags.bits()) } as usize;
        let mut scr = vec![0u8; bytes];
        // SAFETY: scr is `bytes` long; it moves with the Vec's heap buffer, never with self
        check(unsafe { sys::toks_scratch_init(tok.ctx(), scr.as_mut_ptr().cast(), bytes as u64, flags.bits()) })?;
        self.scr = scr;
        self.max_len = max_len;
        Ok(())
    }

    /// The scratch holds texts of `len` bytes: grown to at least twice its size when it does not.
    fn reserve(&mut self, tok: &Tokenizer, len: usize, flags: ScratchFlags) -> Result<()> {
        check_len(len)?;
        if !self.scr.is_empty() && len <= self.max_len {
            return Ok(());
        }
        let mut want = len.max(DEFAULT_TEXT);
        if !self.scr.is_empty() {
            want = want.max(2 * self.max_len);
        }
        self.init(tok, want.min(MAX_TEXT), flags)
    }

    /// `f` (toks_encode or toks_pieces) into `out`; the scratch holds `text` already.
    fn run(&mut self, tok: &Tokenizer, f: EncodeFn, text: &[u8], flags: EncodeFlags, out: &mut [u32]) -> Result<usize> {
        // SAFETY: text and out are valid for their lengths; the scratch is bound to ctx and holds text.len()
        check(unsafe {
            f(
                tok.ctx(),
                text.as_ptr().cast(),
                text.len() as u64,
                flags.bits(),
                out.as_mut_ptr(),
                out.len() as u64,
                self.scr.as_mut_ptr().cast(),
            )
        })
    }

    /// `f`'s whole result, through the staging buffer: a second pass only when the first guess was short.
    fn run_vec(
        &mut self,
        tok: &Tokenizer,
        scratch_flags: ScratchFlags,
        f: EncodeFn,
        text: &[u8],
        flags: EncodeFlags,
    ) -> Result<Vec<u32>> {
        self.reserve(tok, text.len(), scratch_flags)?;
        let mut want = tok.ids_guess(text.len());
        for _ in 0..2 {
            if self.ids.len() < want {
                self.ids.resize(want, 0);
            }
            let mut ids = mem::take(&mut self.ids);
            let r = self.run(tok, f, text, flags, &mut ids);
            self.ids = ids;
            let n = r?;
            if n <= self.ids.len() {
                return Ok(self.ids[..n].to_vec());
            }
            want = n;
        }
        Err(Error::new(sys::TOKS_E_CAP, "encode: the output grew between two passes".to_string()))
    }

    fn trim(&mut self) {
        if self.max_len > KEEP_TEXT {
            self.scr = Vec::new();
            self.max_len = 0;
        }
        if self.ids.len() > KEEP_IDS {
            self.ids = Vec::new();
        }
    }
}

pub(crate) fn check_len(len: usize) -> Result<()> {
    if len > MAX_TEXT {
        return Err(Error::new(sys::TOKS_E_LIMIT, format!("a text of {len} bytes (the limit is {MAX_TEXT})")));
    }
    Ok(())
}

/// A NUL-terminated `c_char` array as a string, lossy.
fn c_chars(chars: &[std::ffi::c_char]) -> String {
    let bytes: Vec<u8> = chars.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(unix)]
fn path_to_c(path: &Path) -> Result<CString> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| Error::new(sys::TOKS_E_ARG, format!("{}: a path with a NUL byte", path.display())))
}

#[cfg(not(unix))]
fn path_to_c(path: &Path) -> Result<CString> {
    let s = path.to_str().ok_or_else(|| Error::new(sys::TOKS_E_ARG, format!("{}: not UTF-8", path.display())))?;
    CString::new(s).map_err(|_| Error::new(sys::TOKS_E_ARG, format!("{s}: a path with a NUL byte")))
}
