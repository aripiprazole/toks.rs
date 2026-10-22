use std::ptr::{self, NonNull};

use crate::error::check;
use crate::{EncodeFlags, Error, Result, ScratchFlags, Tokenizer, check_len, sys};

/// A small persistent pool of worker threads (`toks_par`) encoding batches, or one big text split at
/// certified cuts, with every result equal to [`Tokenizer::encode`]'s. The calling thread is one of the
/// participants; a call one core finishes faster (and any call under 16 KiB) wakes no thread.
///
/// One call at a time runs: concurrent calls from several threads wait for each other.
pub struct Pool<'t> {
    par: NonNull<sys::toks_par>,
    tok: &'t Tokenizer,
}

// SAFETY: toks_par serializes its calls with its own lock (toks_par_get_info included) and only reads the
// shared context, which outlives the pool ('t)
unsafe impl Send for Pool<'_> {}
unsafe impl Sync for Pool<'_> {}

impl Drop for Pool<'_> {
    fn drop(&mut self) {
        // SAFETY: par came from toks_par_create; no call is running (&mut self); this joins the workers
        unsafe { sys::toks_par_destroy(self.par.as_ptr()) }
    }
}

/// What a [`Pool`] measured and decides by (`toks_par_info`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolInfo {
    /// Participants at most, the caller included.
    pub threads: u32,
    /// The fast cpus the process may run on.
    pub fast: u32,
    /// Participants of the last call.
    pub last: u32,
    /// The measured encode cost on this host and tokenizer.
    pub ns_per_mib: u64,
    /// The measured cost of one more participant woken from sleep.
    pub wake_ns: u64,
    /// The same for a worker still spinning after its last call.
    pub join_ns: u64,
    /// The smallest call that would take a second participant now (`None`: never).
    pub min_bytes: Option<u64>,
}

impl<'t> Pool<'t> {
    /// At most `n_threads` participants, the caller included (0: a couple, at most 4), each with a scratch
    /// made with `scratch_flags`.
    pub fn new(tok: &'t Tokenizer, n_threads: u32, scratch_flags: ScratchFlags) -> Result<Self> {
        let mut par = ptr::null_mut();
        // SAFETY: out is valid; the context outlives the pool
        check(unsafe { sys::toks_par_create(&mut par, tok.as_ptr(), n_threads, scratch_flags.bits()) })?;
        let par = NonNull::new(par).ok_or_else(|| Error::new(sys::TOKS_E_NOMEM, "no pool".to_string()))?;
        Ok(Self { par, tok })
    }

    pub fn tokenizer(&self) -> &'t Tokenizer {
        self.tok
    }

    pub fn info(&self) -> Result<PoolInfo> {
        let mut raw = sys::toks_par_info { size: size_of::<sys::toks_par_info>() as u32, ..Default::default() };
        // SAFETY: par is live; raw is sizeof(toks_par_info) bytes
        check(unsafe { sys::toks_par_get_info(self.par.as_ptr(), &mut raw) })?;
        Ok(PoolInfo {
            threads: raw.threads,
            fast: raw.fast,
            last: raw.last,
            ns_per_mib: raw.ns_per_mib,
            wake_ns: raw.wake_ns,
            join_ns: raw.join_ns,
            min_bytes: (raw.min_bytes != 0).then_some(raw.min_bytes),
        })
    }

    /// One text across the pool: the same ids as [`Tokenizer::encode`].
    pub fn encode(&self, text: impl AsRef<[u8]>, flags: EncodeFlags) -> Result<Vec<u32>> {
        let text = text.as_ref();
        check_len(text.len())?; // before sizing the output by it
        let mut ids = vec![0u32; self.tok.ids_guess(text.len())];
        for _ in 0..2 {
            let n = self.encode_into(text, flags, &mut ids)?;
            if n <= ids.len() {
                ids.truncate(n);
                shrink(&mut ids);
                return Ok(ids);
            }
            ids.resize(n, 0);
        }
        Err(Error::new(sys::TOKS_E_CAP, "encode: the output grew between two passes".to_string()))
    }

    /// As [`Tokenizer::encode_into`]: the total number of ids, `out[..min(n, out.len())]` the exact prefix.
    pub fn encode_into(&self, text: impl AsRef<[u8]>, flags: EncodeFlags, out: &mut [u32]) -> Result<usize> {
        let text = text.as_ref();
        // SAFETY: text and out are valid for their lengths and do not overlap; the pool serializes calls
        check(unsafe {
            sys::toks_par_encode(
                self.par.as_ptr(),
                text.as_ptr().cast(),
                text.len() as u64,
                flags.bits(),
                out.as_mut_ptr(),
                out.len() as u64,
            )
        })
    }

    /// Every text's ids, the texts spread over the pool. The first text that fails fails the call.
    pub fn encode_batch<T: AsRef<[u8]>>(&self, texts: &[T], flags: EncodeFlags) -> Result<Vec<Vec<u32>>> {
        texts.iter().try_for_each(|t| check_len(t.as_ref().len()))?; // before sizing the outputs by them
        let mut outs: Vec<Vec<u32>> = texts.iter().map(|t| vec![0u32; self.tok.ids_guess(t.as_ref().len())]).collect();
        let mut short: Vec<usize> = (0..texts.len()).collect();
        for _ in 0..2 {
            let mut items: Vec<sys::toks_par_item> =
                short.iter().map(|&i| item(texts[i].as_ref(), &mut outs[i])).collect();
            // SAFETY: each item points at its own text and its own Vec, alive and untouched until the call returns
            unsafe { self.run_batch(&mut items, flags) }?;
            let mut again = Vec::new();
            for (&i, item) in short.iter().zip(&items) {
                let n = check(item.n)?;
                if n <= outs[i].len() {
                    outs[i].truncate(n);
                    shrink(&mut outs[i]);
                } else {
                    outs[i].resize(n, 0);
                    again.push(i);
                }
            }
            if again.is_empty() {
                return Ok(outs);
            }
            short = again;
        }
        Err(Error::new(sys::TOKS_E_CAP, "encode_batch: the output grew between two passes".to_string()))
    }

    /// Each text encoded into its own output, as [`encode_into`](Self::encode_into) would: every item's
    /// total (or its error). `texts` and `outs` have the same length.
    pub fn encode_batch_into<T: AsRef<[u8]>>(
        &self,
        texts: &[T],
        outs: &mut [&mut [u32]],
        flags: EncodeFlags,
    ) -> Result<Vec<Result<usize>>> {
        if texts.len() != outs.len() {
            return Err(Error::new(
                sys::TOKS_E_ARG,
                format!("encode_batch_into: {} texts, {} outputs", texts.len(), outs.len()),
            ));
        }
        let mut items: Vec<sys::toks_par_item> =
            texts.iter().zip(outs.iter_mut()).map(|(text, out)| item(text.as_ref(), out)).collect();
        // SAFETY: the texts are borrowed and the outs mutably borrowed (so disjoint) for the whole call
        unsafe { self.run_batch(&mut items, flags) }?;
        Ok(items.iter().map(|item| check(item.n)).collect())
    }

    /// # Safety
    /// Every item's text and out are valid for their lengths, no two outs overlap, and no out overlaps a text.
    unsafe fn run_batch(&self, items: &mut [sys::toks_par_item], flags: EncodeFlags) -> Result<()> {
        // SAFETY: the caller's; the pool serializes calls
        check(unsafe {
            sys::toks_par_encode_batch(self.par.as_ptr(), items.as_mut_ptr(), items.len() as u64, flags.bits())
        })
        .map(drop)
    }
}

fn item(text: &[u8], out: &mut [u32]) -> sys::toks_par_item {
    sys::toks_par_item {
        text: text.as_ptr().cast(),
        len: text.len() as u64,
        out: out.as_mut_ptr(),
        cap: out.len() as u64,
        n: 0,
    }
}

impl std::fmt::Debug for Pool<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool").field("tokenizer", self.tok).finish_non_exhaustive()
    }
}

/// Gives back a first guess's spare room when it was far too big.
fn shrink(ids: &mut Vec<u32>) {
    if ids.capacity() / 2 > ids.len() {
        ids.shrink_to_fit();
    }
}
