use std::borrow::Cow;

use crate::error::check;
use crate::{Error, MAX_TEXT, Result, Tokenizer, sys};

/// The bytes a stream holds in itself; a longer byte-fallback run needs a hold of the caller's.
const OWN_HOLD: usize = 44;

/// An incremental decoder (hf `DecodeStream`, `toks_stream`): push ids as they come, get the text they
/// complete. A stream belongs to one thread at a time.
///
/// Not `Clone`: a copy of a `toks_stream` shares its hold.
pub struct DecodeStream<'t> {
    tok: &'t Tokenizer,
    st: sys::toks_stream,
    // st's hold when it outgrows its own 44 bytes: a heap buffer never resized in place, so it stays put
    // while self moves, and is replaced only after toks_stream_hold has copied out of it
    hold: Vec<u8>,
    out: Vec<u8>,
}

impl<'t> DecodeStream<'t> {
    pub(crate) fn new(tok: &'t Tokenizer, skip_special_tokens: bool) -> Self {
        let mut st = sys::toks_stream { opaque: [0; 8] };
        let flags = if skip_special_tokens { sys::TOKS_SKIP_SPECIAL } else { 0 };
        // SAFETY: ctx is live, st is a valid toks_stream
        unsafe { sys::toks_stream_init(tok.as_ptr(), &mut st, flags) };
        Self { tok, st, hold: Vec::new(), out: Vec::new() }
    }

    pub fn tokenizer(&self) -> &'t Tokenizer {
        self.tok
    }

    /// The text `ids` complete, possibly empty while a character or a byte-fallback run is unfinished.
    pub fn push(&mut self, ids: &[u32]) -> Result<Cow<'_, str>> {
        self.push_bytes(ids).map(String::from_utf8_lossy)
    }

    /// As [`push`](Self::push), the bytes (always UTF-8).
    pub fn push_bytes(&mut self, ids: &[u32]) -> Result<&[u8]> {
        let n = self.call(Some(ids))?;
        Ok(&self.out[..n])
    }

    /// What the stream still holds, as the end of the text; the stream starts over (and keeps its hold).
    pub fn flush(&mut self) -> Result<Cow<'_, str>> {
        self.flush_bytes().map(String::from_utf8_lossy)
    }

    pub fn flush_bytes(&mut self) -> Result<&[u8]> {
        let n = self.call(None)?;
        Ok(&self.out[..n])
    }

    /// One push (or the flush), with room for anything it writes; a byte-fallback run longer than the hold
    /// grows the hold and the same ids go again.
    fn call(&mut self, ids: Option<&[u32]>) -> Result<usize> {
        let ctx = self.tok.as_ptr();
        let n = ids.map_or(0, <[u32]>::len);
        let mut grown = false;
        loop {
            // SAFETY: ctx is live
            let bound = unsafe { sys::toks_stream_bound(ctx, n as u64) } as usize;
            let bound = bound.saturating_add(3 * self.hold.len()).max(64);
            if self.out.len() < bound {
                self.out.resize(bound, 0);
            }
            // SAFETY: st is bound to ctx and its hold (if any) is self.hold, alive and untouched; out is
            // out.len() bytes and does not overlap the hold
            let r = unsafe {
                match ids {
                    Some(ids) => sys::toks_stream_push(
                        ctx,
                        &mut self.st,
                        ids.as_ptr(),
                        n as u64,
                        self.out.as_mut_ptr(),
                        self.out.len() as u64,
                    ),
                    None => sys::toks_stream_flush(ctx, &mut self.st, self.out.as_mut_ptr(), self.out.len() as u64),
                }
            };
            // only a too-long run: n > TOKS_MAX_TEXT is TOKS_E_LIMIT too, and no hold takes that
            if r != sys::TOKS_E_LIMIT || ids.is_none() || grown || n > MAX_TEXT {
                return check(r);
            }
            grown = true;
            let cur = if self.hold.is_empty() { OWN_HOLD } else { self.hold.len() };
            let mut hold = vec![0u8; 2 * cur + n];
            // SAFETY: hold is hold.len() bytes; the call copies the run out of the current hold, still alive
            let m = unsafe { sys::toks_stream_hold(ctx, &mut self.st, hold.as_mut_ptr().cast(), hold.len() as u64) };
            if m < 0 {
                return Err(Error::new(m, "DecodeStream: growing the hold".to_string()));
            }
            self.hold = hold; // the old hold is free once st has moved out of it
        }
    }
}

impl std::fmt::Debug for DecodeStream<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeStream").field("hold", &self.hold.len()).finish_non_exhaustive()
    }
}
