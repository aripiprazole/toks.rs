use std::ops::{BitOr, BitOrAssign};

use crate::sys;

macro_rules! flags {
    ($(#[$meta:meta])* $name:ident { $($(#[$fmeta:meta])* $flag:ident = $value:expr;)* }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub struct $name(u32);

        impl $name {
            $($(#[$fmeta])* pub const $flag: Self = Self($value);)*

            pub const fn bits(self) -> u32 {
                self.0
            }

            /// Any bits, passed through as they are (the library rejects unknown ones with `TOKS_E_ARG`).
            pub const fn from_bits_retain(bits: u32) -> Self {
                Self(bits)
            }

            pub const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }

            pub const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }
        }

        impl BitOr for $name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                self.union(rhs)
            }
        }

        impl BitOrAssign for $name {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }
    };
}

flags! {
    /// `toks_encode` flags. The default is exactly what hf `encode(text)` returns: every added token
    /// recognized, the post-processor applied, then the file's truncation and padding.
    ///
    /// `ADDED_NONSPECIAL` and `ADDED_NONE` are values of one two-bit mode: use at most one of them.
    EncodeFlags {
        /// Special and non-special added tokens recognized (hf default).
        DEFAULT = sys::TOKS_ADDED_ALL;
        /// Only non-special added tokens recognized (hf `encode_special_tokens=True`).
        ADDED_NONSPECIAL = sys::TOKS_ADDED_NONSPECIAL;
        /// No added token recognized.
        ADDED_NONE = sys::TOKS_ADDED_NONE;
        /// Skip bos / eos / template tokens (hf `add_special_tokens=False`).
        NO_POSTPROCESS = sys::TOKS_NO_POSTPROCESS;
        /// The text continues a document: no start-of-input behaviour, no truncation or padding.
        CONTINUATION = sys::TOKS_CONTINUATION;
        /// The file's truncation is not applied.
        NO_TRUNCATE = sys::TOKS_NO_TRUNCATE;
        /// The file's padding is not applied.
        NO_PAD = sys::TOKS_NO_PAD;
    }
}

flags! {
    /// `toks_decode` flags.
    DecodeFlags {
        DEFAULT = 0;
        /// hf `skip_special_tokens=True`.
        SKIP_SPECIAL = sys::TOKS_SKIP_SPECIAL;
        /// The bytes the ids spell, never repaired with U+FFFD (tiktoken's `decode_bytes`). Not a stream flag.
        RAW = sys::TOKS_DECODE_RAW;
    }
}

flags! {
    /// An id's flags (`toks_id_flags`), and an added token's listing options (`toks_added`).
    IdFlags {
        /// An added token holds the id.
        ADDED = sys::TOKS_ID_ADDED;
        /// The token is special (its own flag, not decode's skip rule).
        SPECIAL = sys::TOKS_ID_SPECIAL;
        /// The id stands for exactly one raw byte.
        BYTE = sys::TOKS_ID_BYTE;
        /// Added tokens only: a match takes the whitespace before it.
        LSTRIP = sys::TOKS_ID_LSTRIP;
        /// Added tokens only: a match takes the whitespace after it.
        RSTRIP = sys::TOKS_ID_RSTRIP;
        /// Added tokens only: never matched inside a word.
        SINGLE_WORD = sys::TOKS_ID_SINGLE_WORD;
        /// Added tokens only: matched on the normalized text.
        NORMALIZED = sys::TOKS_ID_NORMALIZED;
    }
}

flags! {
    /// `toks_scratch_init` flags: memory budgets only, the ids are the same under any of them.
    ScratchFlags {
        /// A 4 MiB segment memo and 2 MiB of piece caches.
        DEFAULT = 0;
        /// No segment memo: for batch jobs over text that never comes back.
        NO_MEMO = sys::TOKS_SCRATCH_MEMO_MIB(0);
    }
}

impl ScratchFlags {
    /// A segment memo of `m` MiB (`m < 4096`).
    pub const fn memo_mib(m: u32) -> Self {
        Self(sys::TOKS_SCRATCH_MEMO_MIB(m))
    }

    /// BPE piece caches of `n` MiB: a power of two 4..=128 (anything else is `TOKS_E_ARG`).
    pub const fn cache_mib(n: u32) -> Self {
        Self(sys::TOKS_SCRATCH_CACHE_MIB(n))
    }
}
