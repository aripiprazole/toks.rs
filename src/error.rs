use std::fmt;

use crate::sys;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The `TOKS_E_*` code an error carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// The file could not be opened or read.
    Open,
    /// Not a valid tokenizer file (syntax, structure, checksum).
    Format,
    /// A valid tokenizer using a feature toks does not support (named in the detail).
    Unsupported,
    /// A forced tier this machine cannot run.
    Tier,
    /// Scratch missing, too small, not initialized, or bound to another context.
    Scratch,
    /// An id beyond the table.
    Id,
    /// Output capacity below what an atomic call needs.
    Cap,
    /// An input beyond the library's limits (text over [`MAX_TEXT`](crate::MAX_TEXT) bytes, ...).
    Limit,
    /// A precondition the call can check (unknown flag bits, ...).
    Arg,
    /// Load-time allocation failed.
    NoMem,
    /// A code this binding does not know.
    Other(i64),
}

impl ErrorKind {
    pub fn from_code(code: i64) -> Self {
        match code {
            sys::TOKS_E_OPEN => Self::Open,
            sys::TOKS_E_FORMAT => Self::Format,
            sys::TOKS_E_UNSUPPORTED => Self::Unsupported,
            sys::TOKS_E_TIER => Self::Tier,
            sys::TOKS_E_SCRATCH => Self::Scratch,
            sys::TOKS_E_ID => Self::Id,
            sys::TOKS_E_CAP => Self::Cap,
            sys::TOKS_E_LIMIT => Self::Limit,
            sys::TOKS_E_ARG => Self::Arg,
            sys::TOKS_E_NOMEM => Self::NoMem,
            other => Self::Other(other),
        }
    }

    pub fn code(self) -> i64 {
        match self {
            Self::Open => sys::TOKS_E_OPEN,
            Self::Format => sys::TOKS_E_FORMAT,
            Self::Unsupported => sys::TOKS_E_UNSUPPORTED,
            Self::Tier => sys::TOKS_E_TIER,
            Self::Scratch => sys::TOKS_E_SCRATCH,
            Self::Id => sys::TOKS_E_ID,
            Self::Cap => sys::TOKS_E_CAP,
            Self::Limit => sys::TOKS_E_LIMIT,
            Self::Arg => sys::TOKS_E_ARG,
            Self::NoMem => sys::TOKS_E_NOMEM,
            Self::Other(code) => code,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Open => "TOKS_E_OPEN",
            Self::Format => "TOKS_E_FORMAT",
            Self::Unsupported => "TOKS_E_UNSUPPORTED",
            Self::Tier => "TOKS_E_TIER",
            Self::Scratch => "TOKS_E_SCRATCH",
            Self::Id => "TOKS_E_ID",
            Self::Cap => "TOKS_E_CAP",
            Self::Limit => "TOKS_E_LIMIT",
            Self::Arg => "TOKS_E_ARG",
            Self::NoMem => "TOKS_E_NOMEM",
            Self::Other(_) => "TOKS_E_?",
        }
    }
}

/// A failed toks call: its code, and for loads (or checks made on this side) what went wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    detail: Option<Box<str>>,
}

impl Error {
    pub(crate) fn new(code: i64, detail: impl Into<Option<String>>) -> Self {
        Self { kind: ErrorKind::from_code(code), detail: detail.into().map(String::into_boxed_str) }
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The raw `TOKS_E_*` value (negative).
    pub fn code(&self) -> i64 {
        self.kind.code()
    }

    /// The load diagnostic (`toks_diag.what`) or the binding's own reason, when there is one.
    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "toks: {} ({})", self.kind.name(), self.code())?;
        if let Some(detail) = &self.detail {
            write!(f, ": {detail}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Self {
        Self { kind, detail: None }
    }
}

/// A return value: n >= 0 as a count, or the error it codes.
pub(crate) fn check(r: i64) -> Result<usize> {
    if r < 0 { Err(Error::new(r, None)) } else { Ok(r as usize) }
}
