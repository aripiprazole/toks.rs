use std::path::PathBuf;
use std::sync::Arc;

use toks::{Algorithm, DecodeFlags, EncodeFlags, ErrorKind, IdFlags, ScratchFlags, Tokenizer};

// a byte-level BPE (gpt-2 alphabet) with 262 ids: "hello" = 259, "<|endoftext|>" = 261 (special), 'A' = 65
fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/gpt2style.json")
}

fn tok() -> Tokenizer {
    Tokenizer::from_file(data()).expect("load gpt2style.json")
}

const TEXTS: &[&str] = &[
    "",
    "hello",
    "hello world, hello again",
    "A<|endoftext|>hello",
    "  leading and trailing  ",
    "numbers 1234567890 and punctuation!?.",
    "naïve café — ünïcödé ✓ 日本語",
];

#[test]
fn version_and_info() {
    assert_eq!(toks::version(), toks::sys::TOKS_VERSION);
    let tok = tok();
    let info = tok.info();
    assert_eq!(info.abi, toks::ABI);
    assert_eq!(info.algorithm, Algorithm::BpeByteLevel);
    assert_eq!(info.n_ids, 262);
    assert!(info.n_added >= 1);
    assert_eq!(info.max_text, toks::MAX_TEXT as u64);
    assert_eq!(info.truncation, None);
    assert_eq!(info.padding, None);
}

#[test]
fn from_bytes_matches_from_file() {
    let bytes = std::fs::read(data()).unwrap();
    let a = tok();
    let b = Tokenizer::from_bytes(&bytes).unwrap();
    assert_eq!(a.info().image_sha256, b.info().image_sha256);
    for text in TEXTS {
        assert_eq!(a.encode(text, EncodeFlags::DEFAULT).unwrap(), b.encode(text, EncodeFlags::DEFAULT).unwrap());
    }
}

#[test]
fn load_errors() {
    let e = Tokenizer::from_file("/nonexistent/tokenizer.json").unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Open);
    let e = Tokenizer::from_bytes(b"{ not a tokenizer").unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Format);
    assert!(e.detail().is_some(), "a load diagnostic: {e}");
    let e = Tokenizer::from_file("a\0b").unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Arg);
}

#[test]
fn vocabulary() {
    let tok = tok();
    assert_eq!(tok.token_to_id(b"hello"), Some(259));
    assert_eq!(tok.token_to_id("<|endoftext|>"), Some(261));
    assert_eq!(tok.token_to_id(b"no such token"), None);
    assert_eq!(tok.token(259), Some(&b"hello"[..]));
    assert_eq!(tok.token(65), Some(&b"A"[..]));
    assert_eq!(tok.token(1 << 20), None);

    let flags = tok.id_flags(261).unwrap();
    assert!(flags.contains(IdFlags::ADDED | IdFlags::SPECIAL));
    assert!(!tok.id_flags(259).unwrap().contains(IdFlags::SPECIAL));
    assert_eq!(tok.id_flags(1 << 20), None);

    let added: Vec<_> = tok.added_tokens().collect();
    assert_eq!(added.len(), tok.info().n_added as usize);
    let eot = added.iter().find(|t| t.id == 261).expect("<|endoftext|> listed");
    assert_eq!(eot.content, b"<|endoftext|>");
    assert!(eot.flags.contains(IdFlags::SPECIAL));
}

#[test]
fn decode_cases() {
    // upstream's tests/python cases for this file
    let tok = tok();
    assert_eq!(tok.decode(&[261], false).unwrap(), "<|endoftext|>");
    assert_eq!(tok.decode(&[261], true).unwrap(), "");
    assert_eq!(tok.decode(&[259, 261, 65], false).unwrap(), "hello<|endoftext|>A");
    assert_eq!(tok.decode(&[259, 261, 65], true).unwrap(), "helloA");
    assert_eq!(tok.decode(&[], false).unwrap(), "");
    assert_eq!(tok.decode_bytes(&[259, 65], DecodeFlags::RAW).unwrap(), b"helloA");
    assert_eq!(tok.decode(&[1 << 20], false).unwrap_err().kind(), ErrorKind::Id);

    let mut small = [0u8; 4];
    let n = tok.decode_into(&[259, 65], DecodeFlags::DEFAULT, &mut small).unwrap();
    assert_eq!(n, 6);
    assert_eq!(&small, b"hell");
}

#[test]
fn stream_cases() {
    let tok = tok();
    for (ids, skip, want) in [
        (&[261][..], false, "<|endoftext|>"),
        (&[261][..], true, ""),
        (&[259, 261, 65][..], false, "hello<|endoftext|>A"),
        (&[259, 261, 65][..], true, "helloA"),
    ] {
        let mut st = tok.decode_stream(skip);
        let mut got = String::new();
        for &id in ids {
            got.push_str(&st.push(&[id]).unwrap());
        }
        got.push_str(&st.flush().unwrap());
        assert_eq!(got, want);
    }
}

#[test]
fn stream_splits_utf8_like_decode() {
    let tok = tok();
    let text = "naïve café — 日本語 ✓";
    let ids = tok.encode(text, EncodeFlags::DEFAULT).unwrap();
    let mut st = tok.decode_stream(false);
    let mut got = String::new();
    for &id in &ids {
        got.push_str(&st.push(&[id]).unwrap());
    }
    got.push_str(&st.flush().unwrap());
    assert_eq!(got, text);
    // a moved stream keeps working (st holds no pointer to itself)
    let mut moved = Box::new(tok.decode_stream(false));
    assert_eq!(moved.push(&ids).unwrap(), text);
}

#[test]
fn encode_round_trip() {
    let tok = tok();
    assert_eq!(tok.encode("hello", EncodeFlags::DEFAULT).unwrap(), [259]);
    assert_eq!(tok.encode("A<|endoftext|>", EncodeFlags::DEFAULT).unwrap(), [65, 261]);
    assert_ne!(tok.encode("<|endoftext|>", EncodeFlags::ADDED_NONE).unwrap(), [261]);
    for text in TEXTS {
        let ids = tok.encode(text, EncodeFlags::DEFAULT).unwrap();
        assert!(ids.len() <= tok.encode_bound(text.len()));
        assert_eq!(tok.decode(&ids, false).unwrap(), *text);
    }
}

#[test]
fn encode_into_reports_the_total() {
    let tok = tok();
    let text = "hello world, hello again";
    let all = tok.encode(text, EncodeFlags::DEFAULT).unwrap();
    let mut out = vec![0u32; 3];
    let n = tok.encode_into(text, EncodeFlags::DEFAULT, &mut out).unwrap();
    assert_eq!(n, all.len());
    assert_eq!(out, all[..3]);
}

#[test]
fn long_text_grows_the_scratch() {
    let tok = tok();
    let text = "hello world ".repeat(20_000); // 240 KB: past a pool slot's first 64 KiB
    let ids = tok.encode(&text, EncodeFlags::DEFAULT).unwrap();
    assert_eq!(tok.decode(&ids, false).unwrap(), text);

    let mut scr = tok.scratch(16).unwrap();
    assert_eq!(scr.encode(&text, EncodeFlags::DEFAULT).unwrap(), ids);
    assert!(scr.max_len() >= text.len());

    let e = toks::Scratch::new(&tok, toks::MAX_TEXT + 1, ScratchFlags::DEFAULT).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Limit);
}

#[test]
fn pieces_end_at_the_text() {
    let tok = tok();
    let text = "hello world, hello";
    let ends = tok.pieces(text, EncodeFlags::DEFAULT).unwrap();
    assert!(!ends.is_empty());
    assert!(ends.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(*ends.last().unwrap() as usize, text.len());
}

#[test]
fn split_points_concatenate() {
    let tok = tok();
    let text = "hello world, hello again. ".repeat(500);
    let flags = EncodeFlags::NO_POSTPROCESS;
    let cuts = tok.split_points(&text, flags, 8).unwrap();
    assert!(cuts.len() <= 7);
    assert!(cuts.windows(2).all(|w| w[0] < w[1]));
    let mut ids = Vec::new();
    let mut at = 0;
    for (k, &cut) in cuts.iter().chain(std::iter::once(&text.len())).enumerate() {
        let part_flags = if k == 0 { flags } else { flags | EncodeFlags::CONTINUATION };
        ids.extend(tok.encode(&text[at..cut], part_flags).unwrap());
        at = cut;
    }
    assert_eq!(ids, tok.encode(&text, flags).unwrap());
}

#[test]
fn template() {
    let tok = tok();
    let t = tok.template().unwrap();
    assert_eq!(t.ids.len(), t.type_ids.len());
    assert_eq!(t.prefix().len(), tok.info().template_prefix as usize);
    assert_eq!(t.suffix().len(), tok.info().template_suffix as usize);
}

#[test]
fn scratches_on_threads_share_one_tokenizer() {
    let tok = Arc::new(tok());
    let want: Vec<Vec<u32>> = TEXTS.iter().map(|t| tok.encode(t, EncodeFlags::DEFAULT).unwrap()).collect();
    std::thread::scope(|s| {
        for _ in 0..4 {
            let (tok, want) = (&tok, &want);
            s.spawn(move || {
                let mut scr = tok.scratch(1024).unwrap();
                for _ in 0..50 {
                    for (text, want) in TEXTS.iter().zip(want) {
                        assert_eq!(&scr.encode(text, EncodeFlags::DEFAULT).unwrap(), want);
                        assert_eq!(&tok.encode(text, EncodeFlags::DEFAULT).unwrap(), want);
                    }
                }
            });
        }
    });
    let tok = Arc::into_inner(tok).unwrap();
    let mut scr = toks::Scratch::new(&tok, 0, ScratchFlags::NO_MEMO | ScratchFlags::cache_mib(4)).unwrap();
    assert_eq!(scr.encode("hello", EncodeFlags::DEFAULT).unwrap(), [259]);
    assert_eq!(toks::Scratch::new(&tok, 16, ScratchFlags::cache_mib(3)).unwrap_err().kind(), ErrorKind::Arg);
}

#[test]
fn pool_equals_serial() {
    let tok = tok();
    let pool = tok.pool(4).unwrap();
    let info = pool.info().unwrap();
    assert!(info.threads >= 1 && info.threads <= 4);

    let big = "hello world, hello again. naïve café ✓ ".repeat(50_000); // ~2 MB: worth splitting
    assert_eq!(pool.encode(&big, EncodeFlags::DEFAULT).unwrap(), tok.encode(&big, EncodeFlags::DEFAULT).unwrap());

    let mut texts: Vec<String> = TEXTS.iter().map(|t| t.to_string()).collect();
    texts.push(big);
    let batch = pool.encode_batch(&texts, EncodeFlags::DEFAULT).unwrap();
    for (text, ids) in texts.iter().zip(&batch) {
        assert_eq!(ids, &tok.encode(text, EncodeFlags::DEFAULT).unwrap());
    }

    // short outputs report their totals
    let mut a = [0u32; 1];
    let mut b = [0u32; 64];
    let ns = pool.encode_batch_into(&["hello world", "hello"], &mut [&mut a[..], &mut b[..]], EncodeFlags::DEFAULT).unwrap();
    assert!(*ns[0].as_ref().unwrap() > 1);
    assert_eq!(ns[1], Ok(1));
    assert_eq!(b[0], 259);

    // shared by threads: calls wait for each other
    std::thread::scope(|s| {
        for _ in 0..3 {
            s.spawn(|| assert_eq!(pool.encode("hello", EncodeFlags::DEFAULT).unwrap(), [259]));
        }
    });
}

#[test]
fn stream_grows_its_hold_for_long_byte_fallback_runs() {
    // a sentencepiece-style BPE with byte fallback: U+13000 is no piece, so each one is 4 <0xHH> ids, and a
    // stream holds the whole run (valid UTF-8 so far) until a string or the flush ends it: 1200 bytes, far
    // past st's own 44
    let tok = Tokenizer::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/llamalike.json")).unwrap();
    assert_eq!(tok.info().algorithm, Algorithm::BpeSpm);
    let text = format!("hi {} end", "\u{13000}".repeat(300));
    let ids = tok.encode(&text, EncodeFlags::DEFAULT).unwrap();
    let whole = tok.decode(&ids, false).unwrap();
    assert!(whole.contains(&"\u{13000}".repeat(300)));

    let mut st = tok.decode_stream(false);
    let mut got = String::new();
    for &id in &ids {
        got.push_str(&st.push(&[id]).unwrap());
    }
    got.push_str(&st.flush().unwrap());
    assert_eq!(got, whole);
    assert!(!format!("{st:?}").contains("hold: 0"), "the hold grew: {st:?}");

    let mut st = tok.decode_stream(false);
    let mut got = st.push(&ids).unwrap().into_owned();
    got.push_str(&st.flush().unwrap());
    assert_eq!(got, whole);
}
