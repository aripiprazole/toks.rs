//! cargo run --example encode -- <tokenizer.json> [text]   (text from stdin when not given)

use std::io::Read;

use toks::{EncodeFlags, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: encode <tokenizer.json> [text]")?;
    let text = match args.next() {
        Some(text) => text,
        None => {
            let mut text = String::new();
            std::io::stdin().read_to_string(&mut text)?;
            text
        }
    };

    let tok = Tokenizer::from_file(&path)?;
    let info = tok.info();
    eprintln!("{} ({:?}, {} ids, tier {:?}, toks {})", info.name, info.algorithm, info.n_ids, info.tier, toks::version());

    let ids = tok.encode(&text, EncodeFlags::DEFAULT)?;
    println!("{ids:?}");

    let mut stream = tok.decode_stream(false);
    for &id in &ids {
        print!("{}", stream.push(&[id])?);
    }
    println!("{}", stream.flush()?);
    Ok(())
}
