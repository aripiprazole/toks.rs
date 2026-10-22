//! Builds the C library with its own Makefile (as upstream's python/setup.py does) and links it statically.
//!
//! vendor/toks is a git submodule (`git submodule update --init` after cloning).
//!
//!   TOKS_SRC_DIR=<toks checkout>   build that tree instead of vendor/toks
//!   TOKS_LIB_DIR=<dir>             link a prebuilt libtoks.a from <dir>; nothing is built
//!   CC / AR                        passed through to make (the Makefile defaults to clang)

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    for var in ["TOKS_SRC_DIR", "TOKS_LIB_DIR", "CC", "AR", "MACOSX_DEPLOYMENT_TARGET"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let target = env::var("TARGET").unwrap();
    let host = env::var("HOST").unwrap();

    if let Some(dir) = env::var_os("TOKS_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", PathBuf::from(dir).display());
    } else {
        let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
        let src = env::var_os("TOKS_SRC_DIR").map_or_else(|| manifest.join("vendor/toks"), PathBuf::from);
        if !src.join("include/toks.h").is_file() {
            panic!(
                "toks: no include/toks.h under {} (run `git submodule update --init`, or set TOKS_SRC_DIR or TOKS_LIB_DIR)",
                src.display()
            );
        }
        for path in ["include", "src", "Makefile"] {
            println!("cargo:rerun-if-changed={}", src.join(path).display());
        }

        // BUILD_DIR must be absolute: the Makefile writes have.txt there while it parses
        let build = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("toks");
        let lib = build.join("libtoks.a");
        let mut make = Command::new(env::var_os("MAKE").unwrap_or_else(|| "make".into()));
        make.arg("-C").arg(&src).arg(format!("BUILD_DIR={}", build.display())).arg(&lib);
        // cargo's TARGET would reach the Makefile's `TARGET ?= $(HOST)`: only a cross build names one
        make.env_remove("TARGET");
        make.env_remove("GUARD"); // the Makefile's test-only guard build
        if target != host {
            make.arg(format!("TARGET={target}"));
        }
        match env::var_os("CARGO_MAKEFLAGS") {
            Some(flags) => {
                make.env("MAKEFLAGS", flags);
            }
            None => {
                make.arg(format!("-j{}", env::var("NUM_JOBS").unwrap_or_else(|_| "4".into())));
            }
        }
        // rustc's own minimums, so the linker does not warn about objects built for the host's macOS
        if target.contains("apple-darwin") && env::var_os("MACOSX_DEPLOYMENT_TARGET").is_none() {
            make.env("MACOSX_DEPLOYMENT_TARGET", if target.starts_with("aarch64") { "11.0" } else { "10.12" });
        }
        let status = make.status().unwrap_or_else(|e| panic!("toks: could not run make: {e}"));
        if !status.success() {
            panic!("toks: make failed ({status})");
        }
        if target.contains("msvc") {
            // rustc looks for toks.lib on msvc targets
            std::fs::copy(&lib, build.join("toks.lib")).unwrap();
        }
        println!("cargo:rustc-link-search=native={}", build.display());
        println!("cargo:include={}", src.join("include").display());
    }

    println!("cargo:rustc-link-lib=static=toks");
    if !target.contains("windows") {
        println!("cargo:rustc-link-lib=pthread"); // src/par's worker threads
    }
}
