use std::process;
use std::io;
use std::io::prelude::*;
use std::fs;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::process::{Command, Child, Stdio};
use std::str::{FromStr, from_utf8_unchecked};
use std::fmt;
use rustc_serialize::Encodable;
use itertools::Itertools;

use csv;

use toml;

use super::util::*;

pub fn scale(manifest_folder: &Path, dryrun: bool) {
    panic!("NYI");
}
