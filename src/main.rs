extern crate libc;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate nom;
#[macro_use]
extern crate log;
extern crate env_logger;
#[macro_use]
extern crate clap;
extern crate pbr;
extern crate csv;
extern crate x86;
extern crate perfcnt;
extern crate rustc_serialize;
extern crate toml;
extern crate phf;
#[macro_use]
extern crate itertools;

use clap::App;
use std::path::Path;

mod extract;
mod profile;
mod pair;
mod stats;
mod util;

use profile::profile;
use extract::extract;
use pair::pair;
use stats::stats;

fn setup_logging() {
    use log::{LogRecord, LogLevelFilter};
    use env_logger::LogBuilder;

    let format = |record: &LogRecord| {
        format!("[{}] {}:{}: {}",
                record.level(),
                record.location().file(),
                record.location().line(),
                record.args())
    };

    let mut builder = LogBuilder::new();
    builder.format(format).filter(None, LogLevelFilter::Debug);
    builder.init().unwrap();
}

fn main() {
    setup_logging();
    let yaml = load_yaml!("cmd.yml");
    let matches = App::from_yaml(yaml).get_matches();

    if let Some(matches) = matches.subcommand_matches("profile") {
        let output_path = Path::new(matches.value_of("output").unwrap_or("out"));
        let record: bool = matches.is_present("record");
        let cmd: Vec<String> = matches.values_of("COMMAND").unwrap().map(|s| s.to_string()).collect();

        profile(output_path,
                &output_path.to_string_lossy(),
                cmd,
                Default::default(),
                Default::default(),
                record);
    }
    if let Some(matches) = matches.subcommand_matches("extract") {
        let output_path = Path::new(matches.value_of("directory").unwrap_or("out"));
        extract(output_path);
    }
    if let Some(matches) = matches.subcommand_matches("pair") {
        let output_path = Path::new(matches.value_of("directory").unwrap_or("out"));
        let dryrun: bool = matches.is_present("dryrun");
        pair(output_path, dryrun);
    }
    if let Some(matches) = matches.subcommand_matches("stats") {
        let output_path = Path::new(matches.value_of("directory").unwrap_or("out"));
        stats(output_path);
    }

}
