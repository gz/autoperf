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
use std::path::{Path, PathBuf};
use std::str::FromStr;

mod extract;
mod profile;
mod pair;
mod scale;
mod stats;
mod util;

use profile::profile;
use extract::extract;
use pair::pair;
use stats::stats;
use scale::scale;

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
        let cmd: Vec<String> =
            matches.values_of("COMMAND").unwrap().map(|s| s.to_string()).collect();

        profile(output_path,
                &output_path.to_string_lossy(),
                cmd,
                Default::default(),
                Default::default(),
                record);
    }
    if let Some(matches) = matches.subcommand_matches("extract") {
        let input_directory = Path::new(matches.value_of("directory").unwrap_or("out"));
        let output_path: PathBuf = match matches.value_of("output") {
            Some(v) => PathBuf::from(v),
            None => {
                let mut pb = input_directory.to_path_buf();
                pb.push("results.csv");
                pb
            }
        };
        let uncore_filter: &str = matches.value_of("uncore").unwrap_or("exclusive");
        let core_filter: &str = matches.value_of("core").unwrap_or("exclusive");
        
        extract(input_directory, core_filter, uncore_filter, &output_path.as_path());
    }
    if let Some(matches) = matches.subcommand_matches("pair") {
        let output_path = Path::new(matches.value_of("directory").unwrap_or("out"));
        let start: usize = usize::from_str(matches.value_of("start").unwrap_or("0")).unwrap_or(0);
        let stepping: usize = usize::from_str(matches.value_of("step").unwrap_or("1")).unwrap_or(1);
        if stepping == 0 {
            error!("skip amount must be > 0");
            std::process::exit(1);
        }

        let dryrun: bool = matches.is_present("dryrun");
        pair(output_path, dryrun, start, stepping);
    }
    if let Some(matches) = matches.subcommand_matches("scale") {
        let output_path = Path::new(matches.value_of("directory").unwrap_or("out"));
        let dryrun: bool = matches.is_present("dryrun");
        scale(output_path, dryrun);
    }
    if let Some(matches) = matches.subcommand_matches("stats") {
        let output_path = Path::new(matches.value_of("directory").unwrap_or("out"));
        stats(output_path);
    }

}
