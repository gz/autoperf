#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
extern crate env_logger;
#[macro_use]
extern crate clap;
extern crate pbr;
extern crate csv;
extern crate x86;
extern crate perfcnt;
extern crate yaml_rust;

use clap::{App};
use std::path::Path;

mod extract;
mod profile;
mod pair;

use profile::{profile};
use extract::{extract};
use pair::{pair};

fn setup_logging() {
    use log::{LogRecord, LogLevelFilter};
    use env_logger::LogBuilder;

    let format = |record: &LogRecord| {
        format!("[{}] {}:{}: {}", record.level(), record.location().file(), record.location().line(), record.args())
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
        let cmd: Vec<&str> = matches.values_of("COMMAND").unwrap().collect();

        profile(output_path, cmd, record);
    }
    if let Some(matches) = matches.subcommand_matches("extract") {
        let output_path = Path::new(matches.value_of("input").unwrap_or("out"));
        extract(output_path);
    }
    if let Some(matches) = matches.subcommand_matches("pair") {
        let output_path = Path::new(matches.value_of("output").unwrap_or("out"));
        pair(output_path);
    }
}
