use clap::{load_yaml, App};
use std::path::{Path, PathBuf};
use std::str::FromStr;

mod aggregate;
mod mkgroup;
mod pair;
mod profile;
mod scale;
mod search;
mod stats;
mod util;
use log::*;

use aggregate::aggregate;
use pair::pair;
use profile::profile;
use stats::stats;

use mkgroup::mkgroup;
use search::print_unknown_events;

fn setup_logging(lvl: &str) {
    use env_logger::Env;
    env_logger::from_env(Env::default().default_filter_or(lvl)).init();
}

fn main() {
    let yaml = load_yaml!("cmd.yml");
    let matches = App::from_yaml(yaml).get_matches();

    let level = match matches.occurrences_of("verbose") {
        0 => "warn",
        1 => "info",
        2 => "debug",
        3 => "trace",
        _ => "trace",
    };
    setup_logging(level);

    if let Some(matches) = matches.subcommand_matches("profile") {
        let output_path = Path::new(matches.value_of("output").unwrap_or("out"));
        let cmd: Vec<String> = matches
            .values_of("COMMAND")
            .unwrap()
            .map(|s| s.to_string())
            .collect();

        let dryrun: bool = matches.is_present("dryrun");
        profile(
            output_path,
            ".",
            cmd,
            Default::default(),
            Default::default(),
            false,
            None,
            dryrun,
        );
    }
    if let Some(matches) = matches.subcommand_matches("aggregate") {
        let input_directory = Path::new(matches.value_of("directory").unwrap_or("out"));
        let output_path: PathBuf = match matches.value_of("output") {
            Some(v) => PathBuf::from(v),
            None => {
                let mut pb = input_directory.to_path_buf();
                pb.push("results.csv");
                pb
            }
        };
        let uncore_filter: &str = matches.value_of("uncore").unwrap_or("all");
        let core_filter: &str = matches.value_of("core").unwrap_or("all");

        aggregate(
            input_directory,
            core_filter,
            uncore_filter,
            &output_path.as_path(),
        );
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
        let _output_path = Path::new(matches.value_of("directory").unwrap_or("out"));
        let _dryrun: bool = matches.is_present("dryrun");
        // scale(output_path, dryrun);
    }
    if let Some(matches) = matches.subcommand_matches("stats") {
        let output_path = Path::new(matches.value_of("directory").unwrap_or("out"));
        stats(output_path);
    }
    if let Some(_matches) = matches.subcommand_matches("search") {
        print_unknown_events();
    }
    if let Some(matches) = matches.subcommand_matches("mkgroup") {
        let ranking_file = Path::new(matches.value_of("file").unwrap_or("notfound"));
        mkgroup(ranking_file);
    }
}
