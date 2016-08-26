#[macro_use]

extern crate clap;
use clap::{App, AppSettings};

extern crate pbr;
use pbr::ProgressBar;
use std::thread;
use std::time::Duration;

fn main() {

    let yaml = load_yaml!("cmd.yml");
    let matches = App::from_yaml(yaml).setting(AppSettings::TrailingVarArg).get_matches();

    if let Some(matches) = matches.subcommand_matches("profile") {
        if matches.is_present("debug") {
            println!("Printing debug info...");
        } else {
            println!("Printing normally...");
        }

        let config = matches.value_of("config").unwrap_or("default.conf");
        println!("Value for config: {}", config);
        let cmd: Vec<&str> = matches.values_of("COMMAND").unwrap().collect();
        println!("Value for cmd: {:?}", cmd);
    }


    let count = 10;
    let mut pb = ProgressBar::new(count);
    for _ in 0..count {
        pb.inc();
        thread::sleep(Duration::from_millis(25));
    }
}
