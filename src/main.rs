#[macro_use]
extern crate log;
extern crate env_logger;
#[macro_use]
extern crate clap;
extern crate pbr;

use std::thread;
use std::time::Duration;
use std::process::Command;
use std::error::Error;
use std::io::prelude::*;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;

use clap::{App, AppSettings};
use pbr::ProgressBar;

fn perf_record(idx: u64, cmd: &Vec<&str>, counters: &Vec<&str>, datafile: &Path) {
    assert!(cmd.len() >= 1);
    let counter_args: Vec<String> = counters.iter().map(|c| { format!("-e {}", c) } ).collect();

    let mut perf = Command::new("perf");
    let mut perf = perf.arg("record").arg("-o").arg(datafile.as_os_str());
    let mut perf = perf.args(counter_args.as_slice());
    let mut perf = perf.args(cmd.as_slice());

    //Command::new("perf").arg("record").arg("-o").arg(datafile.as_os_str()).args(counters.as_slice()).args(cmd.as_slice()).output() {
    match perf.output() {
        Ok(out) => {
            debug!("{:?} exit status was: {}", perf, out.status);
            //println!("stdout: {}", String::from_utf8_lossy(&out.stdout));
            println!("stderr: {}", String::from_utf8_lossy(&out.stderr));
        },
        Err(err) => {
            error!("Executing '{}' failed : {}", cmd.join(" "), err)
        }
    }
}

fn run_profile(cmd: Vec<&str>) {
    assert!(cmd.len() >= 1);

    let out_dir = Path::new("out");
    if !out_dir.exists() {
        std::fs::create_dir("out").expect("Can't create `out` directory");
    }
    let path = Path::new("out/log.csv");
    let display = path.display();
    let mut file = match File::create(&path) {
        Err(why) => panic!("Couldn't create {}: {}", display, why.description()),
        Ok(file) => file,
    };

    match file.write_all("cmd, counters, bps, datafile\n".as_bytes()) {
        Err(why) => {
            panic!("Couldn't write to {}: {}", display, why.description())
        },
        Ok(_) => println!("successfully wrote to {}", display),
    }

    let count = 1;
    let mut pb = ProgressBar::new(count);
    let counters = vec!["r1B0"];

    for idx in 0..count {
        let mut perf_data_path = PathBuf::new();
        perf_data_path.push("out");
        perf_data_path.push(format!("{}_perf.data", idx));

        file.write_fmt(format_args!("'{}', '{}', '{}', '{}'\n", cmd.join(" "), counters.join(" "), "", perf_data_path.display())).unwrap();
        perf_record(idx, &cmd, &counters, perf_data_path.as_path());
        pb.inc();
    }
}

fn check_for_perf() {
    match Command::new("perf").output() {
        Ok(out) => {
            if out.status.code() != Some(1) {
                error!("'perf' not installed, you may need to install it (`sudo apt-get install linux-tools-common`).");
                error!("{}", String::from_utf8_lossy(&out.stderr));
                std::process::exit(2);
            }
            else {
                debug!("perf exit status was: {}", out.status);
            }
        },
        Err(err) => {
            error!("'perf' does not seem to be executable? You may need to install it (`sudo apt-get install linux-tools-common`).");
            std::process::exit(2);
        }
    }
}

fn setup_logging() {
    use log::{LogRecord, LogLevelFilter};
    use env_logger::LogBuilder;

    let format = |record: &LogRecord| {
        format!("[{}] {}:{}: {}", record.level(), file!(), line!(), record.args())
    };

    let mut builder = LogBuilder::new();
    builder.format(format).filter(None, LogLevelFilter::Debug);
    builder.init().unwrap();
}

fn main() {
    setup_logging();
    let yaml = load_yaml!("cmd.yml");
    let matches = App::from_yaml(yaml).get_matches();
    check_for_perf();

    if let Some(matches) = matches.subcommand_matches("profile") {
        //let config = matches.value_of("config").unwrap_or("default.conf");
        let cmd: Vec<&str> = matches.values_of("COMMAND").unwrap().collect();
        run_profile(cmd);
    }
}
