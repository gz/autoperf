#[macro_use]
extern crate log;
extern crate env_logger;
#[macro_use]
extern crate clap;
extern crate pbr;
extern crate csv;
extern crate x86;

use std::process::Command;
use std::error::Error;
use std::io::prelude::*;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;

use clap::{App};
use pbr::ProgressBar;
use x86::shared::perfcnt::{core_counters, uncore_counters};
use x86::shared::perfcnt::intel::description::{IntelPerformanceCounterDescription, Tuple};
use x86::shared::{cpuid};

fn perf_record(idx: u64, cmd: &Vec<&str>, counters: &Vec<String>, datafile: &Path) {
    assert!(cmd.len() >= 1);
    let counter_args: Vec<String> = counters.iter().map(|c| { format!("-e {}", c) } ).collect();

    let mut perf = Command::new("perf");
    let mut perf = perf.arg("record").arg("-o").arg(datafile.as_os_str());
    let mut perf = perf.args(counter_args.as_slice());
    let mut perf = perf.args(cmd.as_slice());

    match perf.output() {
        Ok(out) => {
            debug!("{:?} exit status was: {}", perf, out.status);
            println!("stderr: {}", String::from_utf8_lossy(&out.stderr));
        },
        Err(err) => {
            error!("Executing '{}' failed : {}", cmd.join(" "), err)
        }
    }
}

fn create_out_directory(out_dir: &Path) {
    if !out_dir.exists() {
        std::fs::create_dir(out_dir).expect("Can't create `out` directory");
    }
}

fn get_events() -> Vec<&'static IntelPerformanceCounterDescription> {
    let mut events: Vec<&IntelPerformanceCounterDescription> = core_counters().unwrap().values().collect();
    let mut uncore_events: Vec<&IntelPerformanceCounterDescription> = uncore_counters().unwrap().values().collect();
    events.append(&mut uncore_events);

    events
}


fn run_profile(output_path: &Path, cmd: Vec<&str>) {
    assert!(cmd.len() >= 1);
    let mut perf_log = PathBuf::new();
    perf_log.push(output_path);
    perf_log.push("perf.csv");

    let mut wtr = csv::Writer::from_file(perf_log).unwrap();
    let r = wtr.encode(("command", "counters", "breakpoints", "datafile"));
    assert!(r.is_ok());

    let events = get_events();
    let cpuid = cpuid::CpuId::new();
    let available_counters: u8 = cpuid.get_performance_monitoring_info().map_or(0, |info| info.number_of_counters());
    debug!("CPUs have {:?} PMCs", available_counters);
    if available_counters == 0 {
        error!("No PMU counters detected? Can't measure anything.");
        std::process::exit(4);
    }

    let event_groups = events.chunks(available_counters as usize);
    debug!("approx event groups: {:?}", event_groups.len());

    let mut pb = ProgressBar::new(event_groups.len() as u64);
    for group in event_groups {
        let mut record_path = PathBuf::new();
        let idx = pb.inc();
        record_path.push(output_path);
        record_path.push(format!("{}_perf.data", idx));

        //debug!("test group is: {:?}", group);
        let mut counters = Vec::with_capacity(available_counters as usize);
        for counter in group {
            debug!("umask: {:?}", counter.umask);
            debug!("event_code: {:?}", counter.event_code);
            debug!("invert: {:?}", counter.invert);
            debug!("cmask: {:?}", counter.counter_mask);

            //panic!("Can't deal with this...");
            let umask = match counter.umask {
                Tuple::One(mask) => mask,
                Tuple::Two(m1, m2) => { println!("{:?}", counter); panic!("NYI"); m1 }
            };

            let event = match counter.event_code {
                Tuple::One(ev) => ev,
                Tuple::Two(e1, e2) => {println!("{:?}", counter); panic!("NYI"); e1 } //panic!("Can't deal with this...")
            };

            let inv = match counter.invert {
                true => ",inv",
                false => ""
            };

            let cmask = counter.counter_mask;

            counters.push(format!("cpu/event=0x{:x},umask=0x{:x},cmask=0x{:x}{}/", event, umask, cmask, inv));
        }

        //perf stat -e cycles -e cpu/event=0x0e,umask=0x01,inv,cmask=0x01/ -a sleep 5
        //
        // If the Intel docs for a QM720 Core i7 describe an event as:
        //
        // Event  Umask  Event Mask
        // Num.   Value  Mnemonic    Description                        Comment
        //
        // A8H      01H  LSD.UOPS    Counts the number of micro-ops     Use cmask=1 and
        // delivered by loop stream detector  invert to count
        // cycles
        //
        // raw encoding of 0x1A8 can be used:
        //perf stat -e r1a8 -a sleep 1
        //perf record -e r1a8 ...
        //

        perf_record(idx, &cmd, &counters, record_path.as_path());

        let path_string = String::from_str(record_path.to_str().unwrap_or("undecodable")).unwrap();
        let r = wtr.encode(vec![cmd.join(" "), counters.join(" "), String::new(),  path_string]);
        assert!(r.is_ok());
        let r = wtr.flush();
        assert!(r.is_ok());
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
        Err(_) => {
            error!("'perf' does not seem to be executable? You may need to install it (Ubuntu: `sudo apt-get install linux-tools-common`).");
            std::process::exit(2);
        }
    }
}

fn check_for_perf_permissions() {
    let path = Path::new("/proc/sys/kernel/kptr_restrict");
    let mut file = File::open(path).expect("kptr_restrict file does not exist?");
    let mut s = String::new();

    match file.read_to_string(&mut s) {
        Ok(_) =>  {
            match s.trim() {
                "1" => {
                    error!("kptr restriction is enabled. You can either run autoperf as root or do:");
                    error!("\tsudo sh -c \"echo 0 >> {}\"", path.display());
                    error!("to disable.");
                    std::process::exit(3);
                }
                "0" => {
                    debug!("kptr_restrict is already disabled (good).");
                }
                _ => {
                    warn!("Unkown content read from '{}': {}. Proceeding anyways...", path.display(), s.trim());
                }
            }
        }

        Err(why) => {
            error!("Couldn't read {}: {}", path.display(), why.description());
            std::process::exit(3);
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
    check_for_perf_permissions();

    if let Some(matches) = matches.subcommand_matches("profile") {
        let output_path = Path::new(matches.value_of("output").unwrap_or("out"));
        create_out_directory(output_path);

        let cmd: Vec<&str> = matches.values_of("COMMAND").unwrap().collect();
        run_profile(output_path, cmd);
    }
}
