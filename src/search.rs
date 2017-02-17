use std;
use std::io::Error;
use std::fs::File;
use std::process::Command;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use csv;
use pbr::ProgressBar;
use x86::shared::perfcnt::intel::{EventDescription, Tuple, MSRIndex, Counter, PebsType};

use super::profile;
use super::profile::{PerfEvent, MonitoringUnit};

pub fn event_is_documented(events: &Vec<PerfEvent>,
                           unit: MonitoringUnit,
                           code: u8,
                           umask: u8)
                           -> bool {
    for event in events.iter() {
        if event.unit() == unit && event.uses_event_code(code) && event.uses_umask(umask) {
            return true;
        }
    }

    return false;
}

fn execute_perf(perf: &mut Command, cmd: &Vec<String>, counters: &Vec<String>) {
    assert!(cmd.len() >= 1);
    //let mut perf = perf.arg("-o").arg(datafile.as_os_str());
    let events: Vec<String> = counters.iter().map(|c| format!("-e {}", c)).collect();

    let mut perf = perf.args(events.as_slice());
    let mut perf = perf.args(cmd.as_slice());
    let perf_cmd_str: String = format!("{:?}", perf).replace("\"", "");

    let (stdout, stderr) = match perf.output() {
        Ok(out) => {
            let stdout = String::from_utf8(out.stdout)
                .unwrap_or(String::from("Unable to read stdout!"));
            let stderr = String::from_utf8(out.stderr)
                .unwrap_or(String::from("Unable to read stderr!"));

            if out.status.success() {
                // debug!("stdout:\n{:?}", stdout);
                // debug!("stderr:\n{:?}", stderr);
            } else if !out.status.success() {
                error!("perf command: {} got unknown exit status was: {}",
                       perf_cmd_str,
                       out.status);
                debug!("stdout:\n{}", stdout);
                debug!("stderr:\n{}", stderr);
            }

            (stdout, stderr)
        }
        Err(err) => {
            error!("Executing {} failed : {}", perf_cmd_str, err);
            (String::new(), String::new())
        }
    };


    let mut rdr =
        csv::Reader::from_string(stderr).has_headers(false).delimiter(b';').flexible(true);
    for record in rdr.decode() {
        if record.is_ok() {
            type SourceRow = (f64, String, String, String, String, String, f64);
            let (time, cpu, value_string, _, event, _, percent): SourceRow =
                record.expect("Should not happen (in is_ok() branch)!");

            // Perf will just report first CPU on the socket for uncore events,
            // so we temporarily encode the location in the event name and
            // extract it here again:
            let (unit, event_name) = if !event.starts_with("uncore_") {
                // Normal case, we just take the regular event and cpu fields from perf stat
                (String::from("cpu"), String::from(event.trim()))
            } else {
                // Uncore events, use first part of the event name as the location
                let (unit, name) = event.split_at(event.find(".").unwrap());
                (String::from(unit), String::from(name.trim_left_matches(".").trim()))
            };

            let value: u64 = value_string.trim().parse().unwrap();
            if value != 0 {
                println!("{:?} {:?} {:?}", unit, event_name, value);
            }
        }
    }


    //(perf_cmd_str, stdout)
}

pub fn check_events<'a, 'b>(output_path: &Path,
                            cmd_working_dir: &str,
                            cmd: Vec<String>,
                            env: Vec<(String, String)>,
                            breakpoints: Vec<String>,
                            record: bool,
                            events: Vec<&'a EventDescription<'b>>)
    where 'b: 'a
{

    let event_groups = profile::schedule_events(events);
    profile::create_out_directory(output_path);

    profile::check_for_perf();
    let ret = profile::check_for_perf_permissions() || profile::check_for_disabled_nmi_watchdog() ||
              profile::check_for_perf_paranoia();
    if !ret {
        std::process::exit(3);
    }

    assert!(cmd.len() >= 1);
    let mut perf_log = PathBuf::new();
    perf_log.push(output_path);
    perf_log.push("perf.csv");

    let mut wtr = csv::Writer::from_file(perf_log).unwrap();
    let r = wtr.encode(("event_name", "perf_command"));
    assert!(r.is_ok());

    let mut pb = ProgressBar::new(event_groups.len() as u64);
    for group in event_groups {
        let idx = pb.inc();

        let mut event_names: Vec<&str> = group.get_event_names();
        let counters: Vec<String> = group.get_perf_config_strings();

        let mut perf =
            profile::get_perf_command(cmd_working_dir, output_path, &env, &breakpoints, record);
        execute_perf(&mut perf, &cmd, &counters);

        //let r = wtr.encode(vec![event_names.join(","), executed_cmd]);
        //assert!(r.is_ok());

        //let r = wtr.flush();
        //assert!(r.is_ok());
    }
}


pub fn print_unknown_events() {
    let events = profile::get_known_events();
    let pevents: Vec<PerfEvent> = events.into_iter().map(|e| PerfEvent(e)).collect();
    let units = vec![MonitoringUnit::CPU,
                     //MonitoringUnit::UBox,
                     //MonitoringUnit::CBox,
                     //MonitoringUnit::HA,
                     //MonitoringUnit::IMC,
                     //MonitoringUnit::PCU,
                     //MonitoringUnit::R2PCIe,
                     //MonitoringUnit::R3QPI,
                     //MonitoringUnit::QPI
    ];

    println!("Find events...");
    let mut event_names = HashMap::new();
    for code in 0..255 {
        for umask in 0..255 {
            let id = (code as u32) << 8 | umask as u32;
            let value = format!("UNKNOWN_EVENT_{}_{}", code, umask);
            event_names.insert(id, value);
        }
    }

    let mut events = Vec::new();
    for unit in units {
        for code in 0..10 {
            for umask in 0..10 {
                let id = (code as u32) << 8 | umask as u32;

                if event_is_documented(&pevents, unit, code, umask) {
                    println!("Skip documented event {} {:?} {:?}", unit, code, umask);
                    continue;
                }

                let e = EventDescription::new(Tuple::One(code),
                                              Tuple::One(umask),
                                              event_names.get(&id).unwrap().as_str(),
                                              "Unknown Event",
                                              None,
                                              Counter::Programmable(15),
                                              None,
                                              None,
                                              0,
                                              MSRIndex::None,
                                              0,
                                              false,
                                              0x0,
                                              false,
                                              false,
                                              false,
                                              PebsType::Regular,
                                              false,
                                              None,
                                              false,
                                              false,
                                              None,
                                              false,
                                              unit.to_intel_event_description(),
                                              None,
                                              false);
                events.push(e);
            }
        }
    }

    let storage_location = PathBuf::from("unknown_events");
    check_events(&storage_location,
                 ".",
                 vec![String::from("sleep"), String::from("1")],
                 Vec::new(),
                 Vec::new(),
                 false,
                 events.iter().collect())
}
