use std;


use std::process::Command;
use std::collections::HashMap;
use std::collections::BTreeSet;
use std::path::Path;
use std::path::PathBuf;

use csv;

use x86::perfcnt::intel::{EventDescription, Tuple, MSRIndex, Counter, PebsType};

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

fn execute_perf(perf: &mut Command,
                cmd: &Vec<String>,
                counters: &Vec<String>)
                -> BTreeSet<(String, String)> {
    assert!(cmd.len() >= 1);
    let events: Vec<String> = counters.iter().map(|c| format!("-e {}", c)).collect();

    let perf = perf.args(events.as_slice());
    let perf = perf.args(cmd.as_slice());
    let perf_cmd_str: String = format!("{:?}", perf).replace("\"", "");

    let (_stdout, stderr) = match perf.output() {
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

    let mut found_events = BTreeSet::new();
    let mut rdr =
        csv::Reader::from_string(stderr).has_headers(false).delimiter(b';').flexible(true);
    for record in rdr.decode() {
        if record.is_ok() {
            type SourceRow = (f64, String, String, String, String, String, f64);
            let (_time, _cpu, value_string, _, event, _, _percent): SourceRow =
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
                // remove the _1 in uncore_cbox_1:
                let mut unit_parts: Vec<&str> = unit.split('_').collect();
                unit_parts.pop();
                (String::from(unit_parts.join("_")),
                 String::from(name.trim_start_matches(".").trim()))
            };

            let value: u64 = value_string.trim().parse().unwrap_or(0);
            if value != 0 {
                debug!("{:?} {:?} {:?}", unit, event_name, value);
                found_events.insert((event_name, unit));
            }
        }
    }

    found_events
}

pub fn check_events<'a, 'b>(output_path: &Path,
                            cmd_working_dir: &str,
                            cmd: Vec<String>,
                            env: Vec<(String, String)>,
                            breakpoints: Vec<String>,
                            record: bool,
                            events: Vec<&'a EventDescription<'b>>)
                            -> BTreeSet<(String, String)>
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
    perf_log.push("unknown_events.csv");

    let mut all_events = BTreeSet::new();
    for group in event_groups {
        let mut _event_names: Vec<&str> = group.get_event_names();
        let counters: Vec<String> = group.get_perf_config_strings();
        let mut perf =
            profile::get_perf_command(cmd_working_dir, output_path, &env, &breakpoints, record);
        let mut found_events = execute_perf(&mut perf, &cmd, &counters);
        all_events.append(&mut found_events);
    }

    all_events
}


pub fn print_unknown_events() {
    let events = profile::get_known_events();
    let pevents: Vec<PerfEvent> = events.into_iter().map(|e| PerfEvent(e)).collect();
    let units = vec![MonitoringUnit::CPU,
                     //MonitoringUnit::UBox,
                     MonitoringUnit::CBox,
                     MonitoringUnit::HA,
                     MonitoringUnit::IMC,
                     //MonitoringUnit::PCU,
                     //MonitoringUnit::R2PCIe,
                     MonitoringUnit::R3QPI,
                     //MonitoringUnit::QPI
    ];

    let mut event_names = HashMap::new();
    for unit in units.iter() {
        for code in 1..255 {
            for umask in 1..255 {
                let id: isize = (*unit as isize) << 32 | (code as isize) << 8 | umask as isize;
                let value = format!("{}_EVENT_{}_{}",
                                    unit.to_intel_event_description().unwrap_or("CPU"),
                                    code,
                                    umask);
                event_names.insert(id, value);
            }
        }
    }

    println!("Find events...");
    let mut storage_location = PathBuf::from("unknown_events");
    profile::create_out_directory(&storage_location);
    storage_location.push("found_events.dat");
    let mut wtr = csv::Writer::from_file(storage_location).unwrap();
    let r = wtr.encode(("unit", "code", "mask", "event_name"));
    assert!(r.is_ok());

    let mut events = Vec::new();
    for code in 1..255 {
        for umask in 1..255 {
            for unit in units.iter() {
                let id: isize = (*unit as isize) << 32 | (code as isize) << 8 | umask as isize;

                if event_is_documented(&pevents, *unit, code, umask) {
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

        let mut storage_location = PathBuf::from("unknown_events");
        let all_found_events = check_events(&storage_location,
                                            ".",
                                            vec![String::from("sleep"), String::from("1")],
                                            Vec::new(),
                                            Vec::new(),
                                            false,
                                            events.iter().collect());
        for &(ref name, ref unit) in all_found_events.iter() {
            let splitted: Vec<&str> = name.split("_").collect();
            let r =
                wtr.encode(vec![unit,
                                &String::from(splitted[2]),
                                &String::from(splitted[3]),
                                name]);
            assert!(r.is_ok());
        }
        let r = wtr.flush();
        assert!(r.is_ok());

        events.clear();
    }
}
