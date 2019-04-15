use std::io;
use std::fs;
use std::fs::{File, Metadata};
use std::path::Path;
use std::path::PathBuf;
use std::io::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::process;
use std::str::FromStr;
use toml;
use csv;
use util::*;

use perfcnt::linux::perf_file::PerfFile;
use perfcnt::linux::perf_format::{EventDesc, EventType, EventData};

// I have no idea if the perf format guarantees that events appear always in the same order :S
fn verify_events_in_order(events: &Vec<EventDesc>, values: &Vec<(u64, Option<u64>)>) -> bool {
    for (idx, v) in values.iter().enumerate() {
        // Don't have id's we can't veryify anything
        if v.1.is_none() {
            warn!("Don't have IDs with the sample values, so we can't tell which event a sample \
                   belongs to.");
            return true;
        }

        let id: u64 = v.1.unwrap_or(0);
        if !events.get(idx).map_or(false, |ev| ev.ids.contains(&id)) {
            return false;
        }
    }

    return true;
}

/// Extracts the perf stat file and writes it to a CSV file that looks like this:
/// "EVENT_NAME", "TIME", "SOCKET", "CORE", "CPU", "NODE", "UNIT", "SAMPLE_VALUE"
fn parse_perf_csv_file(mt: &MachineTopology,
                       cpus: &Vec<&CpuInfo>,
                       cpu_filter: Filter,
                       sockets: &Vec<Socket>,
                       breakpoints: &Vec<String>,
                       path: &Path,
                       writer: &mut csv::Writer<File>)
                       -> io::Result<()> {
    // Check if it's a file:
    let meta: Metadata = try!(fs::metadata(path));
    if !meta.file_type().is_file() {
        error!("Not a file {:?}", path);
    }

    let mut erronous_events: HashMap<String, bool> = HashMap::new();
    type OutputRow = (String, String, Socket, Core, Cpu, Node, String, u64);
    let mut parsed_rows: Vec<OutputRow> = Vec::with_capacity(5000);

    // All the sockets this program is running on:
    let mut all_sockets: Vec<Socket> = cpus.iter().map(|c| c.socket).collect();
    all_sockets.sort();
    all_sockets.dedup();

    // Timestamps for filtering start and end:
    let mut start: Option<f64> = None;
    let mut end: Option<f64> = None;

    let mut rdr =
        csv::Reader::from_file(path).unwrap().has_headers(false).delimiter(b';').flexible(true);
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
                (String::from(unit), String::from(name.trim_start_matches(".").trim()))
            };

            if erronous_events.contains_key(&event_name) {
                // Skip already reported, bad events
                continue;
            }


            if !cpu.starts_with("CPU") {
                error!("{:?}: Unkown CPU value {}, skipping this row.",
                       path.as_os_str(),
                       cpu);
                continue;
            }

            let cpu_nr = match u64::from_str(&cpu[3..].trim()) {
                Ok(v) => v,
                Err(_e) => {
                    error!("{:?}: CPU value is not a number '{}', skipping this row.",
                           path.as_os_str(),
                           cpu);
                    continue;
                }
            };
            let cpuinfo: &CpuInfo = mt.cpu(cpu_nr)
                .expect("Invalid CPU number (check run.toml or lspcu.csv)");

            if value_string.trim() == "<not counted>" {
                error!("{:?}: Event '{}' was not counted. This is a bug, please report it!",
                       path.as_os_str(),
                       event_name);
                erronous_events.insert(event_name.clone(), true);
            }
            if percent < 91.0 {
                error!("{:?}: has multiplexed event '{}'. This is a bug, please report it!",
                       path.as_os_str(),
                       event_name);
                erronous_events.insert(event_name.clone(), true);
            }

            let value = u64::from_str(value_string.trim()).expect("Should be a value by now!");

            if breakpoints.len() >= 1 && value == 1 &&
               event_name.ends_with(breakpoints[0].as_str()) &&
               cpus.iter().any(|c| c.cpu == cpu_nr) {
                if start.is_some() {
                    error!("{:?}: Start breakpoint ({:?}) triggered multiple times.",
                           path.as_os_str(),
                           breakpoints[0]);
                }
                start = Some(time)
            }
            if breakpoints.len() >= 2 && value == 1 &&
               event_name.ends_with(breakpoints[1].as_str()) &&
               cpus.iter().any(|c| c.cpu == cpu_nr) {
                if end.is_some() {
                    warn!("{:?}: End breakpoint ({:?}) triggered multiple times. Update end \
                           breakpoint.",
                          path.as_os_str(),
                          breakpoints[1]);
                }
                end = Some(time)
            }

            parsed_rows.push((event_name,
                              time.to_string(),
                              cpuinfo.socket,
                              cpuinfo.core,
                              cpu_nr,
                              cpuinfo.node.node,
                              unit,
                              value));
        } else {
            // Ignore lines that start with # (comments) but fail in case another
            // line can not be parsed:
            match record.unwrap_err() {
                csv::Error::Decode(s) => {
                    if !s.starts_with("Failed converting '#") {
                        panic!("Can't decode line {}.", s)
                    }
                }
                e => panic!("Unrecoverable error {} while decoding.", e),
            };
        }
    }
    if breakpoints.len() >= 1 && start.is_none() {
        error!("{:?}: We did not find a trigger for start breakpoint ({:?})",
               path.as_os_str(),
               breakpoints[0]);
    }
    if breakpoints.len() == 2 && end.is_none() {
        warn!("{:?}: We did not find a trigger for end breakpoint ({:?})",
              path.as_os_str(),
              breakpoints[1]);
    }
    if breakpoints.len() == 2 && end.is_some() && start.is_some() {
        let start_s = start.unwrap_or(0.0);
        let end_s = end.unwrap_or(0.0);
        if end_s <= start_s {
            error!("{:?}: End breakpoint is before start breakpoint ({:?} -- {:?})",
                   path.as_os_str(),
                   start,
                   end);
        } else if (end_s - start_s) < 1.0 {
            warn!("Region of interest very short ({} s)", end_s - start_s);
        }
    }

    let mut current_index = 0;
    let mut time_to_index: HashMap<String, usize> = HashMap::new();
    let mut is_recording: bool = start.is_none();
    let start = start.map(|s| s.to_string());
    let end = end.map(|s| s.to_string());
    for r in parsed_rows {
        let (event_name, time, socket, core, cpu, node, unit, value): OutputRow = r;

        // Skip all events before we have the breakpoint
        is_recording = match start {
            Some(ref start_time) => is_recording || time == start_time.as_str(),
            None => true,
        };
        is_recording = match end {
            Some(ref end_time) => is_recording && time != end_time.as_str(),
            None => true,
        };
        if !is_recording {
            continue;
        }

        if erronous_events.contains_key(&event_name) {
            // We do two passes here because we may get an erronous event only
            // at a later point in time in the CSV file
            // (when we already parsed this event a few times)
            continue;
        }

        if event_name.contains(breakpoints[0].as_str()) ||
           event_name.contains(breakpoints[1].as_str()) {
            // We don't need to breakpoints in the resulting CSV file
            continue;
        }

        // Skip all events that we don't want to attribute fully to our program:
        let include = if unit.trim() == "cpu" {
            match cpu_filter {
                Filter::All => true,
                Filter::Exclusive => cpus.iter().any(|c| c.cpu == cpu),
                Filter::Shared => all_sockets.contains(&socket),
                Filter::None => false,
            }
        } else if unit.starts_with("uncore") {
            sockets.contains(&socket)
        } else {
            error!("Unkown unit '{}', not included!", unit);
            false
        };

        if !include {
            // Skip this event
            continue;
        }

        if !time_to_index.contains_key(&time) {
            time_to_index.insert(time.clone(), current_index);
            current_index += 1;
        }

        writer.encode(&[event_name.as_str(),
                      format!("{}", *time_to_index.get(&time).unwrap()).as_str(),
                      time.as_str(),
                      socket.to_string().as_str(),
                      core.to_string().as_str(),
                      cpu.to_string().as_str(),
                      node.to_string().as_str(),
                      unit.as_str(),
                      value.to_string().as_str()])
            .unwrap();
    }

    Ok(())
}

/// Extracts the data and writes it to a CSV file that looks like this:
/// "EVENT_NAME", "INDEX", "TIME", "SOCKET", "CORE", "CPU", "NODE", "UNIT", "SAMPLE_VALUE"
fn parse_perf_file(path: &Path,
                   event_names: Vec<&str>,
                   writer: &mut csv::Writer<File>)
                   -> io::Result<()> {

    // Check if it's a file:
    let meta: Metadata = try!(fs::metadata(path));
    if !meta.file_type().is_file() {
        error!("Not a file {:?}", path);
    }
    // TODO: Should just pass Path to PerfFile
    let mut file = try!(File::open(path));
    let mut buf: Vec<u8> = Vec::with_capacity(meta.len() as usize);
    try!(file.read_to_end(&mut buf));
    let pf = PerfFile::new(buf);

    // debug!("GroupDescriptions: {:?}", pf.get_group_descriptions());
    // debug!("EventDescription: {:?}", pf.get_event_description());

    let event_desc = pf.get_event_description().unwrap();
    let event_info: Vec<(&EventDesc, &&str)> = event_desc.iter().zip(event_names.iter()).collect();
    // debug!("Event Infos: {:?}", event_info);

    for e in pf.data() {
        if e.header.event_type != EventType::Sample {
            continue;
        }

        match e.data {
            EventData::Sample(rec) => {
                // println!("{:?}", rec);
                let time = format!("{}", rec.time.unwrap());
                let ptid = rec.ptid.unwrap();
                let _pid = format!("{}", ptid.pid);
                let _tid = format!("{}", ptid.tid);
                let cpu = format!("{}", rec.cpu.unwrap().cpu);
                // let ip = format!("0x{:x}", rec.ip.unwrap());

                let v = rec.v.unwrap();
                assert!(verify_events_in_order(&event_desc, &v.values));
                // TODO: verify event names match EventDesc in `event_info`!

                for reading in v.values.iter() {
                    let (event_count, maybe_id) = *reading;
                    let id = maybe_id.unwrap();
                    let &(_, name) = event_info.iter().find(|ev| ev.0.ids.contains(&id)).unwrap();
                    let sample_value = format!("{}", event_count);

                    writer.encode(&[name, time.as_str(), cpu.as_str(), sample_value.as_str()])
                        .unwrap();
                }

            }
            _ => unreachable!("Should not happen"),
        }
    }

    Ok(())
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
enum Filter {
    All,
    Exclusive,
    Shared,
    None,
}

impl Filter {
    fn new(what: &str) -> Filter {
        match what {
            "all" => Filter::All,
            "exclusive" => Filter::Exclusive,
            "shared" => Filter::Shared,
            "none" => Filter::None,
            _ => panic!("clap-rs should ensure nothing else is passed..."),
        }
    }
}

pub fn extract(path: &Path, cpu_filter: &str, uncore_filter: &str, save_to: &Path) {
    if !path.exists() {
        error!("Input directory does not exist {:?}", path);
        process::exit(1);
    }

    let mut run_config: PathBuf = path.to_path_buf();
    run_config.push("run.toml");
    let mut file = File::open(run_config.as_path()).expect("run.toml file does not exist?");
    let mut run_string = String::new();
    let _ = file.read_to_string(&mut run_string).unwrap();
    let mut parser = toml::Parser::new(run_string.as_str());
    let doc = match parser.parse() {
        Some(doc) => doc,
        None => {
            error!("Can't parse the run.toml file:\n{:?}", parser.errors);
            process::exit(3);
        }
    };

    let a: &toml::Table = doc["a"].as_table().expect("run.toml: 'a' should be a table.");
    let deployment: &toml::Table = doc.get("deployment")
        .expect("deployment?")
        .as_table()
        .expect("run.toml: 'a.deployment' should be a table.");
    let cpus: Vec<u64> = deployment.get("a")
        .expect("deployment.a")
        .as_slice()
        .expect("run.tom: 'a.deployment.a' should be an array")
        .iter()
        .map(|c| c.as_table().expect("table")["cpu"].as_integer().expect("int") as u64)
        .collect();
    let breakpoints: Vec<String> = a.get("breakpoints")
        .expect("no breakpoints?")
        .as_slice()
        .expect("breakpoints not an array?")
        .iter()
        .map(|s| s.as_str().expect("breakpoint not a string?").to_string())
        .collect();

    let mut lscpu_file: PathBuf = path.to_path_buf();
    lscpu_file.push("lscpu.csv");
    let mut numactl_file: PathBuf = path.to_path_buf();
    numactl_file.push("numactl.dat");

    let mt = MachineTopology::from_files(&lscpu_file, &numactl_file);
    // All the CPUs this program is (exclusively) running on:
    let all_cpus: Vec<&CpuInfo> = cpus.into_iter()
        .map(|c| mt.cpu(c).expect("Invalid CPU in run.toml or wrong lscpu.csv?"))
        .collect();

    // All the sockets this program is running on:
    let mut all_sockets: Vec<Socket> = all_cpus.iter().map(|c| c.socket).collect();
    all_sockets.sort();
    all_sockets.dedup();

    let uncore_filter = Filter::new(uncore_filter);
    let cpu_filter = Filter::new(cpu_filter);

    let mut considered_sockets: Vec<Socket> = Vec::new();
    // Find out if we should include the uncore events for every socket that we're running on
    match uncore_filter {
        Filter::Exclusive => {
            for socket in all_sockets.into_iter() {
                let socket_set: HashSet<Cpu> =
                    mt.cpus_on_socket(socket).iter().map(|c| c.cpu).collect();
                let program_set: HashSet<Cpu> = all_cpus.iter().map(|c| c.cpu).collect();
                let diff: Vec<Cpu> = socket_set.difference(&program_set).cloned().collect();

                if diff.len() == 0 {
                    debug!("Uncore from socket {:?} considered since A uses it exclusively.",
                           socket);
                    considered_sockets.push(socket);
                }
            }
        }
        Filter::All => considered_sockets.append(&mut mt.sockets()),
        Filter::Shared => {
            debug!("Uncore from sockets {:?} added since A uses these sockets at least partially.",
                   all_sockets);
            considered_sockets.append(&mut all_sockets);
        }
        Filter::None => debug!("Ignore all uncore events."),
    };

    // Read perf.csv file:
    let mut csv_data: PathBuf = path.to_owned();
    csv_data.push("perf.csv");
    let csv_data_path = csv_data.as_path();
    if !csv_data_path.exists() {
        error!("File not found: {:?}", csv_data_path);
        return;
    }
    type Row = (String, String, String, String, String, String);
    let mut rdr = csv::Reader::from_file(csv_data_path).unwrap();
    let rows = rdr.decode().collect::<csv::Result<Vec<Row>>>().unwrap();

    // Create result.csv file:
    let csv_result: PathBuf = save_to.to_owned();
    let mut wrtr = csv::Writer::from_file(csv_result.as_path()).unwrap();
    wrtr.encode(&["EVENT_NAME",
                  "INDEX",
                  "TIME",
                  "SOCKET",
                  "CORE",
                  "CPU",
                  "NODE",
                  "UNIT",
                  "SAMPLE_VALUE"])
        .unwrap();

    // Write content in result.csv
    for row in rows {
        let (_, event_names, _, _, file, _) = row;
        let _string_names: Vec<&str> = event_names.split(",").collect();

        let mut perf_data = path.to_owned();
        perf_data.push(&file);

        let file_ext = perf_data.extension().expect("File does not have an extension");
        match file_ext.to_str().unwrap() {
            "data" => {
                parse_perf_file(perf_data.as_path(),
                                event_names.split(",").collect(),
                                &mut wrtr)
                    .unwrap()
            }
            "csv" => {
                parse_perf_csv_file(&mt,
                                    &all_cpus,
                                    cpu_filter,
                                    &considered_sockets,
                                    &breakpoints,
                                    perf_data.as_path(),
                                    &mut wrtr)
                    .unwrap()
            }
            _ => panic!("Unknown file extension, I can't parse this."),
        };
    }
}
