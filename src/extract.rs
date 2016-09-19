use std::io;
use std::io::prelude::*;
use std::fs;
use std::fs::{File, Metadata};
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use std::process;

use csv;

use perfcnt::linux::perf_file::PerfFile;
use perfcnt::linux::perf_format::*;

// I have no idea if the perf format guarantees that events appear always in the same order :S
fn verify_events_in_order(events: &Vec<EventDesc>, values: &Vec<(u64, Option<u64>)>) -> bool {
    for (idx, v) in values.iter().enumerate() {
        // Don't have id's we can't veryify anything
        if v.1.is_none() {
            warn!("Don't have IDs with the sample values, so we can't tell which event a sample belongs to.");
            return true;
        }

        let id: u64 = v.1.unwrap_or(0);
        if !events.get(idx).map_or(false, |ev| { ev.ids.contains(&id) }) {
            return false;
        }
    }

    return true;
}

/// Extracts the perf stat file and writes it to a CSV file that looks like this:
/// EVENT_NAME, TIME, CPU, SAMPLE_VALUE
fn parse_perf_csv_file(path: &Path, writer: &mut csv::Writer<File>) -> io::Result<()> {
    // Check if it's a file:
    let meta: Metadata = try!(fs::metadata(path));
    if !meta.file_type().is_file() {
        error!("Not a file {:?}", path);
    }

    type Row = (f64, String, String, String, String, String, f64);
    let mut rdr = csv::Reader::from_file(path).unwrap().has_headers(false).delimiter(b';').flexible(true);

    for record in rdr.decode() {
        if record.is_ok() {
            let (time, cpu, value_string, _, event, _, percent): Row = record.expect("Should not happen (in is_ok() branch)!");

            if value_string.trim() == "<not counted>" {
                error!("Event {} was not measured. Abort for now...", event);
                // Maybe it's better to handle this by removing the event from all written rows
                // and log this as a warning...
                process::exit(1);
            }
            let value = u64::from_str(value_string.trim()).expect("Should be a value now...");

            if percent < 91.0 {
                warn!("Multiplexed counters: {}", event.trim());
            }
            if percent == 0.0 {
                error!("Event {} was not measured at all?", event.trim());
                process::exit(2);
            }

            // Perf will just report CPU0 for uncore events, so we temporarily encode the location
            // in the event name and extract it here again:
            let (location, name) = if !event.starts_with("uncore_") {
                // Normal case, we just take the regular event and cpu fields from perf stat
                (cpu.trim(), event.trim())
            } else {
                // Uncore events, use first part of the event name as the location
                let (location, name) = event.split_at(event.find(".").unwrap());
                (location, name.trim_left_matches(".").trim())
            };

            writer.encode(&[ name, time.to_string().as_str(), location, value.to_string().as_str() ]).unwrap();
        }
        else {
            match record.unwrap_err() {
                csv::Error::Decode(s) => if !s.starts_with("Failed converting '#") { panic!("Can't decode line {}.", s) },
                e => panic!("Unrecoverable error {} while decoding.", e)
            };
        }
    }

    Ok(())
}

/// Extracts the data and writes it to a CSV file that looks like this:
/// EVENT_NAME, TIME, CPU, SAMPLE_VALUE
fn parse_perf_file(path: &Path, event_names: Vec<&str>, writer: &mut csv::Writer<File>) -> io::Result<()> {

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

    //debug!("GroupDescriptions: {:?}", pf.get_group_descriptions());
    //debug!("EventDescription: {:?}", pf.get_event_description());

    let event_desc = pf.get_event_description().unwrap();
    let event_info: Vec<(&EventDesc, &&str)> = event_desc.iter().zip(event_names.iter()).collect();
    //debug!("Event Infos: {:?}", event_info);


    for e in pf.data() {
        if e.header.event_type != EventType::Sample {
            continue
        }

        match e.data {
            EventData::Sample(rec) => {
                //println!("{:?}", rec);
                let time = format!("{}", rec.time.unwrap());
                let ptid = rec.ptid.unwrap();
                let pid = format!("{}", ptid.pid);
                let tid = format!("{}", ptid.tid);
                let cpu = format!("{}", rec.cpu.unwrap().cpu);
                //let ip = format!("0x{:x}", rec.ip.unwrap());

                let v = rec.v.unwrap();
                assert!(verify_events_in_order(&event_desc, &v.values));
                // TODO: verify event names match EventDesc in `event_info`!

                for reading in v.values.iter() {
                    let (event_count, maybe_id) = *reading;
                    let id = maybe_id.unwrap();
                    let &(_, name) = event_info.iter().find(|ev| ev.0.ids.contains(&id)).unwrap();
                    let sample_value = format!("{}", event_count);

                    writer.encode(&[ name, time.as_str(), pid.as_str(), tid.as_str(), cpu.as_str(), sample_value.as_str() ]).unwrap();
                }

            }
            _ => unreachable!("Should not happen")
        }
    }

    Ok(())
}

pub fn extract(path: &Path) {
    debug!("Looking for perf.csv in {:?}", path);

    if !path.exists() {
        error!("Input directory does not exist {:?}", path);
        return;
    }

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

    let mut csv_result: PathBuf = path.to_owned();
    csv_result.push("result.csv");

    let mut wrtr = csv::Writer::from_file(csv_result.as_path()).unwrap();
    wrtr.encode(&["EVENT_NAME", "TIME", "CPU", "SAMPLE_VALUE"]).unwrap();

    for row in rows {
        //println!("{:?}", row);
        let (_, event_names, _, _, file, _) = row;
        let string_names: Vec<&str> = event_names.split(",").collect();
        debug!("Processing: {}", string_names.join(", "));

        let mut perf_data = path.to_owned();
        perf_data.push(&file);

        let file_ext = perf_data.extension().expect("File does not have an extension");
        match file_ext.to_str().unwrap() {
            "data" => parse_perf_file(perf_data.as_path(), event_names.split(",").collect(), &mut wrtr).unwrap(),
            "csv" => parse_perf_csv_file(perf_data.as_path(), &mut wrtr).unwrap(),
            _ => panic!("Unknown file extension, I can't parse this.")
        };
    }
}
