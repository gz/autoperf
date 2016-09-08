use std::io;
use std::io::prelude::*;
use std::fs;
use std::fs::{File, Metadata};
use std::path::Path;
use std::path::PathBuf;
use csv;

use perfcnt::linux::perf_file::PerfFile;

fn parse_perf_file(path: &Path) -> io::Result<()> {
    let meta: Metadata = try!(fs::metadata(path));
    if !meta.file_type().is_file() {
        error!("Not a file {:?}", path);
    }

    let mut file = try!(File::open(path));
    let mut buf: Vec<u8> = Vec::with_capacity(meta.len() as usize);
    try!(file.read_to_end(&mut buf));

    let pf = PerfFile::new(buf);
    println!("Header: {:?}", pf.header);
    println!("Attributes: {:?}", pf.attrs);
    println!("BuildId: {:?}", pf.get_build_id());
    println!("Hostname: {:?}", pf.get_hostname());
    println!("OS Release: {:?}", pf.get_os_release());
    println!("Version: {:?}", pf.get_version());
    println!("Arch: {:?}", pf.get_arch());
    println!("NrCpus: {:?}", pf.get_nr_cpus());
    println!("CpuDesc: {:?}", pf.get_cpu_description());
    println!("CpuId: {:?}", pf.get_cpu_id());
    println!("TotalMemory: {:?}", pf.get_total_memory());
    println!("CmdLine: {:?}", pf.get_cmd_line());
    println!("EventDescription: {:?}", pf.get_event_description());
    println!("CpuTopology: {:?}", pf.get_cpu_topology());
    println!("NumaTopology: {:?}", pf.get_numa_topology());
    println!("PmuMappings: {:?}", pf.get_pmu_mappings());
    println!("GroupDescriptions: {:?}", pf.get_group_descriptions());
    println!("-----------------------------------------------------");
    for e in pf.data() {
        println!("{:?}", e);
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

    type Row = (String, String, String, String);
    let mut rdr = csv::Reader::from_file(csv_data_path).unwrap();
    let rows = rdr.decode().collect::<csv::Result<Vec<Row>>>().unwrap();
    for row in rows {
        let (cmd, events, bps, file) = row;
        let mut perf_data = path.to_owned();
        perf_data.push(&file);

        match parse_perf_file(perf_data.as_path()).map(|res| {  }) {
            Ok(res) => { debug!("Successfully got dat a from {}", &file); }
            Err(e) => { error!("Unable to read file {}: {}", &file, e); }
        }
    }
}
