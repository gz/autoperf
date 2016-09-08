extern crate perfcnt;

use std::io::prelude::*;
use std::fs::File;
use std::path::Path;

use perfcnt::linux::perf_file::PerfFile;

pub fn extract(path: &Path) {
    println!("Parsed perf file: {:?}", path);
    let mut file = File::open(path).expect("File does not exist");
    let mut buf: Vec<u8> = Vec::with_capacity(2*4096*4096);

    match file.read_to_end(&mut buf) {
        Ok(_) => {
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
        }
        Err(e) => {
            panic!("Can't read {:?}: {}", file, e);
        }
    }

}
