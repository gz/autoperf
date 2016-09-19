use std;
use std::io;
use std::io::prelude::*;
//use std::fs;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::error::Error;

use csv;
use yaml_rust::{YamlLoader, YamlEmitter, Yaml};

type Node = u64;
type Socket = u64;
type Core = u64;
type CPU = u64;
type L1 = u64;
type L2 = u64;
type L3 = u64;
type Online = u64;
type MHz = u64;
type TopoTuple = (Node, Socket, Core, CPU, L1, L2, L3, Online, MHz);

struct MachineTopology {
    data: Vec<TopoTuple>
}

impl MachineTopology {

    pub fn new() -> io::Result<MachineTopology> {
        let res = try!(get_cpu_topology());
        println!("{:?}", res);
    }
}

fn get_cpu_topology() -> io::Result<Vec<u8>> {
    let mut out = try!(Command::new("lscpu").arg("--parse=NODE,SOCKET,CORE,CPU,CACHE,ONLINE,MAXMHZ").output());
    if out.status.success() {
        // Save to result directory:
        //let mut lscpu_file: PathBuf = output_path.to_path_buf();
        //lscpu_file.push("lscpu.csv");
        //let mut f = try!(File::create(lscpu_file.as_path()));
        //try!(f.write_all(out.stdout.as_slice()));

        return Ok(out.stdout)
    }
    else {
        error!("lscpu command: got unknown exit status was: {}", out.status);
        debug!("stderr:\n{}", String::from_utf8(out.stderr).unwrap_or("Can't parse output".to_string()));
        Ok(Vec::new())
    }
}

pub fn pair(output_path: &Path) {
    let mt = MachineTopology::new();

    let mut manifest: PathBuf = output_path.to_path_buf();
    manifest.push("manifest.yml");

    let mut file = File::open(manifest.as_path()).expect("manifest file does not exist?");
    let mut s = String::new();

    let input = file.read_to_string(&mut s).unwrap();
    let docs = YamlLoader::load_from_str(s.as_str()).unwrap();

    let doc = &docs[0];
    let configs: Vec<&str> = doc["configurations"].as_vec().unwrap().iter().map(|s| s.as_str().unwrap()).collect();

    let binary1: &str = doc["program1"]["binary"].as_str().unwrap();
    let args1: Vec<&str> = doc["program2"]["arguments"].as_vec().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
    let binary2: &str = doc["program2"]["binary"].as_str().unwrap();
    let args2: Vec<&str> = doc["program2"]["arguments"].as_vec().unwrap().iter().map(|s| s.as_str().unwrap()).collect();

    debug!("{:?}", binary1);
    debug!("{:?}", args1);
    debug!("{:?}", binary2);
    debug!("{:?}", args2);
    debug!("{:?}", configs);



    for config in configs {
        debug!("Config is {}", config);
    }

}
