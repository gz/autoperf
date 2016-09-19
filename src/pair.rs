use std::io;
use std::io::prelude::*;
//use std::fs;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;
use x86::shared::{cpuid};

use csv;
use yaml_rust::{YamlLoader};

pub type Node = u64;
pub type Socket = u64;
pub type Core = u64;
pub type Cpu = u64;
pub type L1 = u64;
pub type L2 = u64;
pub type L3 = u64;
pub type Online = u64;
pub type MHz = u64;

#[derive(Debug)]
pub struct CpuInfo {
    node: Node,
    socket: Socket,
    core: Core,
    cpu: Cpu,
    l1: L1,
    l2: L2,
    l3: L3
}

pub struct MachineTopology {
    data: Vec<CpuInfo>
}

impl MachineTopology {

    pub fn new(lscpu_output: String) -> MachineTopology {
        let no_comments: Vec<&str> = lscpu_output.split('\n').filter(|s| s.trim().len() > 0 && !s.trim().starts_with("#")).collect();
        type Row = (Node, Socket, Core, Cpu, String); // Online MHz

        let mut rdr = csv::Reader::from_string(no_comments.join("\n")).has_headers(false);
        let rows = rdr.decode().collect::<csv::Result<Vec<Row>>>().unwrap();

        let mut data: Vec<CpuInfo> = Vec::with_capacity(rows.len());
        for row in rows {
            let caches: Vec<u64> = row.4.split(":").map(|s| u64::from_str(s).unwrap()).collect();
            assert_eq!(caches.len(), 4);

            let tuple: CpuInfo = CpuInfo { node: row.0, socket: row.1, core: row.2, cpu: row.3, l1: caches[0], l2: caches[2], l3: caches[3] };
            data.push(tuple);
        }

        MachineTopology { data: data }
    }

    pub fn cpus(&self) -> Vec<Cpu> {
        let mut cpus: Vec<Cpu> = self.data.iter().map(|t| t.cpu).collect();
        cpus.sort();
        cpus.dedup();
        cpus
    }

    pub fn cores(&self) -> Vec<Cpu> {
        let mut cores: Vec<Cpu> = self.data.iter().map(|t| t.core).collect();
        cores.sort();
        cores.dedup();
        cores
    }

    pub fn sockets(&self) -> Vec<Socket> {
        let mut sockets: Vec<Cpu> = self.data.iter().map(|t| t.socket).collect();
        sockets.sort();
        sockets.dedup();
        sockets
    }

    pub fn nodes(&self) -> Vec<Node> {
        let mut nodes: Vec<Node> = self.data.iter().map(|t| t.node).collect();
        nodes.sort();
        nodes.dedup();
        nodes
    }

    pub fn l1(&self) -> Vec<L1> {
        let mut l1: Vec<L1> = self.data.iter().map(|t| t.l1).collect();
        l1.sort();
        l1.dedup();
        l1
    }

    pub fn l1_size(&self) -> Option<u64> {
        let cpuid = cpuid::CpuId::new();
        cpuid.get_cache_parameters().map(|mut cparams| {
                let cache = cparams.find(|c| { c.level() == 1 && c.cache_type() == cpuid::CacheType::DATA }).unwrap();
                (cache.associativity() * cache.physical_line_partitions() * cache.coherency_line_size() * cache.sets()) as u64
        })
    }

    pub fn l2(&self) -> Vec<L2> {
        let mut l2: Vec<L2> = self.data.iter().map(|t| t.l2).collect();
        l2.sort();
        l2.dedup();
        l2
    }

    pub fn l2_size(&self) -> Option<u64> {
        let cpuid = cpuid::CpuId::new();
        cpuid.get_cache_parameters().map(|mut cparams| {
                let cache = cparams.find(|c| { c.level() == 2 && c.cache_type() == cpuid::CacheType::UNIFIED }).unwrap();
                (cache.associativity() * cache.physical_line_partitions() * cache.coherency_line_size() * cache.sets()) as u64
        })
    }

    pub fn l3(&self) -> Vec<L3> {
        let mut l3: Vec<L3> = self.data.iter().map(|t| t.l3).collect();
        l3.sort();
        l3.dedup();
        l3
    }

    pub fn l3_size(&self) -> Option<u64> {
        let cpuid = cpuid::CpuId::new();
        cpuid.get_cache_parameters().map(|mut cparams| {
                let cache = cparams.find(|c| { c.level() == 3 && c.cache_type() == cpuid::CacheType::UNIFIED }).unwrap();
                (cache.associativity() * cache.physical_line_partitions() * cache.coherency_line_size() * cache.sets()) as u64
        })
    }

    pub fn cpus_on_node(&self, node: Node) -> Vec<&CpuInfo> {
        self.data.iter().filter(|t| t.node == node).collect()
    }

    pub fn cpus_on_l1(&self, l1: L1) -> Vec<&CpuInfo> {
        self.data.iter().filter(|t| t.l1 == l1).collect()
    }

    pub fn cpus_on_l2(&self, l2: L2) -> Vec<&CpuInfo> {
        self.data.iter().filter(|t| t.l2 == l2).collect()
    }

    pub fn cpus_on_l3(&self, l3: L3) -> Vec<&CpuInfo> {
        self.data.iter().filter(|t| t.l3 == l3).collect()
    }

    pub fn cpus_on_core(&self, core: Core) -> Vec<&CpuInfo> {
        self.data.iter().filter(|t| t.core == core).collect()
    }

    pub fn cpus_on_socket(&self, socket: Socket) -> Vec<&CpuInfo> {
        self.data.iter().filter(|t| t.socket == socket).collect()
    }

    pub fn same_socket(&self) -> Vec<Vec<&CpuInfo>> {
        self.sockets().into_iter().map(|s| self.cpus_on_socket(s)).collect()
    }

    pub fn same_core(&self) -> Vec<Vec<&CpuInfo>> {
        self.cores().into_iter().map(|c| self.cpus_on_core(c)).collect()
    }

    pub fn same_node(&self) -> Vec<Vec<&CpuInfo>> {
        self.nodes().into_iter().map(|c| self.cpus_on_node(c)).collect()
    }

    pub fn same_l1(&self) -> Vec<Vec<&CpuInfo>> {
        self.l1().into_iter().map(|c| self.cpus_on_l1(c)).collect()
    }

    pub fn same_l2(&self) -> Vec<Vec<&CpuInfo>> {
        self.l2().into_iter().map(|c| self.cpus_on_l2(c)).collect()
    }

    pub fn same_l3(&self) -> Vec<Vec<&CpuInfo>> {
        self.l3().into_iter().map(|c| self.cpus_on_l3(c)).collect()
    }
}

fn save_cpu_topology(output_path: &Path) -> io::Result<String> {
    let out = try!(Command::new("lscpu").arg("--parse=NODE,SOCKET,CORE,CPU,CACHE").output());
    if out.status.success() {
        // Save to result directory:
        let mut lscpu_file: PathBuf = output_path.to_path_buf();
        lscpu_file.push("lscpu.csv");
        let mut f = try!(File::create(lscpu_file.as_path()));
        let content = String::from_utf8(out.stdout).unwrap_or(String::new());
        {
            let no_comments: Vec<&str> = content.split('\n').filter(|s| s.trim().len() > 0 && !s.trim().starts_with("#")).collect();
            try!(f.write(no_comments.join("\n").as_bytes()));
        }

        Ok(content)
    }
    else {
        error!("lscpu command: got unknown exit status was: {}", out.status);
        debug!("stderr:\n{}", String::from_utf8(out.stderr).unwrap_or("Can't parse output".to_string()));
        unreachable!()
    }
}

pub fn pair(output_path: &Path) {
    let topo_string = save_cpu_topology(output_path).expect("Can't save CPU topology");
    let mt = MachineTopology::new(topo_string);

    println!("{:?} size: {:?}", mt.same_l1(), mt.l1_size());
    println!("{:?} size: {:?}", mt.same_l2(), mt.l2_size());
    println!("{:?} size: {:?}", mt.same_l3(), mt.l3_size());
    println!("{:?}", mt.same_socket());
    println!("{:?}", mt.same_node());
    println!("{:?}", mt.same_core());

    let mut manifest: PathBuf = output_path.to_path_buf();
    manifest.push("manifest.yml");

    let mut file = File::open(manifest.as_path()).expect("manifest file does not exist?");
    let mut s = String::new();

    let _ = file.read_to_string(&mut s).unwrap();
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
        if config == "Caches" {
            println!("L1 Interference:");
            println!("Run on cores {:?}", mt.same_l1().last().unwrap());
            println!("With memory sizes {:?} and {:?}", mt.l1_size().unwrap(), "xxx");


            println!("L2 Interference:");
            println!("{:?}", mt.same_l2().last().unwrap());

            println!("L3 Interference:");
            println!("{:?}", mt.same_l3().last().unwrap());
        }
    }

}
