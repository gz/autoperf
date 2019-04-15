use std::io;
use std::io::prelude::*;
use std::fs;
use std::fs::File;
use std::path::PathBuf;
use std::path::Path;
use std::str::{FromStr, from_utf8_unchecked};
use std::process::{Command, Output};
use log::error as lerror;
use nom::*;
use x86::cpuid;
use csv;
use itertools::*;


pub type Node = u64;
pub type Socket = u64;
pub type Core = u64;
pub type Cpu = u64;
pub type L1 = u64;
pub type L2 = u64;
pub type L3 = u64;
pub type Online = u64;
pub type MHz = u64;

pub fn mkdir(out_dir: &Path) {
    if !out_dir.exists() {
        fs::create_dir(out_dir).expect("Can't create directory");
    }
}

fn to_string(s: &[u8]) -> &str {
    unsafe { from_utf8_unchecked(s) }
}

fn to_u64(s: &str) -> u64 {
    FromStr::from_str(s).unwrap()
}

fn buf_to_u64(s: &[u8]) -> u64 {
    to_u64(to_string(s))
}

named!(parse_numactl_size<&[u8], NodeInfo>,
    chain!(
        tag!("node") ~
        take_while!(is_space) ~
        node: take_while!(is_digit) ~
        take_while!(is_space) ~
        tag!("size:") ~
        take_while!(is_space) ~
        size: take_while!(is_digit) ~
        take_while!(is_space) ~
        tag!("MB"),
        || NodeInfo { node: buf_to_u64(node), memory: buf_to_u64(size) * 1000000 }
    )
);

fn get_node_info(node: Node, numactl_output: &String) -> Option<NodeInfo> {
    let find_prefix = format!("node {} size:", node);
    for line in numactl_output.split('\n') {
        if line.starts_with(find_prefix.as_str()) {
            let res = parse_numactl_size(line.as_bytes());
            return Some(res.unwrap().1);
        }
    }

    None
}

#[derive(Debug, Eq, PartialEq, RustcEncodable)]
pub struct CpuInfo {
    pub node: NodeInfo,
    pub socket: Socket,
    pub core: Core,
    pub cpu: Cpu,
    pub l1: L1,
    pub l2: L2,
    pub l3: L3,
}

impl CpuInfo {
    pub fn cbox(&self, mt: &MachineTopology) -> String {
        let cbox = self.core % mt.cores_on_socket(self.socket).len() as u64;
        format!("uncore_cbox_{}", cbox)
    }
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Copy, Clone, RustcEncodable)]
pub struct NodeInfo {
    pub node: Node,
    pub memory: u64,
}

#[derive(Debug)]
pub struct MachineTopology {
    data: Vec<CpuInfo>,
}

fn save_file(cmd: &'static str,
             output_path: &Path,
             file: &'static str,
             out: Output)
             -> io::Result<String> {
    if out.status.success() {
        // Save to result directory:
        let mut out_file: PathBuf = output_path.to_path_buf();
        out_file.push(file);
        let mut f = try!(File::create(out_file.as_path()));
        let content = String::from_utf8(out.stdout).unwrap_or(String::new());
        try!(f.write(content.as_bytes()));
        Ok(content)
    } else {
        lerror!("{} command: got unknown exit status was: {}",
               cmd,
               out.status);
        debug!("stderr:\n{}",
               String::from_utf8(out.stderr).unwrap_or("Can't parse output".to_string()));
        unreachable!()
    }
}

pub fn save_lstopo(output_path: &Path) -> io::Result<String> {
    let out = try!(Command::new("lstopo").arg("--of console").arg("--taskset").output());
    save_file("lstopo", output_path, "lstopo.txt", out)
}

pub fn save_cpuid(output_path: &Path) -> io::Result<String> {
    let out = try!(Command::new("cpuid").output());
    save_file("cpuid", output_path, "cpuid.txt", out)
}

pub fn save_likwid_topology(output_path: &Path) -> io::Result<String> {
    let out = try!(Command::new("likwid-topology").arg("-gc").output());
    save_file("likwid-topology", output_path, "likwid_topology.txt", out)
}

pub fn save_numa_topology(output_path: &Path) -> io::Result<String> {
    let out = try!(Command::new("numactl").arg("--hardware").output());
    save_file("numactl", output_path, "numactl.dat", out)
}

pub fn save_cpu_topology(output_path: &Path) -> io::Result<String> {
    let out = try!(Command::new("lscpu").arg("--parse=NODE,SOCKET,CORE,CPU,CACHE").output());
    save_file("lscpu", output_path, "lscpu.csv", out)
}

impl MachineTopology {
    pub fn new() -> MachineTopology {
        let lscpu_out =
            Command::new("lscpu").arg("--parse=NODE,SOCKET,CORE,CPU,CACHE").output().unwrap();
        let lscpu_string = String::from_utf8(lscpu_out.stdout).unwrap_or(String::new());

        let numactl_out = Command::new("numactl").arg("--hardware").output().unwrap();
        let numactl_string = String::from_utf8(numactl_out.stdout).unwrap_or(String::new());

        MachineTopology::from_strings(lscpu_string, numactl_string)
    }

    pub fn from_files(lcpu_path: &Path, numactl_path: &Path) -> MachineTopology {
        let mut file = File::open(lcpu_path).expect("lscpu.csv file does not exist?");
        let mut lscpu_string = String::new();
        let _ = file.read_to_string(&mut lscpu_string).unwrap();

        let mut file = File::open(numactl_path).expect("numactl.dat file does not exist?");
        let mut numactl_string = String::new();
        let _ = file.read_to_string(&mut numactl_string).unwrap();

        MachineTopology::from_strings(lscpu_string, numactl_string)
    }

    pub fn from_strings(lscpu_output: String, numactl_output: String) -> MachineTopology {
        let no_comments: Vec<&str> = lscpu_output.split('\n')
            .filter(|s| s.trim().len() > 0 && !s.trim().starts_with("#"))
            .collect();

        type Row = (Node, Socket, Core, Cpu, String); // Online MHz
        let mut rdr = csv::Reader::from_string(no_comments.join("\n")).has_headers(false);
        let rows = rdr.decode().collect::<csv::Result<Vec<Row>>>().unwrap();

        let mut data: Vec<CpuInfo> = Vec::with_capacity(rows.len());
        for row in rows {
            let caches: Vec<u64> = row.4.split(":").map(|s| u64::from_str(s).unwrap()).collect();
            assert_eq!(caches.len(), 4);
            let node: NodeInfo = get_node_info(row.0, &numactl_output)
                .expect("Can't find node in numactl output?");
            let tuple: CpuInfo = CpuInfo {
                node: node,
                socket: row.1,
                core: row.2,
                cpu: row.3,
                l1: caches[0],
                l2: caches[2],
                l3: caches[3],
            };
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

    pub fn cpu(&self, cpu: Cpu) -> Option<&CpuInfo> {
        self.data.iter().find(|t| t.cpu == cpu)
    }

    pub fn cores(&self) -> Vec<Core> {
        let mut cores: Vec<Core> = self.data.iter().map(|t| t.core).collect();
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

    pub fn nodes(&self) -> Vec<NodeInfo> {
        let mut nodes: Vec<NodeInfo> = self.data.iter().map(|t| t.node).collect();
        nodes.sort();
        nodes.dedup();
        nodes
    }

    pub fn max_memory(&self) -> u64 {
        self.nodes().iter().map(|t| t.memory).sum()
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
            let cache = cparams.find(|c| c.level() == 1 && c.cache_type() == cpuid::CacheType::DATA)
                .unwrap();
            (cache.associativity() * cache.physical_line_partitions() *
             cache.coherency_line_size() * cache.sets()) as u64
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
            let cache =
                cparams.find(|c| c.level() == 2 && c.cache_type() == cpuid::CacheType::UNIFIED)
                    .unwrap();
            (cache.associativity() * cache.physical_line_partitions() *
             cache.coherency_line_size() * cache.sets()) as u64
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
            let cache =
                cparams.find(|c| c.level() == 3 && c.cache_type() == cpuid::CacheType::UNIFIED)
                    .unwrap();
            (cache.associativity() * cache.physical_line_partitions() *
             cache.coherency_line_size() * cache.sets()) as u64
        })
    }

    pub fn cpus_on_node(&self, node: NodeInfo) -> Vec<&CpuInfo> {
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

    fn cores_on_socket(&self, socket: Socket) -> Vec<Core> {
        let mut cores: Vec<Core> =
            self.data.iter().filter(|c| c.socket == socket).map(|c| c.core).collect();
        cores.sort();
        cores.dedup();
        cores
    }

    fn cores_on_l3(&self, l3: L3) -> Vec<&CpuInfo> {
        let mut cpus: Vec<&CpuInfo> = self.data.iter().filter(|t| t.l3 == l3).collect();
        cpus.sort_by_key(|c| c.core);
        // TODO: implicit assumption that we have two HTs
        cpus.into_iter().step(2).collect()
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

    pub fn same_l3_cores(&self) -> Vec<Vec<&CpuInfo>> {
        self.l3().into_iter().map(|l3| self.cores_on_l3(l3)).collect()
    }

    pub fn whole_machine(&self) -> Vec<Vec<&CpuInfo>> {
        vec![self.data.iter().collect()]
    }

    pub fn whole_machine_cores(&self) -> Vec<Vec<&CpuInfo>> {
        let mut cpus: Vec<&CpuInfo> = self.data.iter().collect();
        cpus.sort_by_key(|c| c.core);
        // TODO: implicit assumption that we have two HTs
        vec![cpus.into_iter().step(2).collect()]
    }
}

// TODO: Should ideally be generic:
pub fn socket_uncore_devices() -> Vec<&'static str> {
    vec!["uncore_ha_0",
         "uncore_imc_0",
         "uncore_imc_1",
         "uncore_imc_2",
         "uncore_imc_3",
         "uncore_pcu",
         "uncore_r2pcie",
         "uncore_r3qpi_0",
         "uncore_r3qpi_1",
         "uncore_ubox"]
}
