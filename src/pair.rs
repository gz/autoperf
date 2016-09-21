use std::io;
use std::io::prelude::*;
use std::fs;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::process::{Command, Child, Stdio};
use std::str::{FromStr, from_utf8_unchecked};
use std::fmt;

use x86::shared::{cpuid};
use nom::*;
use csv;
use yaml_rust::{YamlLoader};

use profile;

pub type Node = u64;
pub type Socket = u64;
pub type Core = u64;
pub type Cpu = u64;
pub type L1 = u64;
pub type L2 = u64;
pub type L3 = u64;
pub type Online = u64;
pub type MHz = u64;


fn get_hostname() -> Option<String> {
    use libc::{gethostname, c_char, size_t, c_int};

    let mut buf: [i8; 64] = [0; 64];
    let err = unsafe { gethostname (buf.as_mut_ptr(), buf.len()) };

    if err != 0 {
        info!("Can't read the hostname with gethostname: {}", io::Error::last_os_error());
        return None;
    }

    // find the first 0 byte (i.e. just after the data that gethostname wrote)
    let actual_len = buf.iter().position(|byte| *byte == 0).unwrap_or(buf.len());
    let c_str: Vec<u8> = buf[..actual_len].into_iter().map(|i| *i as u8).collect();

    Some( String::from_utf8(c_str).unwrap() )
}

fn mkdir(out_dir: &PathBuf) {
    if !out_dir.exists() {
        fs::create_dir(out_dir.as_path()).expect("Can't create `out` directory");
    }
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Copy, Clone)]
pub struct NodeInfo {
    node: Node,
    memory: u64
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


#[derive(Debug, Eq, PartialEq)]
pub struct CpuInfo {
    node: NodeInfo,
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

    pub fn new(lscpu_output: String, numactl_output: String) -> MachineTopology {
        let no_comments: Vec<&str> = lscpu_output.split('\n').filter(|s| s.trim().len() > 0 && !s.trim().starts_with("#")).collect();
        type Row = (Node, Socket, Core, Cpu, String); // Online MHz

        let mut rdr = csv::Reader::from_string(no_comments.join("\n")).has_headers(false);
        let rows = rdr.decode().collect::<csv::Result<Vec<Row>>>().unwrap();

        let mut data: Vec<CpuInfo> = Vec::with_capacity(rows.len());
        for row in rows {
            let caches: Vec<u64> = row.4.split(":").map(|s| u64::from_str(s).unwrap()).collect();
            assert_eq!(caches.len(), 4);
            let node: NodeInfo = get_node_info(row.0, &numactl_output).expect("Can't find node in numactl output?");
            let tuple: CpuInfo = CpuInfo { node: node, socket: row.1, core: row.2, cpu: row.3, l1: caches[0], l2: caches[2], l3: caches[3] };
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

fn save_numa_topology(output_path: &Path) -> io::Result<String> {
    let out = try!(Command::new("numactl").arg("--hardware").output());
    if out.status.success() {
        // Save to result directory:
        let mut numactl_file: PathBuf = output_path.to_path_buf();
        numactl_file.push("numactl.dat");
        let mut f = try!(File::create(numactl_file.as_path()));
        let content = String::from_utf8(out.stdout).unwrap_or(String::new());
        try!(f.write(content.as_bytes()));
        Ok(content)
    }
    else {
        error!("numactl command: got unknown exit status was: {}", out.status);
        debug!("stderr:\n{}", String::from_utf8(out.stderr).unwrap_or("Can't parse output".to_string()));
        unreachable!()
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

struct Deployment<'a> {
    description: &'static str,
    a: Vec<&'a CpuInfo>,
    b: Vec<&'a CpuInfo>,
    mem: Vec<NodeInfo>
}

impl<'a> Deployment<'a> {
    pub fn split(desc: &'static str, possible_groupings: Vec<Vec<&'a CpuInfo>>, size: u64, avoid_smt: bool) -> Deployment<'a> {
        let mut cpus = possible_groupings.into_iter().last().unwrap();

        if avoid_smt {
            // Find all the cores:
            let mut cores: Vec<Cpu> = cpus.iter().map(|t| t.core).collect();
            cores.sort();
            cores.dedup();
            assert!(cores.len() == cpus.len() / 2); // Assume we have 2 SMT per core

            // Pick a CpuInfo for every core:
            let mut to_remove: Vec<usize> = Vec::with_capacity(cores.len());
            for core in cores.into_iter() {
                for cpu in cpus.iter() {
                    if cpu.core == core {
                        to_remove.push(core as usize);
                        break;
                    }
                }
            }

            // Remove one of the hyper-thread pairs:
            for idx in to_remove {
                cpus.remove(idx);
            }
        }

        let cpus_len = cpus.len();
        assert!(cpus_len % 2 == 0);

        let upper_half = cpus.split_off(cpus_len / 2);
        let lower_half = cpus;

        let mut node: NodeInfo = lower_half[0].node;
        node.memory = size; //as f64 * 0.95;

        Deployment { description: desc, a: lower_half, b: upper_half, mem: vec![node] }
    }
}

impl<'a> fmt::Display for Deployment<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let a: Vec<Cpu> = self.a.iter().map(|c| c.cpu).collect();
        let b: Vec<Cpu> = self.b.iter().map(|c| c.cpu).collect();

        try!(write!(f, "Deployment Plan for {}:\n", self.description));
        try!(write!(f, "-- Program A cores: {:?}\n", a));
        try!(write!(f, "-- Program B cores: {:?}\n", b));
        try!(write!(f, "-- Use memory:\n"));
        for n in self.mem.iter() {
            try!(write!(f, " - On node {}: {} Bytes\n", n.node, n.memory));
        }
        Ok(())
    }
}

struct Run<'a> {
    output_path: PathBuf,
    binary_a: &'a str,
    args_a: &'a Vec<&'a str>,
    is_openmp_a: bool,

    child_b: Option<Child>,
    binary_b: &'a str,
    args_b: &'a Vec<&'a str>,
    is_openmp_b: bool,

    deployment: &'a Deployment<'a>
}

impl<'a> Run<'a> {
    fn new(output_path: &Path, a: &'a str, args_a: &'a Vec<&str>, b: &'a str, args_b: &'a Vec<&str>, deployment: &'a Deployment) -> Run<'a> {
        let mut out_dir = output_path.to_path_buf();
        out_dir.push(deployment.description);
        mkdir(&out_dir);

        Run { output_path: out_dir,
              binary_a: a, args_a: args_a, is_openmp_a: true,
              binary_b: b, args_b: args_b, is_openmp_b: true,
              deployment: deployment, child_b: None }
    }

    fn get_env_for_a(&self) -> Vec<(String, String)> {
        let mut env: Vec<(String, String)> = Vec::with_capacity(2);
        if self.is_openmp_a {
            let cpus: Vec<String> = self.deployment.a.iter().map(|c| format!("{}", c.cpu)).collect();
            println!("{:?}", cpus);
            env.push( (String::from("GOMP_CPU_AFFINITY"), format!("\"{}\"", cpus.join(" "))) );
        }

        env
    }

    fn get_args_for_a(&self) -> Vec<String> {
        let nthreads = self.deployment.a.len();

        self.args_a.iter()
            .map(|s| s.to_string())
            .map(|s| {
                s.replace("$NUM_THREADS", format!("{}", nthreads).as_str())
        }).collect()
    }

    fn get_args_for_b(&self) -> Vec<String> {
        let nthreads = self.deployment.b.len();

        self.args_a.iter()
            .map(|s| s.to_string())
            .map(|s| {
                s.replace("$NUM_THREADS", format!("{}", nthreads).as_str())
        }).collect()
    }

    fn profile_a(&self) -> io::Result<()> {
        debug!("Starting profiling and running A: {}", self.binary_a);
        let mut perf_data_path_buf = self.output_path.clone();
        perf_data_path_buf.push("stat");
        mkdir(&perf_data_path_buf);
        let perf_path = perf_data_path_buf.as_path();

        let args = self.get_args_for_a();
        let mut cmd: Vec<&str> = vec![ self.binary_a ];
        cmd.extend( args.iter().map(|c| c.as_str() ));

        //println!("{:?}", self.get_env_for_a());

        profile::profile(&perf_path, cmd, self.get_env_for_a(), false);
        Ok(())
    }

    fn start_b(&mut self) -> io::Result<Child> {
        debug!("Starting B: {}", self.binary_b);
        Command::new(self.binary_a)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .args(self.get_args_for_b().as_slice())
                .spawn()
    }

    fn save_output<T: io::Read>(&self, filename: &str, what: &mut T) -> io::Result<()> {
        let mut stdout = String::new();
        what.read_to_string(&mut stdout);
        let mut stdout_path = self.output_path.clone();
        stdout_path.push(filename);
        let mut f = try!(File::create(stdout_path.as_path()));
        try!(f.write_all(stdout.as_bytes()));

        Ok(())
    }

    fn profile(&mut self) -> io::Result<()> {
        let mut deployment_path = self.output_path.clone();
        deployment_path.push("deployment.txt");
        let mut f = try!(File::create(deployment_path.as_path()));
        try!(f.write_all(format!("{}", self.deployment).as_bytes()));

        let mut app_b = try!(self.start_b());
        try!(self.profile_a());

        // Done, do clean-up:
        try!(app_b.kill());

        app_b.stdout.map(|mut c| { self.save_output("B_stdout.txt", &mut c) });
        app_b.stderr.map(|mut c| { self.save_output("B_stderr.txt", &mut c) });

        Ok(())
    }
}

pub fn pair(output_path: &Path) {
    let mut out_dir = output_path.to_path_buf();
    let hostname = get_hostname().unwrap_or(String::from("unknown"));
    out_dir.push(hostname);
    mkdir(&out_dir);

    let lscpu_string = save_cpu_topology(&out_dir).expect("Can't save CPU topology");
    let numactl_string = save_numa_topology(&out_dir).expect("Can't save NUMA topology");
    let mt = MachineTopology::new(lscpu_string, numactl_string);

    let mut manifest: PathBuf = output_path.to_path_buf();
    manifest.push("manifest.yml");

    let mut file = File::open(manifest.as_path()).expect("manifest file does not exist?");
    let mut s = String::new();

    let _ = file.read_to_string(&mut s).unwrap();
    let docs = YamlLoader::load_from_str(s.as_str()).unwrap();

    let doc = &docs[0];
    let configs: Vec<&str> = doc["configurations"].as_vec().unwrap().iter().map(|s| s.as_str().unwrap()).collect();

    let binary1: &str = doc["program1"]["binary"].as_str().unwrap();
    let args1: Vec<&str> = doc["program1"]["arguments"].as_vec().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
    let binary2: &str = doc["program2"]["binary"].as_str().unwrap();
    let args2: Vec<&str> = doc["program2"]["arguments"].as_vec().unwrap().iter().map(|s| s.as_str().unwrap()).collect();

    let mut deployments: Vec<Deployment> = Vec::with_capacity(4);
    for config in configs {
        if config == "Caches" {
            // Cache interference on HW threads
            deployments.push(Deployment::split("L1-SMT", mt.same_l1(), mt.l1_size().unwrap_or(0), false));
            deployments.push(Deployment::split("L2-SMT", mt.same_l2(), mt.l2_size().unwrap_or(0), false));

            // LLC
            deployments.push(Deployment::split("L3-SMT", mt.same_l3(), mt.l3_size().unwrap_or(0), false));
            deployments.push(Deployment::split("L3-no-SMT", mt.same_l3(), mt.l3_size().unwrap_or(0), true));
        }
        if config == "Memory" {
            warn!("Memory configuration not yet supported");
        }
    }

    for d in deployments.iter() {
        let mut run = Run::new(out_dir.as_path(), binary1, &args1, binary2, &args2, d);
        run.profile();
    }
}
