use std::process;
use std::io;
use std::io::prelude::*;
use std::fs;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::process::{Command, Child, Stdio};
use std::str::{FromStr, from_utf8_unchecked};
use std::fmt;
use rustc_serialize::Encodable;

use x86::shared::cpuid;
use csv;
use toml;

use profile;
use super::util::*;

fn get_hostname() -> Option<String> {
    use libc::{gethostname, c_char, size_t, c_int};

    let mut buf: [i8; 64] = [0; 64];
    let err = unsafe { gethostname(buf.as_mut_ptr(), buf.len()) };

    if err != 0 {
        info!("Can't read the hostname with gethostname: {}",
              io::Error::last_os_error());
        return None;
    }

    // find the first 0 byte (i.e. just after the data that gethostname wrote)
    let actual_len = buf.iter().position(|byte| *byte == 0).unwrap_or(buf.len());
    let c_str: Vec<u8> = buf[..actual_len].into_iter().map(|i| *i as u8).collect();

    Some(String::from_utf8(c_str).unwrap())
}

#[derive(Debug, RustcEncodable)]
struct Deployment<'a> {
    description: &'static str,
    a: Vec<&'a CpuInfo>,
    b: Vec<&'a CpuInfo>,
    mem: Vec<NodeInfo>,
}

impl<'a> Deployment<'a> {
    pub fn split(desc: &'static str,
                 possible_groupings: Vec<Vec<&'a CpuInfo>>,
                 size: u64,
                 avoid_smt: bool)
                 -> Deployment<'a> {
        let mut cpus = possible_groupings.into_iter().last().unwrap();

        if avoid_smt {
            // Find all the cores:
            let mut cores: Vec<Cpu> = cpus.iter().map(|t| t.core).collect();
            cores.sort();
            cores.dedup();
            assert!(cores.len() == cpus.len() / 2); // Assume we have 2 SMT per core

            // Pick a CpuInfo for every core:
            let mut to_remove: Vec<usize> = Vec::with_capacity(cores.len());
            for (idx, core) in cores.into_iter().enumerate() {
                for cpu in cpus.iter() {
                    if cpu.core == core {
                        to_remove.push(idx);
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

        Deployment {
            description: desc,
            a: lower_half,
            b: upper_half,
            mem: vec![node],
        }
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

#[derive(Debug, RustcEncodable)]
struct Program<'a> {
    name: String,
    manifest_path: &'a Path,
    binary: String,
    args: Vec<String>,
    antagonist_args: Vec<String>,
    breakpoints: Vec<String>,
    is_openmp: bool,
    alone: bool,
}

impl<'a> Program<'a> {

    fn from_toml(manifest_path: &'a Path, config: &toml::Table) -> Program<'a> {
        let name: String = config["name"].as_str()
                .expect("program.binary not a string").to_string();
        let binary: String = config["binary"].as_str()
                .expect("program.binary not a string").to_string();
        let openmp: bool = config["openmp"]
                .as_bool().expect("'program.openmp' should be boolean");
        let alone: bool = config["alone"]
                .as_bool().expect("'program.alone' should be boolean");
        let args: Vec<String> = config["arguments"]
                .as_slice().expect("program.arguments not an array?")
                .iter().map(|s| s.as_str().expect("program1 argument not a string?").to_string())
                .collect();
        let antagonist_args: Vec<String> = config["antagonist_arguments"]
                .as_slice().expect("program.antagonist_arguments not an array?")
                .iter().map(|s| s.as_str().expect("program1 argument not a string?").to_string())
                .collect();
        let breakpoints: Vec<String> = config["breakpoints"]
                .as_slice().expect("program.breakpoints not an array?")
                .iter().map(|s| s.as_str().expect("program breakpoint not a string?").to_string())
                .collect();

        Program { name: name, manifest_path: manifest_path, binary: binary, is_openmp: openmp, alone: alone,
                  args: args, antagonist_args: antagonist_args, breakpoints: breakpoints }
    }

    fn get_cmd(&self, antagonist: bool, cores: &Vec<&CpuInfo>) -> Vec<String> {
        let nthreads = cores.len();
        let mut cmd = vec![ &self.binary ];

        if !antagonist {
            cmd.extend(self.args.iter());
        }
        else {
            cmd.extend(self.antagonist_args.iter());
        }

        cmd.iter()
            .map(|s| s.replace("$NUM_THREADS", format!("{}", nthreads).as_str() ))
            .map(|s| s.replace("$MANIFEST_DIR", format!("{}", self.manifest_path.to_str().unwrap()).as_str()))
            .collect()
    }

    fn get_env(&self, antagonist: bool, cores: &Vec<&CpuInfo>) -> Vec<(String, String)> {
        let mut env: Vec<(String, String)> = Vec::with_capacity(2);
        if self.is_openmp {
            let cpus: Vec<String> = cores.iter().map(|c| format!("{}", c.cpu)).collect();
            env.push((String::from("OMP_PROC_BIND"), String::from("true")));
            env.push((String::from("OMP_PLACES"), format!("{{{}}}", cpus.join(","))));
        }

        env
    }
}

#[derive(RustcEncodable)]
struct Run<'a> {
    manifest_path: &'a Path,
    output_path: PathBuf,
    a: &'a Program<'a>,
    b: Option<&'a Program<'a>>,
    deployment: &'a Deployment<'a>,
}

impl<'a> Run<'a> {
    fn new(manifest_path: &'a Path,
           output_path: &'a Path,
           a: &'a Program<'a>,
           b: Option<&'a Program<'a>>,
           deployment: &'a Deployment)
           -> Run<'a> {
        let mut out_dir = output_path.to_path_buf();
        out_dir.push(deployment.description);
        mkdir(&out_dir);
        match b {
            Some(p) => out_dir.push(format!("{}_vs_{}", a.name, p.name)),
            None => out_dir.push(a.name.as_str()),
        }
        mkdir(&out_dir);

        Run {
            manifest_path: manifest_path,
            output_path: out_dir,
            a: a, b: b,
            deployment: deployment
        }
    }

    fn profile_a(&self) -> io::Result<()> {
        let cmd = self.a.get_cmd(false, &self.deployment.a);
        let env = self.a.get_env(false, &self.deployment.a);
        let bps = self.a.breakpoints.iter().map(|s| s.to_string()).collect();

        debug!("Spawning {:?} with environment {:?}", cmd, env);
        profile::profile(&self.output_path, cmd, env, bps, false);
        Ok(())
    }

    fn start_b(&mut self) -> Option<Child> {
        self.b.map(|b| {
            let command_args = b.get_cmd(true, &self.deployment.b);
            let env = b.get_env(true, &self.deployment.b);

            let ref name = command_args[0];
            debug!("Spawning {:?} with environment {:?}", command_args, env);
            let mut cmd = Command::new(name);
            let mut cmd = cmd.stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .args(&command_args[1..]);

            // Add the environment:
            for (key, value) in env{
                cmd.env(key, value);
            }

            match cmd.spawn() {
                Ok(child) => child,
                Err(_) => panic!("Can't spawn program B")
            }
        })
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

    fn save_run_information(&self) -> io::Result<()> {
        let mut run_toml_path = self.output_path.clone();
        run_toml_path.push("run.toml");
        let mut f = try!(File::create(run_toml_path.as_path()));
        let mut e = toml::Encoder::new();
        self.encode(&mut e).unwrap();
        try!(f.write_all(toml::encode_str(&e.toml).as_bytes()));

        let mut run_txt_path = self.output_path.clone();
        run_txt_path.push("run.txt");
        let mut f = try!(File::create(run_txt_path.as_path()));
        f.write_all(format!("{}", self).as_bytes())
    }

    fn profile(&mut self) -> io::Result<()> {
        self.save_run_information();

        // Profile together with B
        let mut maybe_app_b = self.start_b();

        try!(self.profile_a());

        match maybe_app_b {
            Some(mut app_b) => {
                // Done, do clean-up:
                try!(app_b.kill());
                app_b.stdout.map(|mut c| self.save_output("B_stdout.txt", &mut c));
                app_b.stderr.map(|mut c| self.save_output("B_stderr.txt", &mut c));
            },
            None => ()
        };

        Ok(())
    }
}

impl<'a> fmt::Display for Run<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "A: {:?} {:?}\n", self.a.get_env(false, &self.deployment.a), self.a.get_cmd(false, &self.deployment.a)));
        try!(write!(f, "A Breakpoints: {:?}\n", self.a.breakpoints));
        match self.b {
            Some(b) => {
                try!(write!(f, "B: {:?} {:?}\n", b.get_env(true, &self.deployment.b), b.get_cmd(true, &self.deployment.b)));
                try!(write!(f, "{}:\n", &self.deployment));
            }
            None => {
                try!(write!(f, "No other program running."));
            }
        }
        Ok(())
    }
}

pub fn pair(manifest_folder: &Path, dryrun: bool) {
    let mut out_dir = manifest_folder.to_path_buf();
    let hostname = get_hostname().unwrap_or(String::from("unknown"));
    out_dir.push(hostname);
    mkdir(&out_dir);

    let mt = MachineTopology::new();

    let mut manifest: PathBuf = manifest_folder.to_path_buf();
    manifest.push("manifest.toml");
    let mut file = File::open(manifest.as_path()).expect("manifest.toml file does not exist?");
    let mut manifest_string = String::new();
    let _ = file.read_to_string(&mut manifest_string).unwrap();
    let mut parser = toml::Parser::new(manifest_string.as_str());
    let doc = match parser.parse() {
        Some(doc) => {
            doc
        }
        None => {
            error!("Can't parse the manifest file:\n{:?}", parser.errors);
            process::exit(1);
        }
    };
    let experiment: &toml::Table = doc["experiment"].as_table().expect("Error in manifest.toml: 'experiment' should be a table.");
    let configuration: &[toml::Value] = experiment["configurations"].as_slice().expect("Error in manifest.toml: 'configuration' attribute should be an array.");
    let configs: Vec<String> = configuration.iter().map(|s| s.as_str().expect("configuration elements should be strings").to_string()).collect();
    let run_alone: bool = experiment["alone"].as_bool().expect("'alone' should be boolean");

    let mut programs: Vec<Program> = Vec::with_capacity(2);
    for (key, value) in &doc {
        if key.starts_with("program") {
            let program_desc: &toml::Table = doc[key].as_table().expect("Error in manifest.toml: 'program' should be a table.");
            programs.push(Program::from_toml(manifest_folder, program_desc));
        }
    }

    let mut deployments: Vec<Deployment> = Vec::with_capacity(4);
    for config in configs {

        // L1/L2 interference
        if config == "L1-SMT" {
            deployments.push(Deployment::split("L1-SMT", mt.same_l1(), mt.l1_size().unwrap_or(0), false));
        }
        if config == "L2-SMT" {
            deployments.push(Deployment::split("L2-SMT", mt.same_l2(), mt.l2_size().unwrap_or(0), false));
        }

        // LLC interference
        if config == "L3-SMT" {
            deployments.push(Deployment::split("L3-SMT", mt.same_l3(), mt.l3_size().unwrap_or(0), false));
        }
        if config == "L3-no-SMT" {
            deployments.push(Deployment::split("L3-no-SMT", mt.same_l3(),
                                               mt.l3_size().unwrap_or(0),
                                               true));
        }

        // Whole machine good/bad placements
    }

    // Run programs alone
    if run_alone {
        for a in programs.iter() {
            if !a.alone {
                continue;
            }

            for d in deployments.iter() {
                let mut run = Run::new(manifest_folder,
                                       out_dir.as_path(),
                                       a, None, d);
                if !dryrun {
                    run.profile();
                }
                else {
                    println!("{}", run);
                }
            }
        }
    }

    // Run programs pairwise together
    for (a, b) in iproduct!(programs.iter(), programs.iter()) {
        for d in deployments.iter() {
            let mut run = Run::new(manifest_folder,
                                   out_dir.as_path(),
                                   a, Some(b), d);
            if !dryrun {
                run.profile();
            }
            else {
                println!("{}", run);
            }
        }
    }
}
