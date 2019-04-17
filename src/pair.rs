use std::fs;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::path::Path;
use std::path::PathBuf;
use std::process;
use std::process::{Child, Command, Stdio};
use std::thread;

use std::fmt;

use itertools::{iproduct, Itertools};
use rustc_serialize::Encodable;
use std::time::Duration;
use wait_timeout::ChildExt;

use log::*;
use toml;

use super::profile;
use super::util::*;

fn get_hostname() -> Option<String> {
    use libc::gethostname;

    let mut buf: [i8; 64] = [0; 64];
    let err = unsafe { gethostname(buf.as_mut_ptr(), buf.len()) };

    if err != 0 {
        info!(
            "Can't read the hostname with gethostname: {}",
            io::Error::last_os_error()
        );
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
    pub fn new(
        desc: &'static str,
        half_a: Vec<&'a CpuInfo>,
        half_b: Vec<&'a CpuInfo>,
        mem: Vec<NodeInfo>,
    ) -> Deployment<'a> {
        Deployment {
            description: desc,
            a: half_a,
            b: half_b,
            mem: mem,
        }
    }

    /// Split by just simply interleaving everything
    /// TODO: this only works because we make assumption on how CpuInfo is ordered..
    pub fn split_interleaved(
        desc: &'static str,
        possible_groupings: Vec<Vec<&'a CpuInfo>>,
        size: u64,
    ) -> Deployment<'a> {
        let mut cpus = possible_groupings.into_iter().last().unwrap();

        let cpus_len = cpus.len();
        assert!(cpus_len % 2 == 0);

        let upper_half = cpus.split_off(cpus_len / 2);
        let lower_half = cpus;

        let mut node: NodeInfo = lower_half[0].node;
        node.memory = size;

        Deployment::new(desc, lower_half, upper_half, vec![node])
    }

    /// Split but makes sure a group shares the SMT threads
    pub fn split_smt_aware(
        desc: &'static str,
        possible_groupings: Vec<Vec<&'a CpuInfo>>,
        size: u64,
    ) -> Deployment<'a> {
        let cpus = possible_groupings.into_iter().last().unwrap();
        let cpus_len = cpus.len();
        assert!(cpus_len % 2 == 0);

        let mut cores: Vec<Core> = cpus.iter().map(|c| c.core).collect();
        assert!(cores.len() % 2 == 0);
        cores.sort();
        cores.dedup();

        let mut upper_half: Vec<&CpuInfo> = Vec::with_capacity(cpus_len / 2);
        let mut lower_half: Vec<&CpuInfo> = Vec::with_capacity(cpus_len / 2);

        for (i, core) in cores.into_iter().enumerate() {
            let cpus_on_core: Vec<&&CpuInfo> = cpus.iter().filter(|c| c.core == core).collect();
            if i % 2 == 0 {
                upper_half.extend(cpus_on_core.into_iter());
            } else {
                lower_half.extend(cpus_on_core.into_iter());
            }
        }

        let mut node: NodeInfo = lower_half[0].node;
        node.memory = size;

        Deployment::new(desc, lower_half, upper_half, vec![node])
    }

    /// Split but makes sure a group shares the SMT threads
    pub fn split_l3_aware(
        desc: &'static str,
        possible_groupings: Vec<Vec<&'a CpuInfo>>,
        size: u64,
    ) -> Deployment<'a> {
        let cpus = possible_groupings.into_iter().last().unwrap();
        let cpus_len = cpus.len();
        assert!(cpus_len % 2 == 0);

        let mut l3s: Vec<L3> = cpus.iter().map(|c| c.l3).collect();
        assert!(l3s.len() % 2 == 0);
        l3s.sort();
        l3s.dedup();

        let mut upper_half: Vec<&CpuInfo> = Vec::with_capacity(cpus_len / 2);
        let mut lower_half: Vec<&CpuInfo> = Vec::with_capacity(cpus_len / 2);

        for (i, l3) in l3s.into_iter().enumerate() {
            let cpus_on_l3: Vec<&&CpuInfo> = cpus.iter().filter(|c| c.l3 == l3).collect();
            if i % 2 == 0 {
                upper_half.extend(cpus_on_l3.into_iter());
            } else {
                lower_half.extend(cpus_on_l3.into_iter());
            }
        }

        let mut node: NodeInfo = lower_half[0].node;
        node.memory = size;

        Deployment::new(desc, lower_half, upper_half, vec![node])
    }
}

impl<'a> fmt::Display for Deployment<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let a: Vec<Cpu> = self.a.iter().map(|c| c.cpu).collect();
        let b: Vec<Cpu> = self.b.iter().map(|c| c.cpu).collect();

        write!(f, "Deployment Plan for {}:\n", self.description)?;
        write!(f, "-- Program A cores: {:?}\n", a)?;
        write!(f, "-- Program B cores: {:?}\n", b)?;
        write!(f, "-- Use memory:\n")?;
        for n in self.mem.iter() {
            write!(f, " - On node {}: {} Bytes\n", n.node, n.memory)?;
        }
        Ok(())
    }
}

#[derive(Debug, RustcEncodable)]
struct Program<'a> {
    name: String,
    manifest_path: &'a Path,
    binary: String,
    working_dir: String,
    args: Vec<String>,
    env: Vec<(String, String)>,
    antagonist_args: Vec<String>,
    breakpoints: Vec<String>,
    checkpoints: Vec<String>,
    is_openmp: bool,
    is_parsec: bool,
    use_watch_repeat: bool,
    alone: bool,
}

impl<'a> Program<'a> {
    fn from_toml(
        manifest_path: &'a Path,
        config: &toml::Table,
        alone_default: bool,
    ) -> Program<'a> {
        let name: String = config["name"]
            .as_str()
            .expect("program.name not a string")
            .to_string();
        let binary: String = config["binary"]
            .as_str()
            .expect("program.binary not a string")
            .to_string();

        let default_working_dir = String::from(manifest_path.to_str().unwrap());
        let working_dir: String = config
            .get("working_dir")
            .map_or(default_working_dir.clone(), |v| {
                v.as_str()
                    .expect("program.working_dir not a string")
                    .to_string()
            })
            .replace("$MANIFEST_DIR", default_working_dir.as_str());

        let openmp: bool = config.get("openmp").map_or(false, |v| {
            v.as_bool().expect("'program.openmp' should be boolean")
        });
        let parsec: bool = config.get("parsec").map_or(false, |v| {
            v.as_bool().expect("'program.parsec' should be boolean")
        });
        let watch_repeat: bool = config.get("use_watch_repeat").map_or(false, |v| {
            v.as_bool()
                .expect("'program.use_watch_repeat' should be boolean")
        });
        let alone: bool = config.get("alone").map_or(alone_default, |v| {
            v.as_bool().expect("'program.alone' should be boolean")
        });
        let args: Vec<String> = config["arguments"]
            .as_slice()
            .expect("program.arguments not an array?")
            .iter()
            .map(|s| {
                s.as_str()
                    .expect("program1 argument not a string?")
                    .to_string()
            })
            .collect();
        let antagonist_args: Vec<String> =
            config
                .get("antagonist_arguments")
                .map_or(args.clone(), |v| {
                    v.as_slice()
                        .expect("program.antagonist_arguments not an array?")
                        .iter()
                        .map(|s| {
                            s.as_str()
                                .expect("program2 argument not a string?")
                                .to_string()
                        })
                        .collect()
                });
        let env: Vec<(String, String)> = config.get("env").map_or(Vec::new(), |t| {
            t.as_table()
                .expect("program.env not a table?")
                .iter()
                .map(|(k, v)| {
                    (
                        k.as_str().to_string(),
                        v.as_str()
                            .expect("env value needs to be a string")
                            .to_string(),
                    )
                })
                .collect()
        });

        let breakpoints: Vec<String> = config.get("breakpoints").map_or(Vec::new(), |bs| {
            bs.as_slice()
                .expect("program.breakpoints not an array?")
                .iter()
                .map(|s| {
                    s.as_str()
                        .expect("program breakpoint not a string?")
                        .to_string()
                })
                .collect()
        });
        // TODO: this is currently not in use (remove?)
        let checkpoints: Vec<String> = config.get("checkpoints").map_or(Vec::new(), |cs| {
            cs.as_slice()
                .expect("program.checkpoints not an array?")
                .iter()
                .map(|s| {
                    s.as_str()
                        .expect("program checkpoint not a string?")
                        .to_string()
                })
                .collect()
        });

        Program {
            name: name,
            manifest_path: manifest_path,
            binary: binary,
            is_openmp: openmp,
            is_parsec: parsec,
            env: env,
            alone: alone,
            working_dir: working_dir,
            use_watch_repeat: watch_repeat,
            args: args,
            antagonist_args: antagonist_args,
            breakpoints: breakpoints,
            // TODO: this is currently not in use (remove?)
            checkpoints: checkpoints,
        }
    }

    fn get_cmd(&self, antagonist: bool, cores: &Vec<&CpuInfo>) -> Vec<String> {
        let nthreads = cores.len();
        let mut cmd = vec![&self.binary];

        if !antagonist {
            cmd.extend(self.args.iter());
        } else {
            cmd.extend(self.antagonist_args.iter());
        }

        cmd.iter()
            .map(|s| s.replace("$NUM_THREADS", format!("{}", nthreads).as_str()))
            .map(|s| {
                s.replace(
                    "$MANIFEST_DIR",
                    format!("{}", self.manifest_path.to_str().unwrap()).as_str(),
                )
            })
            .collect()
    }

    fn get_env(&self, antagonist: bool, cores: &Vec<&CpuInfo>) -> Vec<(String, String)> {
        let mut env: Vec<(String, String)> = Vec::with_capacity(2);
        let cpus: Vec<String> = cores.iter().map(|c| format!("{}", c.cpu)).collect();
        // TODO: remove this feature:
        if self.is_openmp {
            env.push((String::from("OMP_PROC_BIND"), String::from("true")));
            env.push((
                String::from("OMP_PLACES"),
                format!("{{{}}}", cpus.join(",")),
            ));
        }
        // TODO: remove this feature:
        else if self.is_parsec {
            assert!(!self.is_openmp);
            env.push((
                String::from("LD_PRELOAD"),
                format!(
                    "{}/bin/libhooks.so.0.0.0",
                    self.manifest_path.to_str().unwrap()
                ),
            ));
            env.push((String::from("PARSEC_CPU_NUM"), format!("{}", cpus.len())));
            env.push((
                String::from("PARSEC_CPU_BASE"),
                format!("{}", cpus.join(",")),
            ));
            if antagonist {
                env.push((String::from("PARSEC_REPEAT"), String::from("1")));
            }
        }

        // keep this one:
        for (k, v) in self.env.clone() {
            env.push((k, v));
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
    fn new(
        manifest_path: &'a Path,
        output_path: &'a Path,
        a: &'a Program<'a>,
        b: Option<&'a Program<'a>>,
        deployment: &'a Deployment,
    ) -> Run<'a> {
        let mut out_dir = output_path.to_path_buf();
        out_dir.push(deployment.description);
        mkdir(&out_dir);
        match b {
            Some(p) => out_dir.push(format!("{}_vs_{}", a.name, p.name)),
            None => out_dir.push(a.name.as_str()),
        }

        Run {
            manifest_path: manifest_path,
            output_path: out_dir,
            a: a,
            b: b,
            deployment: deployment,
        }
    }

    fn profile_a(&self) -> io::Result<()> {
        let cmd = self.a.get_cmd(false, &self.deployment.a);
        let env = self.a.get_env(false, &self.deployment.a);
        let mut bps: Vec<String> = self.a.breakpoints.iter().map(|s| s.to_string()).collect();
        // TODO: this is currently not in use (remove?)
        bps.extend(self.a.checkpoints.iter().map(|s| s.to_string()));
        // let cps = self.a.checkpoints.iter().map(|s| s.to_string()).collect();

        debug!(
            "Spawning {:?} with environment {:?} breakpoints {:?}",
            cmd, env, bps
        );
        profile::profile(
            &self.output_path,
            self.a.working_dir.as_str(),
            cmd,
            env,
            bps,
            false,
            None,
            false,
        );
        Ok(())
    }

    fn start_b(&mut self) -> Option<Child> {
        self.b.map(|b| {
            let mut command_args = b.get_cmd(true, &self.deployment.b);
            let env = b.get_env(true, &self.deployment.b);
            if b.use_watch_repeat {
                command_args.insert(0, String::from("-t"));
                command_args.insert(0, String::from("-n0"));
                command_args.insert(0, String::from("watch"));
            }

            debug!("Spawning {:?} with environment {:?}", command_args, env);
            debug!("Working dir for B is: {}", b.working_dir.as_str());

            let mut cmd = Command::new(&command_args[0]);
            let cmd = cmd
                .stdout(Stdio::piped())
                .current_dir(b.working_dir.as_str())
                .stderr(Stdio::piped())
                .args(&command_args[1..]);

            // Add the environment:
            for (key, value) in env {
                cmd.env(key, value);
            }

            match cmd.spawn() {
                Ok(child) => child,
                Err(_) => panic!("Can't spawn program B"),
            }
        })
    }

    fn save_output<T: io::Read>(&self, filename: &str, what: &mut T) -> io::Result<()> {
        let mut stdout = String::new();
        what.read_to_string(&mut stdout)?;
        let mut stdout_path = self.output_path.clone();
        stdout_path.push(filename);
        let mut f = File::create(stdout_path.as_path())?;
        f.write_all(stdout.as_bytes())
    }

    fn save_run_information(&self) -> io::Result<()> {
        let mut run_toml_path = self.output_path.clone();
        run_toml_path.push("run.toml");
        let mut f = File::create(run_toml_path.as_path())?;
        let mut e = toml::Encoder::new();
        self.encode(&mut e).unwrap();
        f.write_all(toml::encode_str(&e.toml).as_bytes())?;

        let mut run_txt_path = self.output_path.clone();
        run_txt_path.push("run.txt");
        let mut f = File::create(run_txt_path.as_path())?;
        f.write_all(format!("{}", self).as_bytes())
    }

    fn is_completed(&self) -> bool {
        // Is this run already done (in case we restart):
        let mut completed_file: PathBuf = self.output_path.to_path_buf();
        completed_file.push("completed");
        if completed_file.exists() {
            true
        } else {
            false
        }
    }

    fn profile(&mut self) -> io::Result<()> {
        mkdir(&self.output_path);
        if self.is_completed() {
            warn!(
                "Run {} already completed, skipping.",
                self.output_path.to_string_lossy()
            );
            return Ok(());
        }

        self.save_run_information()?;

        // Profile together with B
        let maybe_app_b: Option<Child> = self.start_b();
        if maybe_app_b.is_some() {
            debug!("Wait for B to warmup before starting to profile A");
            let one_min = Duration::from_millis(60000);
            thread::sleep(one_min);
        }

        self.profile_a()?;

        match maybe_app_b {
            Some(mut app_b) => {
                match app_b.wait_timeout(Duration::from_millis(200)).unwrap() {
                    Some(status) => {
                        // The Application B has already exited, this means it probably crashed
                        // while we were profiling (bad). We can't use these results.
                        app_b
                            .stdout
                            .map(|mut c| self.save_output("B_stdout.txt", &mut c));
                        app_b
                            .stderr
                            .map(|mut c| self.save_output("B_stderr.txt", &mut c));

                        let mut completed_path = self.output_path.clone();
                        completed_path.push("completed");
                        fs::remove_file(completed_path)?;

                        panic!(
                            "B has crashed during measurements {:?}. This is bad.",
                            status.code()
                        );
                        // TODO: save error code and continue (?)
                    }
                    None => {
                        app_b.kill()?;
                        app_b.wait()?;
                        app_b
                            .stdout
                            .map(|mut c| self.save_output("B_stdout.txt", &mut c));
                        app_b
                            .stderr
                            .map(|mut c| self.save_output("B_stderr.txt", &mut c));
                    }
                }
            }
            None => {}
        };

        Ok(())
    }
}

impl<'a> fmt::Display for Run<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "A: ENV = {:?} CMD = {:?}\n",
            self.a.get_env(false, &self.deployment.a),
            self.a.get_cmd(false, &self.deployment.a)
        )?;
        write!(f, "A Breakpoints: {:?}\n", self.a.breakpoints)?;
        write!(f, "A Checkpoints: {:?}\n", self.a.checkpoints)?;
        match self.b {
            Some(b) => {
                write!(
                    f,
                    "B: {:?} {:?}\n",
                    b.get_env(true, &self.deployment.b),
                    b.get_cmd(true, &self.deployment.b)
                )?;
                write!(f, "{}", &self.deployment)?;
            }
            None => {
                write!(f, "No other program running.")?;
            }
        }
        Ok(())
    }
}

pub fn pair(manifest_folder: &Path, dryrun: bool, start: usize, stepping: usize) {
    let canonical_manifest_path =
        fs::canonicalize(&manifest_folder).expect("canonicalize manifest path does not work");

    let mut out_dir = canonical_manifest_path.to_path_buf();
    let hostname = get_hostname().unwrap_or(String::from("unknown"));
    out_dir.push(hostname);
    mkdir(&out_dir);

    let mt = MachineTopology::new();

    let mut manifest: PathBuf = canonical_manifest_path.to_path_buf();
    manifest.push("manifest.toml");
    let mut file = File::open(manifest.as_path()).expect("manifest.toml file does not exist?");
    let mut manifest_string = String::new();
    let _ = file.read_to_string(&mut manifest_string).unwrap();
    let mut parser = toml::Parser::new(manifest_string.as_str());
    let doc = match parser.parse() {
        Some(doc) => doc,
        None => {
            error!("Can't parse the manifest file:\n{:?}", parser.errors);
            process::exit(1);
        }
    };
    let experiment: &toml::Table = doc["experiment"]
        .as_table()
        .expect("Error in manifest.toml: 'experiment' should be a table.");
    let configuration: &[toml::Value] = experiment["configurations"]
        .as_slice()
        .expect("Error in manifest.toml: 'configuration' attribute should be a list.");
    let configs: Vec<String> = configuration
        .iter()
        .map(|s| {
            s.as_str()
                .expect("configuration elements should be strings")
                .to_string()
        })
        .collect();
    let run_alone: bool = experiment
        .get("alone")
        .map_or(true, |v| v.as_bool().expect("'alone' should be boolean"));
    let profile_only: Option<Vec<String>> = experiment.get("profile_only_a").map(|progs| {
        progs
            .as_slice()
            .expect("Error in manifest.toml: 'profile_only_a' should be a list.")
            .into_iter()
            .map(|p| {
                p.as_str()
                    .expect("profile_only_a elements should name programs (strings)")
                    .to_string()
            })
            .collect()
    });
    let profile_only_b: Option<Vec<String>> = experiment.get("profile_only_b").map(|progs| {
        progs
            .as_slice()
            .expect("Error in manifest.toml: 'profile_only_b' should be a list.")
            .into_iter()
            .map(|p| {
                p.as_str()
                    .expect("profile_only_b elements should name programs (strings)")
                    .to_string()
            })
            .collect()
    });

    let mut programs: Vec<Program> = Vec::with_capacity(2);
    for (key, _value) in &doc {
        if key.starts_with("program") {
            let program_desc: &toml::Table = doc[key]
                .as_table()
                .expect("Error in manifest.toml: 'program' should be a table.");
            programs.push(Program::from_toml(
                &canonical_manifest_path,
                program_desc,
                run_alone,
            ));
        }
    }

    let mut deployments: Vec<Deployment> = Vec::with_capacity(4);
    for config in configs {
        match config.as_str() {
            "L1-SMT" => deployments.push(Deployment::split_interleaved(
                "L1-SMT",
                mt.same_l1(),
                mt.l1_size().unwrap_or(0),
            )),
            "L3-SMT" => deployments.push(Deployment::split_interleaved(
                "L3-SMT",
                mt.same_l3(),
                mt.l3_size().unwrap_or(0),
            )),
            "L3-SMT-cores" => deployments.push(Deployment::split_smt_aware(
                "L3-SMT-cores",
                mt.same_l3(),
                mt.l3_size().unwrap_or(0),
            )),
            "L3-cores" => deployments.push(Deployment::split_smt_aware(
                "L3-cores",
                mt.same_l3_cores(),
                mt.l3_size().unwrap_or(0),
            )),
            "Full-L3" => deployments.push(Deployment::split_l3_aware(
                "Full-L3",
                mt.whole_machine_cores(),
                mt.l3_size().unwrap_or(0),
            )),
            "Full-SMT-L3" => deployments.push(Deployment::split_l3_aware(
                "Full-SMT-L3",
                mt.whole_machine(),
                mt.l3_size().unwrap_or(0),
            )),
            "Full-cores" => deployments.push(Deployment::split_interleaved(
                "Full-cores",
                mt.whole_machine_cores(),
                mt.l3_size().unwrap_or(0),
            )),
            "Full-SMT-cores" => deployments.push(Deployment::split_smt_aware(
                "Full-SMT-cores",
                mt.whole_machine(),
                mt.l3_size().unwrap_or(0),
            )),

            _ => error!("Ignored unknown deployment config '{}'.", config),
        };
    }

    // Add all possible pairs:
    let mut pairs: Vec<(&Program, Option<&Program>)> = Vec::new();
    for p in programs.iter() {
        pairs.push((p, None));
    }
    for (a, b) in iproduct!(programs.iter(), programs.iter()) {
        pairs.push((a, Some(b)));
    }

    // Filter out the pairs we do not want to execute:
    let mut runs: Vec<Run> = Vec::new();
    for (a, b) in pairs.into_iter() {
        let profile_a = profile_only
            .as_ref()
            .map_or(true, |ps| ps.contains(&a.name));
        let profile_b = !b.is_none()
            && profile_only_b
                .as_ref()
                .map_or(profile_a, |ps| ps.contains(&b.unwrap().name));
        if !profile_a && !profile_b {
            continue;
        }

        for d in deployments.iter() {
            if b.is_none() && (!run_alone || !a.alone) {
                continue;
            }
            runs.push(Run::new(
                &canonical_manifest_path,
                out_dir.as_path(),
                a,
                b,
                d,
            ));
        }
    }

    // Finally, profile the runs we are supposed to execute based on the command line args
    let mut i = 0;
    for run in runs.iter_mut().skip(start).step(stepping) {
        if !dryrun {
            run.profile().ok();
        } else {
            warn!("Dryrun mode: {}", run);
        }
        i += 1;
    }

    info!("{} runs completed.", i);
}
