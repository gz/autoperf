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
use itertools::Itertools;

use csv;

use toml;

use super::util::*;

pub fn scale(manifest_folder: &Path, dryrun: bool, start: usize, stepping: usize) {
    let canonical_manifest_path = fs::canonicalize(&manifest_folder)
        .expect("canonicalize manifest path does not work");

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
    let configs: Vec<String> = configuration.iter()
        .map(|s| s.as_str().expect("configuration elements should be strings").to_string())
        .collect();
    let run_alone: bool = experiment.get("alone")
        .map_or(true, |v| v.as_bool().expect("'alone' should be boolean"));
    let profile_only: Option<Vec<String>> = experiment.get("profile_only_a")
        .map(|progs| {
            progs.as_slice()
                .expect("Error in manifest.toml: 'profile_only_a' should be a list.")
                .into_iter()
                .map(|p| {
                    p.as_str()
                        .expect("profile_only_a elements should name programs (strings)")
                        .to_string()
                })
                .collect()
        });
    let profile_only_b: Option<Vec<String>> = experiment.get("profile_only_b")
        .map(|progs| {
            progs.as_slice()
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
    for (key, value) in &doc {
        if key.starts_with("program") {
            let program_desc: &toml::Table =
                doc[key].as_table().expect("Error in manifest.toml: 'program' should be a table.");
            programs.push(Program::from_toml(&canonical_manifest_path, program_desc, run_alone));
        }
    }

    let mut deployments: Vec<Deployment> = Vec::with_capacity(4);
    for config in configs {
        match config.as_str() {
            "L1-SMT" => {
                deployments.push(Deployment::split_interleaved("L1-SMT",
                                                               mt.same_l1(),
                                                               mt.l1_size().unwrap_or(0)))
            }
            "L3-SMT" => {
                deployments.push(Deployment::split_interleaved("L3-SMT",
                                                               mt.same_l3(),
                                                               mt.l3_size().unwrap_or(0)))
            }
            "L3-SMT-cores" => {
                deployments.push(Deployment::split_smt_aware("L3-SMT-cores",
                                                             mt.same_l3(),
                                                             mt.l3_size().unwrap_or(0)))
            }
            "L3-cores" => {
                deployments.push(Deployment::split_smt_aware("L3-cores",
                                                             mt.same_l3_cores(),
                                                             mt.l3_size().unwrap_or(0)))
            }
            "Full-L3" => {
                deployments.push(Deployment::split_l3_aware("Full-L3",
                                                            mt.whole_machine_cores(),
                                                            mt.l3_size().unwrap_or(0)))
            }
            "Full-SMT-L3" => {
                deployments.push(Deployment::split_l3_aware("Full-SMT-L3",
                                                            mt.whole_machine(),
                                                            mt.l3_size().unwrap_or(0)))
            }
            "Full-cores" => {
                deployments.push(Deployment::split_interleaved("Full-cores",
                                                               mt.whole_machine_cores(),
                                                               mt.l3_size().unwrap_or(0)))
            }
            "Full-SMT-cores" => {
                deployments.push(Deployment::split_smt_aware("Full-SMT-cores",
                                                             mt.whole_machine(),
                                                             mt.l3_size().unwrap_or(0)))
            }

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
        let profile_a = profile_only.as_ref().map_or(true, |ps| ps.contains(&a.name));
        let profile_b = !b.is_none() &&
                        profile_only_b.as_ref()
            .map_or(profile_a, |ps| ps.contains(&b.unwrap().name));
        if !profile_a && !profile_b {
            continue;
        }

        for d in deployments.iter() {
            if b.is_none() && (!run_alone || !a.alone) {
                continue;
            }
            runs.push(Run::new(&canonical_manifest_path, out_dir.as_path(), a, b, d));
        }
    }

    // Finally, profile the runs we are supposed to execute based on the command line args
    let mut i = 0;
    for run in runs.iter_mut().skip(start).step(stepping) {
        if !dryrun {
            run.profile();
        } else {
            println!("{}", run);
        }
        i += 1;
    }

    println!("{} runs completed.", i);
}
