use std;
use std::collections::HashMap;

use csv;
use lazy_static::lazy_static;
use pbr::ProgressBar;
use std::error;
use std::error::Error;
use std::fmt;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;
use x86::cpuid;
use x86::perfcnt::intel::{events, Counter, EventDescription, MSRIndex, PebsType, Tuple};

use super::util::*;
use log::*;

lazy_static! {

    /// Check if HT is enabled on this CPU (if HT is disabled it doubles the amount of available
    /// performance counters on a core).
    static ref HT_AVAILABLE: bool = {
        let cpuid = cpuid::CpuId::new();
        cpuid.get_extended_topology_info().unwrap().any(|t| {
            t.level_type() == cpuid::TopologyType::SMT
        })
    };

    /// For every MonitoringUnit try to figure out how many counters we support.
    /// This is handled through a config file since Linux doesn't export this information in
    /// it's PMU devices (but probably should)...
    static ref PMU_COUNTERS: HashMap<MonitoringUnit, usize> = {
        let cpuid = cpuid::CpuId::new();
        let cpu_counter = cpuid.get_performance_monitoring_info().map_or(0, |info| info.number_of_counters()) as usize;
        let mut res = HashMap::with_capacity(11);
        res.insert(MonitoringUnit::CPU, cpu_counter);
        let (family, model) = cpuid.get_feature_info().map_or((0,0), |fi| (fi.family_id(), ((fi.extended_model_id() as u8) << 4) | fi.model_id() as u8));

        let ctr_config = include_str!("counters.toml");
        let mut parser = toml::Parser::new(ctr_config);

        let doc = match parser.parse() {
            Some(doc) => doc,
            None => {
                error!("Can't parse the counter configuration file:\n{:?}", parser.errors);
                std::process::exit(9);
            }
        };

        trace!("Trying to find architecture for family = {:#x} model = {:#x}", family, model);
        let mut found: bool = false;
        for (name, architecture) in doc {
            let architecture = architecture.as_table().expect("counters.toml architectures must be a table");
            let cfamily = &architecture["family"];
            for cmodel in architecture["models"].as_slice().expect("counters.toml models must be a list.") {
                let cfamily = cfamily.as_integer().expect("Family must be int.") as u8;
                let cmodel = cmodel.as_integer().expect("Model must be int.") as u8;
                if family == cfamily && model == cmodel {
                    trace!("Running on {}, reading MonitoringUnit limits from config", name);
                    found = true;

                    // TODO: We should ideally get both, prgrammable and fixed counters:
                    for (unit, limit) in architecture["programmable_counters"].as_table().expect("programmable_counters must be a table") {
                        let unit = MonitoringUnit::new(unit.as_str());
                        let limit = limit.as_integer().expect("Counter limit should be an integer");
                        res.insert(unit, limit as usize);
                    }
                }
            }
        }

        if !found {
            warn!("Didn't recogize this architecture so we can't infer #counters for MonitoringUnit (Please update counters.toml for family = {:#x} model = {:#x})", family, model);
            res.insert(MonitoringUnit::UBox, 4);
            res.insert(MonitoringUnit::HA, 4);
            res.insert(MonitoringUnit::IRP, 4);
            res.insert(MonitoringUnit::PCU, 4);
            res.insert(MonitoringUnit::R2PCIe, 4);
            res.insert(MonitoringUnit::R3QPI, 4);
            res.insert(MonitoringUnit::QPI, 4);
            res.insert(MonitoringUnit::CBox, 2);
            res.insert(MonitoringUnit::IMC, 4);
            res.insert(MonitoringUnit::Arb, 2);
            res.insert(MonitoringUnit::M2M, 4);
            res.insert(MonitoringUnit::CHA, 4);
            res.insert(MonitoringUnit::M3UPI, 4);
            res.insert(MonitoringUnit::IIO, 4);
            res.insert(MonitoringUnit::UPI_LL, 4);
        }

        res
    };

    /// Find the linux PMU devices that we need to program through perf
    static ref PMU_DEVICES: Vec<String> = {
        let paths = fs::read_dir("/sys/bus/event_source/devices/").expect("Can't read devices directory.");
        let mut devices = Vec::with_capacity(15);
        for p in paths {
            let path = p.expect("Is not a path.");
            let file_name = path.file_name().into_string().expect("Is valid UTF-8 string.");
            devices.push(file_name);
        }

        devices
    };

    /// Bogus or clocks that we don't want to measure or tend to break things
    static ref IGNORE_EVENTS: HashMap<&'static str, bool> = {
        let mut ignored = HashMap::with_capacity(1);
        ignored.insert("UNC_CLOCK.SOCKET", true); // Just says 'fixed' and does not name which counter :/
        ignored.insert("UNC_M_CLOCKTICKS_F", true);
        ignored.insert("UNC_U_CLOCKTICKS", true);
        ignored
    };

    /// Which events should be measured in isolation on this architecture.
    static ref ISOLATE_EVENTS: Vec<&'static str> = {
        let cpuid = cpuid::CpuId::new();
        let (family, model) = cpuid.get_feature_info().map_or((0,0), |fi| (fi.family_id(), ((fi.extended_model_id() as u8) << 4) | fi.model_id() as u8));

        // Sometimes the perfmon data is missing the errata information
        // as is the case for IvyBridge where MEM_LOAD* things can't be measured
        // together with other things.
        if family == 0x6 && (model == 62 || model == 58) {
            vec![   "MEM_UOPS_RETIRED.ALL_STORES",
                    "MEM_LOAD_UOPS_RETIRED.L1_MISS",
                    "MEM_LOAD_UOPS_RETIRED.HIT_LFB",
                    "MEM_LOAD_UOPS_LLC_HIT_RETIRED.XSNP_HITM",
                    "MEM_LOAD_UOPS_RETIRED.L2_HIT",
                    "MEM_UOPS_RETIRED.SPLIT_LOADS",
                    "MEM_UOPS_RETIRED.ALL_LOADS",
                    "MEM_LOAD_UOPS_LLC_MISS_RETIRED.LOCAL_DRAM",
                    "MEM_LOAD_UOPS_LLC_HIT_RETIRED.XSNP_NONE",
                    "MEM_LOAD_UOPS_RETIRED.L1_HIT",
                    "MEM_UOPS_RETIRED.STLB_MISS_STORES",
                    "MEM_LOAD_UOPS_LLC_HIT_RETIRED.XSNP_HIT",
                    "MEM_LOAD_UOPS_RETIRED.LLC_MISS",
                    "MEM_LOAD_UOPS_RETIRED.L2_MISS",
                    "MEM_LOAD_UOPS_LLC_HIT_RETIRED.XSNP_MISS",
                    "MEM_UOPS_RETIRED.STLB_MISS_LOADS",
                    "MEM_UOPS_RETIRED.LOCK_LOADS",
                    "MEM_LOAD_UOPS_RETIRED.LLC_HIT",
                    "MEM_UOPS_RETIRED.SPLIT_STORES",
                    // Those are IvyBridge-EP events:
                    "MEM_LOAD_UOPS_LLC_MISS_RETIRED.REMOTE_DRAM",
                    "MEM_LOAD_UOPS_LLC_MISS_RETIRED.REMOTE_HITM",
                    "MEM_LOAD_UOPS_LLC_MISS_RETIRED.REMOTE_FWD"]
        }
        else {
            vec![]
        }
    };
}

fn execute_perf(
    perf: &mut Command,
    cmd: &Vec<String>,
    counters: &Vec<String>,
    datafile: &Path,
    dryrun: bool,
) -> (String, String, String) {
    assert!(cmd.len() >= 1);
    let perf = perf.arg("-o").arg(datafile.as_os_str());
    let events: Vec<String> = counters.iter().map(|c| format!("-e {}", c)).collect();

    let perf = perf.args(events.as_slice());
    let perf = perf.args(cmd.as_slice());
    let perf_cmd_str: String = format!("{:?}", perf).replace("\"", "");

    let (stdout, stderr) = if !dryrun {
        match perf.output() {
            Ok(out) => {
                let stdout =
                    String::from_utf8(out.stdout).unwrap_or(String::from("Unable to read stdout!"));
                let stderr =
                    String::from_utf8(out.stderr).unwrap_or(String::from("Unable to read stderr!"));

                if out.status.success() {
                    trace!("stdout:\n{:?}", stdout);
                    trace!("stderr:\n{:?}", stderr);
                } else if !out.status.success() {
                    error!(
                        "perf command: {} got unknown exit status was: {}",
                        perf_cmd_str, out.status
                    );
                    debug!("stdout:\n{}", stdout);
                    debug!("stderr:\n{}", stderr);
                }

                if !datafile.exists() {
                    error!(
                        "perf command: {} succeeded but did not produce the required file {:?} \
                         (you should file a bug report!)",
                        perf_cmd_str, datafile
                    );
                }

                (stdout, stderr)
            }
            Err(err) => {
                error!("Executing {} failed : {}", perf_cmd_str, err);
                (String::new(), String::new())
            }
        }
    } else {
        warn!("Dry run mode -- would execute: {}", perf_cmd_str);
        (String::new(), String::new())
    };

    (perf_cmd_str, stdout, stderr)
}

pub fn create_out_directory(out_dir: &Path) {
    if !out_dir.exists() {
        std::fs::create_dir(out_dir).expect("Can't create `out` directory");
    }
}

pub fn get_known_events<'a>() -> Vec<&'a EventDescription<'static>> {
    events()
        .expect("No performance events found?")
        .values()
        .collect()
}

#[allow(non_camel_case_types)]
#[derive(Hash, Eq, PartialEq, Debug, Copy, Clone, PartialOrd, Ord)]
pub enum MonitoringUnit {
    /// Devices
    CPU,
    /// Memory stuff
    Arb,
    /// The CBox manages the interface between the core and the LLC, so
    /// the instances of uncore CBox is equal to number of cores
    CBox,
    /// ???
    SBox,
    /// ???
    UBox,
    /// QPI Stuff
    QPI,
    /// Ring to QPI
    R3QPI,
    /// IIO Coherency
    IRP,
    /// Ring to PCIe
    R2PCIe,
    /// Memory Controller
    IMC,
    /// Home Agent
    HA,
    /// Power Control Unit
    PCU,
    /// XXX
    M2M,
    /// XXX  
    CHA,
    /// XXX
    M3UPI,
    /// XXX
    IIO,
    /// XXX
    UPI_LL,
    /// Types we don't know how to handle...
    Unknown,
}

impl fmt::Display for MonitoringUnit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            MonitoringUnit::CPU => write!(f, "CPU"),
            MonitoringUnit::Arb => write!(f, "Arb"),
            MonitoringUnit::CBox => write!(f, "CBox"),
            MonitoringUnit::SBox => write!(f, "SBox"),
            MonitoringUnit::UBox => write!(f, "UBox"),
            MonitoringUnit::QPI => write!(f, "QPI"),
            MonitoringUnit::R3QPI => write!(f, "R3QPI"),
            MonitoringUnit::IRP => write!(f, "IRP"),
            MonitoringUnit::R2PCIe => write!(f, "R2PCIe"),
            MonitoringUnit::IMC => write!(f, "IMC"),
            MonitoringUnit::HA => write!(f, "HA"),
            MonitoringUnit::PCU => write!(f, "PCU"),
            MonitoringUnit::M2M => write!(f, "M2M"),
            MonitoringUnit::CHA => write!(f, "CHA"),
            MonitoringUnit::M3UPI => write!(f, "M3UPI"),
            MonitoringUnit::IIO => write!(f, "IIO"),
            MonitoringUnit::UPI_LL => write!(f, "UPI LL"),
            MonitoringUnit::Unknown => write!(f, "Unknown"),
        }
    }
}

impl MonitoringUnit {
    fn new<'a>(unit: &'a str) -> MonitoringUnit {
        match unit.to_lowercase().as_str() {
            "cpu" => MonitoringUnit::CPU,
            "cbo" => MonitoringUnit::CBox,
            "qpi_ll" => MonitoringUnit::QPI,
            "sbo" => MonitoringUnit::SBox,
            "imph-u" => MonitoringUnit::Arb,
            "arb" => MonitoringUnit::Arb,
            "r3qpi" => MonitoringUnit::R3QPI,
            "qpi ll" => MonitoringUnit::QPI,
            "irp" => MonitoringUnit::IRP,
            "r2pcie" => MonitoringUnit::R2PCIe,
            "imc" => MonitoringUnit::IMC,
            "ha" => MonitoringUnit::HA,
            "pcu" => MonitoringUnit::PCU,
            "ubox" => MonitoringUnit::UBox,
            "m2m" => MonitoringUnit::M2M,
            "cha" => MonitoringUnit::CHA,
            "m3upi" => MonitoringUnit::M3UPI,
            "iio" => MonitoringUnit::IIO,
            "upi ll" => MonitoringUnit::UPI_LL,
            "upi" => MonitoringUnit::UPI_LL,
            "ubo" => MonitoringUnit::UBox,
            "qpi" => MonitoringUnit::QPI,
            _ => {
                error!("Don't support MonitoringUnit {}", unit);
                MonitoringUnit::Unknown
            }
        }
    }

    pub fn to_intel_event_description(&self) -> Option<&'static str> {
        match *self {
            MonitoringUnit::CPU => None,
            MonitoringUnit::CBox => Some("CBO"),
            MonitoringUnit::QPI => Some("QPI_LL"),
            MonitoringUnit::SBox => Some("SBO"),
            MonitoringUnit::Arb => Some("ARB"),
            MonitoringUnit::R3QPI => Some("R3QPI"),
            MonitoringUnit::IRP => Some("IRP"),
            MonitoringUnit::R2PCIe => Some("R2PCIE"),
            MonitoringUnit::IMC => Some("IMC"),
            MonitoringUnit::HA => Some("HA"),
            MonitoringUnit::PCU => Some("PCU"),
            MonitoringUnit::UBox => Some("UBOX"),
            MonitoringUnit::M2M => Some("M2M"),
            MonitoringUnit::CHA => Some("CHA"),
            MonitoringUnit::M3UPI => Some("M3UPI"),
            MonitoringUnit::IIO => Some("IIO"),
            MonitoringUnit::UPI_LL => Some("UPI LL"),
            MonitoringUnit::Unknown => None,
        }
    }

    /// Return the perf prefix for selecting the right PMU unit in case of uncore counters.
    pub fn to_perf_prefix(&self) -> Option<&'static str> {
        let res = match *self {
            MonitoringUnit::CPU => Some("cpu"),
            MonitoringUnit::CBox => Some("uncore_cbox"),
            MonitoringUnit::QPI => Some("uncore_qpi"),
            MonitoringUnit::SBox => Some("uncore_sbox"),
            MonitoringUnit::Arb => Some("uncore_arb"),
            MonitoringUnit::R3QPI => Some("uncore_r3qpi"), // Adds postfix value
            MonitoringUnit::IRP => Some("uncore_irp"), // According to libpfm4 (lib/pfmlib_intel_ivbep_unc_irp.c)
            MonitoringUnit::R2PCIe => Some("uncore_r2pcie"),
            MonitoringUnit::IMC => Some("uncore_imc"), // Adds postfix value
            MonitoringUnit::HA => Some("uncore_ha"),   // Adds postfix value
            MonitoringUnit::PCU => Some("uncore_pcu"),
            MonitoringUnit::UBox => Some("uncore_ubox"),
            MonitoringUnit::M2M => Some("uncore_m2m"), // Adds postfix value
            MonitoringUnit::CHA => Some("uncore_cha"), // Adds postfix value
            MonitoringUnit::M3UPI => Some("uncore_m3upi"), // Adds postfix value
            MonitoringUnit::IIO => Some("uncore_iio"), // Adds postfix value
            MonitoringUnit::UPI_LL => Some("uncore_upi"), // Adds postfix value
            MonitoringUnit::Unknown => None,
        };

        // Note: If anything here does not return uncore_ as a prefix, you need to update extract.rs!
        res.map(|string| assert!(string.starts_with("uncore_") || string.starts_with("cpu")));

        res
    }
}

#[derive(Debug)]
pub struct PerfEvent<'a, 'b>(pub &'a EventDescription<'b>)
where
    'b: 'a;

impl<'a, 'b> PerfEvent<'a, 'b> {
    /// Returns all possible configurations of the event.
    /// This is a two vector tuple containing devices and configs:
    ///
    ///   * Devices are a subset of the ones listed in `/sys/bus/event_source/devices/`
    ///     Usually just `cpu` but uncore events can be measured on multiple devices.
    ///   * Configs are all possible combinations of attributes for this event.
    ///     Usually one but offcore events have two.
    ///
    /// # Note
    /// The assumption of the return type is that we can always match any
    /// device with any config. Let's see how long this assumption will remain valid...
    ///
    pub fn perf_configs(&self) -> (Vec<String>, Vec<Vec<String>>) {
        let mut devices = Vec::with_capacity(1);
        let mut configs = Vec::with_capacity(2);

        let typ = self.unit();

        // XXX: Horrible vector transformation:
        let matched_devices: Vec<String> = PMU_DEVICES
            .iter()
            .filter(|d| typ.to_perf_prefix().map_or(false, |t| d.starts_with(t)))
            .map(|d| d.clone())
            .collect();
        devices.extend(matched_devices);

        // We can have no devices if we don't understand how to match the unit name to perf names:
        if devices.len() == 0 {
            debug!(
                "Unit {:?} is not available to measure '{}'.",
                self.unit(),
                self,
            );
        }

        for args in self.perf_args() {
            configs.push(args);
        }

        (devices, configs)
    }

    /// Does this event use the passed code?
    pub fn uses_event_code(&self, event_code: u8) -> bool {
        match self.0.event_code {
            Tuple::One(e1) => e1 == event_code,
            Tuple::Two(e1, e2) => e1 == event_code || e2 == event_code,
        }
    }

    /// Does this event use the passed code?
    pub fn uses_umask(&self, umask: u8) -> bool {
        match self.0.umask {
            Tuple::One(m1) => m1 == umask,
            Tuple::Two(m1, m2) => m1 == umask || m2 == umask,
        }
    }

    /// Is this event an uncore event?
    pub fn is_uncore(&self) -> bool {
        self.0.unit.is_some()
    }

    pub fn unit(&self) -> MonitoringUnit {
        self.0
            .unit
            .map_or(MonitoringUnit::CPU, |u| MonitoringUnit::new(u))
    }

    /// Is this event an offcore event?
    pub fn is_offcore(&self) -> bool {
        match self.0.event_code {
            Tuple::One(_) => {
                assert!(!self.0.offcore);
                false
            }
            Tuple::Two(_, _) => {
                assert!(self.0.event_name.contains("OFFCORE"));
                // The OR is because there is this weird meta-event OFFCORE_RESPONSE
                // in the data files. It has offcore == false and is not really a proper event :/
                assert!(self.0.offcore || self.0.event_name == "OFFCORE_RESPONSE");
                true
            }
        }
    }

    /// Get the correct counter mask
    pub fn counter(&self) -> Counter {
        if *HT_AVAILABLE || self.is_uncore() {
            self.0.counter
        } else {
            self.0.counter_ht_off.expect("A bug in JSON?") // Ideally, all CPU events should have this attribute
        }
    }

    fn push_arg(configs: &mut Vec<Vec<String>>, value: String) {
        for config in configs.iter_mut() {
            config.push(value.clone());
        }
    }

    /// Returns a set of attributes used to build the perf event description.
    ///
    /// # Arguments
    ///   * try_alternative: Can give a different event encoding (for offcore events).
    fn perf_args(&self) -> Vec<Vec<String>> {
        // OFFCORE_RESPONSE_0 and OFFCORE_RESPONSE_1  provide identical functionality.  The reason
        // that there are two of them is that these events are associated with a separate MSR that is
        // used to program the types of requests/responses that you want to count (instead of being
        // able to include this information in the Umask field of the PERFEVT_SELx MSR).   The
        // performance counter event OFFCORE_RESPONSE_0 (Event 0xB7) is associated with MSR 0x1A6,
        // while the performance counter event OFFCORE_RESPONSE_1 (Event 0xBB) is associated with MSR
        // 0x1A7.
        // So having two events (with different associated MSRs) allows you to count two different
        // offcore response events at the same time.
        // Source: https://software.intel.com/en-us/forums/software-tuning-performance-optimization-platform-monitoring/topic/559227

        let two_configs: bool = match self.0.event_code {
            Tuple::One(_) => false,
            Tuple::Two(_, _) => true,
        };

        let mut ret: Vec<Vec<String>> = vec![Vec::with_capacity(7)];
        if two_configs {
            ret.push(Vec::with_capacity(7));
        }
        PerfEvent::push_arg(&mut ret, format!("name={}", self.0.event_name));

        let is_pcu = self.0.unit.map_or(false, |u| {
            return MonitoringUnit::new(u) == MonitoringUnit::PCU;
        });

        match self.0.event_code {
            Tuple::One(ev) => {
                // PCU events have umasks defined but they're OR'd with event (wtf)
                let pcu_umask = if is_pcu {
                    match self.0.umask {
                        Tuple::One(mask) => mask,
                        Tuple::Two(_m1, _m2) => unreachable!(),
                    }
                } else {
                    0x0
                };

                ret[0].push(format!("event=0x{:x}", ev | pcu_umask));
            }
            Tuple::Two(e1, e2) => {
                assert!(two_configs);
                assert!(!is_pcu);
                ret[0].push(format!("event=0x{:x}", e1));
                ret[1].push(format!("event=0x{:x}", e2));
            }
        };

        if !is_pcu {
            match self.0.umask {
                Tuple::One(mask) => {
                    PerfEvent::push_arg(&mut ret, format!("umask=0x{:x}", mask));
                }
                Tuple::Two(m1, m2) => {
                    assert!(two_configs);
                    ret[0].push(format!("umask=0x{:x}", m1));
                    ret[1].push(format!("umask=0x{:x}", m2));
                }
            };
        }

        if self.0.counter_mask != 0 {
            PerfEvent::push_arg(&mut ret, format!("cmask=0x{:x}", self.0.counter_mask));
        }

        if self.0.fc_mask != 0 {
            PerfEvent::push_arg(&mut ret, format!("fc_mask=0x{:x}", self.0.fc_mask));
        }

        if self.0.port_mask != 0 {
            PerfEvent::push_arg(&mut ret, format!("ch_mask=0x{:x}", self.0.port_mask));
        }

        if self.0.offcore {
            PerfEvent::push_arg(&mut ret, format!("offcore_rsp=0x{:x}", self.0.msr_value));
        } else {
            match self.0.msr_index {
                MSRIndex::One(0x3F6) => {
                    PerfEvent::push_arg(&mut ret, format!("ldlat=0x{:x}", self.0.msr_value));
                }
                MSRIndex::One(0x1A6) => {
                    PerfEvent::push_arg(&mut ret, format!("offcore_rsp=0x{:x}", self.0.msr_value));
                }
                MSRIndex::One(0x1A7) => {
                    PerfEvent::push_arg(&mut ret, format!("offcore_rsp=0x{:x}", self.0.msr_value));
                }
                MSRIndex::One(0x3F7) => {
                    PerfEvent::push_arg(&mut ret, format!("frontend=0x{:x}", self.0.msr_value));
                }
                MSRIndex::One(a) => {
                    unreachable!("Unknown MSR value {}, check linux/latest/source/tools/perf/pmu-events/jevents.c", a)
                }
                MSRIndex::Two(_, _) => {
                    unreachable!("Should not have non offcore events with two MSR index values.")
                }
                MSRIndex::None => {
                    // ignored, not a load latency event
                }
            };
        }

        if self.0.invert {
            PerfEvent::push_arg(&mut ret, String::from("inv=1"));
        }

        if self.0.edge_detect {
            PerfEvent::push_arg(&mut ret, String::from("edge=1"));
        }

        if self.0.any_thread {
            PerfEvent::push_arg(&mut ret, String::from("any=1"));
        }

        if self.match_filter("CBoFilter0[23:17]") {
            PerfEvent::push_arg(&mut ret, String::from("filter_state=0x1f"));
        }

        if self.match_filter("CBoFilter1[15:0]") {
            // TODO: Include both sockets by default -- we should probably be smarter...
            PerfEvent::push_arg(&mut ret, String::from("filter_nid=0x3"));
        }

        if self.match_filter("CBoFilter1[28:20]") {
            // TOR events requires filter_opc
            // Set to: 0x192 PrefData Prefetch Data into LLC but don’t pass to L2. Includes Hints
            PerfEvent::push_arg(&mut ret, String::from("filter_opc=0x192"));
        }

        ret
    }

    pub fn perf_qualifiers(&self) -> String {
        let mut qualifiers = String::from("S");
        if self.0.pebs == PebsType::PebsOrRegular {
            qualifiers.push('p');
        } else if self.0.pebs == PebsType::PebsOnly {
            // Adding a 'p' here is counterproducive (breaks perf), at least for Skylake
            // So do nothing
        }
        qualifiers
    }

    fn filters(&self) -> Vec<&str> {
        self.0.filter.map_or(Vec::new(), |value| {
            value
                .split(",")
                .map(|x| x.trim())
                .filter(|x| x.len() > 0)
                .collect()
        })
    }

    pub fn match_filter(&self, filter: &str) -> bool {
        self.filters().contains(&filter)
    }
}

impl<'a, 'b> fmt::Display for PerfEvent<'a, 'b> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0.event_name)
    }
}

/// Adding a new event to a group of existing events (that can be measured
/// together) can fail for a variety of reasons which are encoded in this type.
#[derive(Debug)]
pub enum AddEventError {
    /// We couldn't measure any more offcore events
    OffcoreCapacityReached,
    /// We don't have more counters left on this monitoring unit
    UnitCapacityReached(MonitoringUnit),
    /// We have a constraint that we can't measure the new event together with
    /// an existing event in the group
    CounterConstraintConflict,
    /// We have a conflict with filters
    FilterConstraintConflict,
    /// The errata specifies an issue with this event (we tend to isolate these)
    ErrataConflict,
    /// This counter must be measured alone
    TakenAloneConflict,
    /// This is one of these events that we manually specified to be isolated
    IsolatedEventConflict,
}

impl fmt::Display for AddEventError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            AddEventError::OffcoreCapacityReached => write!(f, "Offcore event limit reached."),
            AddEventError::UnitCapacityReached(u) => {
                write!(f, "Unit '{}' capacity for reached.", u)
            }
            AddEventError::CounterConstraintConflict => write!(f, "Counter constraints conflict."),
            AddEventError::FilterConstraintConflict => write!(f, "Filter constraints conflict."),
            AddEventError::ErrataConflict => write!(f, "Errata conflict."),
            AddEventError::TakenAloneConflict => write!(f, "Group contains a taken alone counter."),
            AddEventError::IsolatedEventConflict => write!(f, "Group contains an isolated event."),
        }
    }
}

impl error::Error for AddEventError {
    fn description(&self) -> &str {
        match *self {
            AddEventError::OffcoreCapacityReached => "Offcore event limit reached.",
            AddEventError::UnitCapacityReached(_) => "Unit capacity reached.",
            AddEventError::CounterConstraintConflict => "Counter constraints conflict.",
            AddEventError::FilterConstraintConflict => "Filter constraints conflict.",
            AddEventError::ErrataConflict => "Errata conflict.",
            AddEventError::TakenAloneConflict => "Group contains a taken alone counter.",
            AddEventError::IsolatedEventConflict => "Group contains an isolated event.",
        }
    }
}

#[derive(Debug)]
pub struct PerfEventGroup<'a, 'b>
where
    'b: 'a,
{
    events: Vec<PerfEvent<'a, 'b>>,
    limits: &'a HashMap<MonitoringUnit, usize>,
}

impl<'a, 'b> PerfEventGroup<'a, 'b> {
    /// Make a new performance event group.
    pub fn new(unit_sizes: &'a HashMap<MonitoringUnit, usize>) -> PerfEventGroup {
        PerfEventGroup {
            events: Default::default(),
            limits: unit_sizes,
        }
    }

    /// Returns how many offcore events are in the group.
    fn offcore_events(&self) -> usize {
        self.events.iter().filter(|e| e.is_offcore()).count()
    }

    /// Returns how many uncore events are in the group for a given unit.
    fn events_by_unit(&self, unit: MonitoringUnit) -> Vec<&PerfEvent> {
        self.events.iter().filter(|e| e.unit() == unit).collect()
    }

    /// Backtracking algorithm to find assigment of events to available counters
    /// while respecting the counter constraints every event has.
    /// The events passed here should all have the same counter type
    /// (i.e., either all programmable or all fixed) and the same unit.
    ///
    /// Returns a possible placement or None if no assignment was possible.
    fn find_counter_assignment(
        level: usize,
        max_level: usize,
        events: Vec<&'a PerfEvent<'a, 'b>>,
        assignment: Vec<&'a PerfEvent<'a, 'b>>,
    ) -> Option<Vec<&'a PerfEvent<'a, 'b>>> {
        // Are we done yet?
        if events.len() == 0 {
            return Some(assignment);
        }
        // Are we too deep?
        if level >= max_level {
            return None;
        }

        for (idx, event) in events.iter().enumerate() {
            let mask: usize = match event.counter() {
                Counter::Programmable(mask) => mask as usize,
                Counter::Fixed(mask) => mask as usize,
            };

            let mut assignment = assignment.clone();
            let mut events = events.clone();

            // If event supports counter, let's assign it to this counter and go deeper
            if (mask & (1 << level)) > 0 {
                assignment.push(event);
                events.remove(idx);
                let ret = PerfEventGroup::find_counter_assignment(
                    level + 1,
                    max_level,
                    events,
                    assignment,
                );
                if ret.is_some() {
                    return ret;
                }
            }
            // Otherwise let's not assign the event at this level and go deeper (for groups that
            // don't use all counters)
            else {
                let ret = PerfEventGroup::find_counter_assignment(
                    level + 1,
                    max_level,
                    events,
                    assignment,
                );
                if ret.is_some() {
                    return ret;
                }
            }
            // And finally, just try with the next event in the list
        }

        None
    }

    /// Check if this event conflicts with the counter requirements
    /// of events already in this group
    fn has_counter_constraint_conflicts(&self, new_event: &PerfEvent) -> bool {
        let unit = new_event.unit();
        let unit_limit = *self.limits.get(&unit).unwrap_or(&0);
        //error!("unit = {:?} unit_limit {:?}", unit, unit_limit);

        // Get all the events that share the same counters as new_event:
        let mut events: Vec<&PerfEvent> = self
            .events_by_unit(unit)
            .into_iter()
            .filter(|c| match (c.counter(), new_event.counter()) {
                (Counter::Programmable(_), Counter::Programmable(_)) => true,
                (Counter::Fixed(_), Counter::Fixed(_)) => true,
                _ => false,
            })
            .collect();

        events.push(new_event);
        PerfEventGroup::find_counter_assignment(0, unit_limit, events, Vec::new()).is_none()
    }

    /// Check if this events conflicts with the filter requirements of
    /// events already in this group
    fn has_filter_constraint_conflicts(&self, new_event: &PerfEvent) -> bool {
        let unit = new_event.unit();
        let events: Vec<&PerfEvent> = self.events_by_unit(unit);

        for event in events.iter() {
            for filter in event.filters() {
                if new_event.filters().contains(&filter) {
                    return true;
                }
            }
        }

        false
    }

    /// Try to add an event to an event group.
    ///
    /// Returns true if the event can be added to the group, false if we would be Unable
    /// to measure the event in the same group (given the PMU limitations).
    ///
    /// Things we consider correctly right now:
    /// * Fixed amount of counters per monitoring unit (so we don't multiplex).
    /// * Some events can only use some counters.
    /// * Taken alone attribute of the events.
    ///
    /// Things we consider not entirely correct right now:
    /// * Event Erratas this is not complete in the JSON files, and we just run them in isolation
    ///
    pub fn add_event(&mut self, event: PerfEvent<'a, 'b>) -> Result<(), AddEventError> {
        // 1. Can't measure more than two offcore events:
        if event.is_offcore() && self.offcore_events() == 2 {
            return Err(AddEventError::OffcoreCapacityReached);
        }

        // 2. Check we don't measure more events than we have counters
        // for on the repspective units
        let unit = event.unit();
        let unit_limit = *self.limits.get(&unit).unwrap_or(&0);
        if self.events_by_unit(unit).len() >= unit_limit {
            return Err(AddEventError::UnitCapacityReached(unit));
        }

        // 3. Now, consider the counter <-> event mapping constraints:
        // Try to see if there is any event already in the group
        // that would conflict when running together with the new `event`:
        if self.has_counter_constraint_conflicts(&event) {
            return Err(AddEventError::CounterConstraintConflict);
        }

        if self.has_filter_constraint_conflicts(&event) {
            return Err(AddEventError::FilterConstraintConflict);
        }

        // 4. Isolate things that have erratas to not screw other events (see HSW30)
        let errata = self.events.iter().any(|cur| cur.0.errata.is_some());
        if errata || event.0.errata.is_some() && self.events.len() != 0 {
            return Err(AddEventError::ErrataConflict);
        }

        // 5. If an event has the taken alone attribute set it needs to be measured alone
        let already_have_taken_alone_event = self.events.iter().any(|cur| cur.0.taken_alone);
        if already_have_taken_alone_event || event.0.taken_alone && self.events.len() != 0 {
            return Err(AddEventError::TakenAloneConflict);
        }

        // 6. If our own isolate event list contains the name we also run them alone:
        let already_have_isolated_event = self.events.get(0).map_or(false, |e| {
            ISOLATE_EVENTS.iter().any(|cur| *cur == e.0.event_name)
        });
        if already_have_isolated_event
            || ISOLATE_EVENTS.iter().any(|cur| *cur == event.0.event_name) && self.events.len() != 0
        {
            return Err(AddEventError::IsolatedEventConflict);
        }

        self.events.push(event);
        Ok(())
    }

    /// Find the right config to use for every event in the group.
    ///
    /// * We need to make sure we use the correct config if we have two offcore events in the same group.
    pub fn get_perf_config(&self) -> Vec<String> {
        let mut event_strings: Vec<String> = Vec::with_capacity(2);
        let mut have_one_offcore = false; // Have we already added one offcore event?

        for event in self.events.iter() {
            let (devices, mut configs) = event.perf_configs();

            if devices.len() == 0 || configs.len() == 0 {
                error!(
                    "Event {} supported by hardware, but your Linux does not allow you to measure it (available PMU devices = {:?})",
                    event, devices
                );

                continue;
            }

            // TODO: handle fixed counters
            // fixed_counters = {
            //    "inst_retired.any": (0xc0, 0, 0),
            //    "cpu_clk_unhalted.thread": (0x3c, 0, 0),
            //    "cpu_clk_unhalted.thread_any": (0x3c, 0, 1),
            // }

            // Adding offcore event:
            if event.is_offcore() {
                assert!(devices.len() == 1);
                assert!(configs.len() == 2);
                assert!(devices[0] == "cpu");

                let config = match have_one_offcore {
                    false => configs.get(0).unwrap(), // Ok, always has at least one config
                    true => configs.get(1).unwrap(),  // Ok, as offcore implies two configs
                };

                event_strings.push(format!(
                    "{}/{}/{}",
                    devices[0],
                    config.join(","),
                    event.perf_qualifiers()
                ));
                have_one_offcore = true;
            }
            // Adding uncore event:
            else if event.is_uncore() {
                assert!(configs.len() == 1);

                // If we have an uncore event we just go ahead and measure it on all possible devices:
                for device in devices {
                    // Patch name in config so we know where this event was running
                    // `perf stat` just reports CPU 0 for uncore events :-(
                    configs[0][0] = format!("name={}.{}", device, event.0.event_name);
                    event_strings.push(format!(
                        "{}/{}/{}",
                        device,
                        configs[0].join(","),
                        event.perf_qualifiers()
                    ));
                }
            }
            // Adding normal event:
            else {
                assert!(devices.len() == 1);
                assert!(configs.len() == 1);
                assert!(devices[0] == "cpu");

                event_strings.push(format!(
                    "{}/{}/{}",
                    devices[0],
                    configs[0].join(","),
                    event.perf_qualifiers()
                ));
            }
        }

        event_strings
    }

    /// Returns a list of events as strings that can be passed to perf-record using
    /// the -e arguments.
    pub fn get_perf_config_strings(&self) -> Vec<String> {
        self.get_perf_config()
    }

    /// Returns a list of event names in this group.
    ///
    /// The order of the list of names matches with the order
    /// returned by `get_perf_config_strings` or `get_perf_config`.
    pub fn get_event_names(&self) -> Vec<&'b str> {
        self.events.iter().map(|event| event.0.event_name).collect()
    }
}

/// Given a list of events, create a list of event groups that can be measured together.
pub fn schedule_events<'a, 'b>(events: Vec<&'a EventDescription<'b>>) -> Vec<PerfEventGroup<'a, 'b>>
where
    'b: 'a,
{
    let mut groups: Vec<PerfEventGroup> = Vec::with_capacity(42);

    for event in events {
        if IGNORE_EVENTS.contains_key(event.event_name) {
            continue;
        }

        let perf_event: PerfEvent = PerfEvent(event);
        let mut added: Result<(), AddEventError> = Err(AddEventError::ErrataConflict);
        match perf_event.unit() {
            MonitoringUnit::Unknown => {
                info!("Ignoring event with unknown unit '{}'", event);
                continue;
            }
            _ => (),
        };

        // Try to add the event to an existing group:
        for group in groups.iter_mut() {
            let perf_event: PerfEvent = PerfEvent(event);
            added = group.add_event(perf_event);
            if added.is_ok() {
                break;
            }
        }

        // Unable to add event to any existing group, make a new group instead:
        if !added.is_ok() {
            let mut pg = PerfEventGroup::new(&*PMU_COUNTERS);
            let perf_event: PerfEvent = PerfEvent(event);

            let added = pg.add_event(perf_event);
            match added {
                Err(e) => {
                    let perf_event: PerfEvent = PerfEvent(event);
                    panic!(
                        "Can't add a new event {:?} to an empty group: {:?}",
                        perf_event, e
                    );
                }
                Ok(_) => (),
            };

            groups.push(pg);
        }
    }

    // println!("{:?}", groups);
    groups
}

pub fn get_perf_command(
    cmd_working_dir: &str,
    _output_path: &Path,
    env: &Vec<(String, String)>,
    breakpoints: &Vec<String>,
    record: bool,
) -> Command {
    let mut perf = Command::new("perf");
    perf.current_dir(cmd_working_dir);
    let _filename: String;
    if !record {
        perf.arg("stat");
        perf.arg("-aA");
        perf.arg("-I 250");
        perf.arg("-x ;");
    } else {
        perf.arg("record");
        perf.arg("--group");
        perf.arg("-F 4");
        perf.arg("-a");
        perf.arg("--raw-samples");
    }

    // Ensure we use dots as number separators in csv output (see issue #1):
    perf.env("LC_NUMERIC", "C");

    // Add the environment variables:
    for &(ref key, ref value) in env.iter() {
        perf.env(key, value);
    }
    let breakpoint_args: Vec<String> = breakpoints.iter().map(|s| format!("-e \\{}", s)).collect();
    perf.args(breakpoint_args.as_slice());

    perf
}

pub fn profile<'a, 'b>(
    output_path: &Path,
    cmd_working_dir: &str,
    cmd: Vec<String>,
    env: Vec<(String, String)>,
    breakpoints: Vec<String>,
    record: bool,
    events: Option<Vec<&'a EventDescription<'b>>>,
    dryrun: bool,
) where
    'b: 'a,
{
    let event_groups = match events {
        Some(evts) => schedule_events(evts),
        None => schedule_events(get_known_events()),
    };

    // Is this run already done (in case we restart):
    let mut completed_file: PathBuf = output_path.to_path_buf();
    completed_file.push("completed");
    if completed_file.exists() {
        warn!(
            "Run {} already completed, skipping.",
            output_path.to_string_lossy()
        );
        return;
    }

    create_out_directory(output_path);
    if !dryrun {
        check_for_perf();
        let ret = check_for_perf_permissions()
            || check_for_disabled_nmi_watchdog()
            || check_for_perf_paranoia();
        if !ret {
            std::process::exit(3);
        }

        let _ = save_numa_topology(&output_path).expect("Can't save NUMA topology");
        let _ = save_cpu_topology(&output_path).expect("Can't save CPU topology");
        let _ = save_lstopo(&output_path).expect("Can't save lstopo information");
        let _ = save_cpuid(&output_path).expect("Can't save CPUID information");
        let _ = save_likwid_topology(&output_path).expect("Can't save likwid information");
    }

    assert!(cmd.len() >= 1);
    let mut perf_log = PathBuf::new();
    perf_log.push(output_path);
    perf_log.push("perf.csv");

    let mut wtr = csv::Writer::from_file(perf_log).unwrap();
    let r = wtr.encode((
        "command",
        "event_names",
        "perf_events",
        "breakpoints",
        "datafile",
        "perf_command",
        "stdout",
        "stdin",
    ));
    assert!(r.is_ok());

    // For warm-up do a dummy run of the program with perf
    let record_path = Path::new("/dev/null");
    let mut perf = get_perf_command(cmd_working_dir, output_path, &env, &breakpoints, record);
    perf.arg("-n"); // null run - don’t start any counters
    let (_, _, _) = execute_perf(&mut perf, &cmd, &Vec::new(), &record_path, dryrun);
    debug!("Warmup complete, let's start measuring.");

    let mut pb = ProgressBar::new(event_groups.len() as u64);

    for (idx, group) in event_groups.iter().enumerate() {
        if !dryrun {
            pb.inc();
        }

        let event_names: Vec<&str> = group.get_event_names();
        let counters: Vec<String> = group.get_perf_config_strings();

        let mut record_path = PathBuf::new();
        let filename = match record {
            false => format!("{}_stat.csv", idx + 1),
            true => format!("{}_perf.data", idx + 1),
        };
        record_path.push(output_path);
        record_path.push(&filename);

        let mut perf = get_perf_command(cmd_working_dir, output_path, &env, &breakpoints, record);
        let (executed_cmd, stdout, stdin) =
            execute_perf(&mut perf, &cmd, &counters, record_path.as_path(), dryrun);
        if !dryrun {
            let r = wtr.encode(vec![
                cmd.join(" "),
                event_names.join(","),
                counters.join(","),
                String::new(),
                filename,
                executed_cmd,
                stdout,
                stdin,
            ]);
            assert!(r.is_ok());

            let r = wtr.flush();
            assert!(r.is_ok());
        }
    }

    // Mark this run as completed:
    let _ = File::create(completed_file.as_path()).unwrap();
}

pub fn check_for_perf() {
    match Command::new("perf").output() {
        Ok(out) => {
            if out.status.code() != Some(1) {
                error!("'perf' seems to have some problems?");
                debug!("perf exit status was: {}", out.status);
                error!("{}", String::from_utf8_lossy(&out.stderr));
                error!(
                    "You may require a restart after fixing this so \
                     `/sys/bus/event_source/devices` is updated!"
                );
                std::process::exit(2);
            }
        }
        Err(_) => {
            error!(
                "'perf' does not seem to be executable? You may need to install it (Ubuntu: \
                 `sudo apt-get install linux-tools-common`)."
            );
            error!(
                "You may require a restart after fixing this so \
                 `/sys/bus/event_source/devices` is updated!"
            );
            std::process::exit(2);
        }
    }
}

pub fn check_for_perf_permissions() -> bool {
    let path = Path::new("/proc/sys/kernel/kptr_restrict");
    let mut file = File::open(path).expect("kptr_restrict file does not exist?");
    let mut s = String::new();

    match file.read_to_string(&mut s) {
        Ok(_) => {
            match s.trim() {
                "1" => {
                    error!(
                        "kptr restriction is enabled. You can either run autoperf as root or \
                         do:"
                    );
                    error!("\tsudo sh -c 'echo 0 >> {}'", path.display());
                    error!("to disable.");
                    return false;
                }
                "0" => {
                    // debug!("kptr_restrict is already disabled (good).");
                }
                _ => {
                    warn!(
                        "Unkown content read from '{}': {}. Proceeding anyways...",
                        path.display(),
                        s.trim()
                    );
                }
            }
        }

        Err(why) => {
            error!("Couldn't read {}: {}", path.display(), why.description());
            std::process::exit(3);
        }
    }

    true
}

pub fn check_for_disabled_nmi_watchdog() -> bool {
    let path = Path::new("/proc/sys/kernel/nmi_watchdog");
    let mut file = File::open(path).expect("nmi_watchdog file does not exist?");
    let mut s = String::new();

    match file.read_to_string(&mut s) {
        Ok(_) => {
            match s.trim() {
                "1" => {
                    error!(
                        "nmi_watchdog is enabled. This can lead to counters not read (<not \
                         counted>). Execute"
                    );
                    error!("\tsudo sh -c 'echo 0 > {}'", path.display());
                    error!("to disable.");
                    return false;
                }
                "0" => {
                    // debug!("nmi_watchdog is already disabled (good).");
                }
                _ => {
                    warn!(
                        "Unkown content read from '{}': {}. Proceeding anyways...",
                        path.display(),
                        s.trim()
                    );
                }
            }
        }

        Err(why) => {
            error!("Couldn't read {}: {}", path.display(), why.description());
            std::process::exit(4);
        }
    }

    true
}

pub fn check_for_perf_paranoia() -> bool {
    let path = Path::new("/proc/sys/kernel/perf_event_paranoid");
    let mut file = File::open(path).expect("perf_event_paranoid file does not exist?");
    let mut s = String::new();

    let res = match file.read_to_string(&mut s) {
        Ok(_) => {
            let digit = i64::from_str(s.trim()).unwrap_or_else(|_op| {
                warn!(
                    "Unkown content read from '{}': {}. Proceeding anyways...",
                    path.display(),
                    s.trim()
                );
                1
            });

            if digit >= 0 {
                error!(
                    "perf_event_paranoid is enabled. This means we can't collect system wide \
                     stats. Execute"
                );
                error!("\tsudo sh -c 'echo -1 > {}'", path.display());
                error!("to disable.");
                false
            } else {
                true
            }
        }

        Err(why) => {
            error!("Couldn't read {}: {}", path.display(), why.description());
            std::process::exit(4);
        }
    };

    res
}
