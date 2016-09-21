use std;

use std::io::prelude::*;
use std::fs;
use std::fs::File;
use std::process::Command;
use std::error::Error;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use csv;

use pbr::ProgressBar;
use x86::shared::perfcnt::{core_counters, uncore_counters};
use x86::shared::perfcnt::intel::description::{IntelPerformanceCounterDescription, Tuple, MSRIndex, Counter};
use x86::shared::{cpuid};

lazy_static! {
    static ref HT_AVAILABLE: bool = {
        let cpuid = cpuid::CpuId::new();
        cpuid.get_extended_topology_info().unwrap().any(|t| {
            t.level_type() == cpuid::TopologyType::SMT
        })
    };

    static ref PMU_COUNTERS: usize = {
        let cpuid = cpuid::CpuId::new();
        cpuid.get_performance_monitoring_info().map_or(0, |info| info.number_of_counters()) as usize
    };

    static ref PMU_DEVICES: Vec<String> = {
        // TODO: Return empty Vec in case of error
        let paths = fs::read_dir("/sys/bus/event_source/devices/").expect("Can't read devices directory.");
        let mut devices = Vec::with_capacity(15);
        for p in paths {
            let path = p.expect("Is not a path.");
            let file_name = path.file_name().into_string().expect("Is valid UTF-8 string.");
            devices.push(file_name);
        }

        devices
    };

    // Sometimes the perfmon data is missing the errata information (as is the case with the IvyBridge file).\
    // We provide a list of IvyBridge events instead here.
    static ref ISOLATE_EVENTS: Vec<&'static str> = {
        let cpuid = cpuid::CpuId::new();
        cpuid.get_feature_info().map_or(
            vec![],
            |fi| {
                let (family, extended_model, model) = (fi.family_id(), fi.extended_model_id(), fi.model_id());
                // IvyBridge
                if family == 0x6 && extended_model == 0x3 && model == 0xa {
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
                            "MEM_UOPS_RETIRED.SPLIT_STORES" ]
                }
                else {
                    vec![]
                }
        })
    };
}

fn execute_perf(perf: &mut Command, cmd: &Vec<&str>, counters: &Vec<String>, datafile: &Path) -> String {
    assert!(cmd.len() >= 1);
    let mut perf = perf.arg("-o").arg(datafile.as_os_str());
    let events: Vec<String> = counters.iter().map(|c| format!("-e {}", c)).collect();

    let mut perf = perf.args(events.as_slice());
    let mut perf = perf.args(cmd.as_slice());
    let perf_cmd_str: String = format!("{:?}", perf).replace("\"", "");

    match perf.output() {
        Ok(out) => {
            if !out.status.success() {
                error!("perf command: {} got unknown exit status was: {}", perf_cmd_str, out.status);
                debug!("stderr:\n{}", String::from_utf8(out.stderr).unwrap_or("Can't parse output".to_string()));
            }
            if !datafile.exists() {
                error!("perf command: {} succeeded but did not produce the required file {:?} (you should file a bug report!)", perf_cmd_str, datafile);
            }
        },
        Err(err) => {
            error!("Executing {} failed : {}", perf_cmd_str, err)
        }
    }

    perf_cmd_str
}

fn create_out_directory(out_dir: &Path) {
    if !out_dir.exists() {
        std::fs::create_dir(out_dir).expect("Can't create `out` directory");
    }
}

fn get_events() -> Vec<&'static IntelPerformanceCounterDescription> {
    let mut events: Vec<&IntelPerformanceCounterDescription> = core_counters().unwrap().values().collect();
    let mut uncore_events: Vec<&IntelPerformanceCounterDescription> = uncore_counters().unwrap().values().collect();
    events.append(&mut uncore_events);

    events
}

pub enum UncoreType {
    /// Memory stuff
    Arb,
    /// The CBox manages the interface between the core and the LLC, so
    /// the instances of uncore CBox is equal to number of cores
    CBox,
    /// ???
    SBox,
    /// QPI Stuff
    QPI,
    /// Types we don't know how to handle...
    Unknown(&'static str)
}

impl UncoreType {

    fn new(unit: &'static str) -> UncoreType {
        match unit.to_lowercase().as_str() {
            "cbo" => UncoreType::CBox,
            "qpi_ll" => UncoreType::QPI,
            "sbo" => UncoreType::SBox,
            "imph-u" => UncoreType::Arb,
            "arb" => UncoreType::Arb,
            _ => UncoreType::Unknown(unit)
        }
    }

    /// Return the perf prefix for selecting the right PMU unit in case of uncore counters.
    fn to_perf_prefix(&self) -> Option<&'static str> {

        let res = match *self {
            UncoreType::CBox => Some("uncore_cbox"),
            UncoreType::QPI => Some("uncore_qpi"),
            UncoreType::SBox => Some("uncore_sbox"),
            UncoreType::Arb => Some("uncore_arb"),
            UncoreType::Unknown(_) => None
        };

        // Note: If anything here does not return uncore_ as a prefix, you need to update extract.rs!
        res.map(|string| { assert!(string.starts_with("uncore_")) });

        res
    }
}


#[derive(Debug)]
struct PerfEvent(&'static IntelPerformanceCounterDescription);

impl PerfEvent {

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

        if self.is_uncore() {
            let typ = self.uncore_type().expect("is_uncore() implies uncore type");

            // XXX: Horrible vector transformation:
            let matched_devices: Vec<String> = PMU_DEVICES
                .iter()
                .filter(|d| typ.to_perf_prefix().map_or(false, |t| d.starts_with(t)) ).map(|d| d.clone()).collect();
            devices.extend(matched_devices);
            if devices.len() == 0 {
                debug!("Got no device to measure this event:\n{:?}", self.0);
            }
        }
        else { // Offcore or regular event
            devices.push(String::from("cpu"));
        }

        for args in self.perf_args() {
            configs.push(args);
        }

        (devices, configs)
    }

    /// Is this event an uncore event?
    pub fn is_uncore(&self) -> bool {
        self.0.unit.is_some()
    }

    fn uncore_type(&self) -> Option<UncoreType> {
        self.0.unit.map(|u| UncoreType::new(u) )
    }

    /// Is this event an offcore event?
    pub fn is_offcore(&self) -> bool {
        match self.0.event_code {
            Tuple::One(_) => {
                assert!(!self.0.offcore);
                false
            },
            Tuple::Two(_, _) => {
                assert!(self.0.event_name.contains("OFFCORE"));
                // The OR is because there is this weird meta-event OFFCORE_RESPONSE
                // in the data files. It has offcore == false and is not really a proper event :/
                assert!(self.0.offcore || self.0.event_name == "OFFCORE_RESPONSE");
                true
            },
        }
    }

    /// Get the correct counter mask
    pub fn counter(&self) -> Counter {
        if *HT_AVAILABLE {
            self.0.counter
        } else {
            self.0.counter_ht_off
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
            Tuple::Two(_, _) => true
        };

        let mut ret: Vec<Vec<String>> = vec![ Vec::with_capacity(7) ];
        ret[0].push(format!("name={}", self.0.event_name));
        if two_configs {
            ret.push(Vec::with_capacity(7));
            ret[1].push(format!("name={}", self.0.event_name));
        }

        match self.0.event_code {
            Tuple::One(ev) => ret[0].push(format!("event=0x{:x}", ev)),
            Tuple::Two(e1, e2) => {
                assert!(two_configs);
                ret[0].push(format!("event=0x{:x}", e1));
                ret[1].push(format!("event=0x{:x}", e2));
            }
        };

        match self.0.umask {
            Tuple::One(mask) => ret[0].push(format!("umask=0x{:x}", mask)),
            Tuple::Two(m1, m2) => {
                assert!(two_configs);
                ret[0].push(format!("umask=0x{:x}", m1));
                ret[1].push(format!("umask=0x{:x}", m2));
            }
        };

        if self.0.counter_mask != 0 {
            ret[0].push(format!("cmask=0x{:x}", self.0.counter_mask));
            if two_configs { ret[1].push(format!("cmask=0x{:x}", self.0.counter_mask)); }
        }

        if self.0.offcore {
            ret[0].push(format!("offcore_rsp=0x{:x}", self.0.msr_value));
            if two_configs { ret[1].push(format!("offcore_rsp=0x{:x}", self.0.msr_value)); }
        }
        else {
            match self.0.msr_index {
                MSRIndex::One(idx) => {
                    assert!(idx == 0x3F6); // Load latency MSR
                    ret[0].push(format!("ldlat=0x{:x}", self.0.msr_value));
                    if two_configs { ret[1].push(format!("ldlat=0x{:x}", self.0.msr_value)); }
                }
                MSRIndex::Two(_, _) => unreachable!("Should not have non offcore events with two MSR index values."),
                MSRIndex::None => { /* ignored, not a load latency event */ },
            };
        }

        if self.0.invert {
            ret[0].push(String::from("inv=1"));
            if two_configs { ret[1].push(String::from("inv=1")); }
        }

        if self.0.edge_detect {
            ret[0].push(String::from("edge=1"));
            if two_configs { ret[1].push(String::from("edge=1")); }
        }

        ret
    }

}

#[derive(Debug)]
struct PerfEventGroup {
    events: Vec<PerfEvent>,
    size: usize,
}

impl PerfEventGroup {

    /// Make a new performance event group.
    ///
    /// Group size should not exceed amount of available counters.
    pub fn new(group_size: usize) -> PerfEventGroup {
        PerfEventGroup { size: group_size, events: Vec::with_capacity(group_size) }
    }

    /// Returns how many offcore events are in the group.
    fn offcore_events(&self) -> usize {
        self.events.iter().filter(|e| {
            e.is_offcore()
        }).count()
    }

    /// Returns how many uncore events are in the group.
    fn uncore_events(&self) -> usize {
        self.events.iter().filter(|e| {
            e.is_uncore()
        }).count()
    }

    /// Try to add an event to an event group.
    ///
    /// Returns true if the event can be added to the group, false if we would be Unable
    /// to measure the event in the same group (given the PMU limitations).
    ///
    /// Things we consider (correctly) right now:
    /// * Can't have more than two offcore events because we only have two MSRs to measure them.
    /// * Some events can only use some counters.
    /// * Taken alone attribute of the events
    ///
    /// Things we consider (not entirely correct) right now:
    /// * Event Erratas (not really detailed, just run them in isolation)
    /// * Uncore limitations (just not more than two at once)
    ///
    pub fn add_event(&mut self, event: PerfEvent) -> bool {
        if self.events.len() >= self.size {
            false
        }
        else {
            // 1. Can't measure more than two offcore events:
            if event.is_offcore() && self.offcore_events() == 2 {
                return false;
            }

            // Don't measure more than two uncore events:
            // (I don't know how to get the exact limits for every uncore piece in the system
            // but two seems like a sane limit where we don't have multi-plexing)
            if event.is_uncore() && self.uncore_events() == 2 {
                return false;
            }

            // 2. Now, consider the counter <-> event mapping constraints:
            // Try to see if there is any event already in the group
            // that would conflicts when running together with the new `event`:
            let conflicts = self.events.iter().any(|cur| {
                match cur.counter() {
                    Counter::Programmable(cmask) => {
                        match event.counter() {
                            Counter::Programmable(emask) => (cmask | emask).count_ones() < 2,
                            _ => false, // No conflict
                        }
                    },
                    Counter::Fixed(cmask) => {
                        match event.counter() {
                            Counter::Fixed(emask) => (cmask | emask).count_ones() < 2,
                            _ => false, // No conflict
                        }
                    },
                }
            });
            if conflicts {
                return false;
            }

            // 3. Isolate things that have erratas to not screw other events (see HSW30)
            let errata = self.events.iter().any(|cur| {
                cur.0.errata.is_some()
            });
            if errata || event.0.errata.is_some() && self.events.len() != 0 {
                return false;
            }

            // 4. If an event has the taken alone attribute set it needs to be measured alone
            let have_taken_alone_event = self.events.iter().any(|cur| {
                cur.0.taken_alone
            });
            if have_taken_alone_event || event.0.taken_alone && self.events.len() != 0 {
                return false;
            }

            // 5. If our own isolate event list contains the name we also run them alone:
            let have_isolated_event = self.events.get(0).map_or(false, |e| {
                ISOLATE_EVENTS.iter().any(|cur| *cur == e.0.event_name)
            });
            if have_isolated_event || ISOLATE_EVENTS.iter().any(|cur| *cur == event.0.event_name) && self.events.len() != 0 {
                return false;
            }

            self.events.push(event);
            true
        }
    }

    /// Find the right config to use for every event in the group.
    ///
    /// * We need to make sure we use the correct config if we have two offcore events in the same group.
    pub fn get_perf_config(&self) -> Vec<String> {
        let mut event_strings: Vec<String> = Vec::with_capacity(self.size);
        let mut have_one_offcore = false; // Have we already added one offcore event?

        for event in self.events.iter() {
            let (devices, mut configs) = event.perf_configs();

            // Adding offcore event:
            if event.is_offcore() {
                assert!(devices.len() == 1);
                assert!(configs.len() == 2);
                assert!(devices[0] == "cpu");

                let config = match have_one_offcore {
                    false => configs.get(0).unwrap(), // Ok, always has at least one config
                    true => configs.get(1).unwrap() // Ok, as offcore implies two configs
                };

                event_strings.push( format!("{}/{}/S", devices[0], config.join(",")) );
                have_one_offcore = true;
            }
            // Adding uncore event:
            else if event.is_uncore() {
                assert!(configs.len() == 1);

                // We can have no devices if we don't understand how to match the unit name to perf names:
                if devices.len() == 0 {
                    debug!("Event '{}' currently not supported, ignored.", event.0.event_name);
                }

                // If we have an uncore event we just go ahead and measure it on all possible devices:
                for device in devices {
                    // Patch name in config so we know where this event was running
                    // `perf stat` just reports CPU 0 for uncore events :-(
                    configs[0][0] = format!("name={}.{}", device, event.0.event_name);
                    event_strings.push( format!("{}/{}/S", device, configs[0].join(",") ) );
                }
            }
            // Adding normal event:
            else  {
                assert!(devices.len() == 1);
                assert!(configs.len() == 1);
                assert!(devices[0] == "cpu");

                event_strings.push( format!("{}/{}/S", devices[0], configs[0].join(",")) );
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
    pub fn get_event_names(&self) -> Vec<&'static str> {
        self.events.iter().map(|event| event.0.event_name).collect()
    }

}

/// Given a list of events, create a list of event groups that can be measured together.
fn schedule_events(events: Vec<&'static IntelPerformanceCounterDescription>) -> Vec<PerfEventGroup> {
    if *PMU_COUNTERS == 0 {
        error!("No PMU counters? Can't measure anything.");
        return Vec::default();
    }
    let expected_groups = events.len() / *PMU_COUNTERS;
    let mut groups: Vec<PerfEventGroup> = Vec::with_capacity(expected_groups);

    for event in events {
        let mut added: bool = false;
        // Try to add the event to an existing group:
        for group in groups.iter_mut() {
            let perf_event: PerfEvent = PerfEvent(event);
            added = group.add_event(perf_event);
            if added {
                break;
            }
        }

        // Unable to add event to any existing group, make a new group instead:
        if !added {
            let mut pg = PerfEventGroup::new(*PMU_COUNTERS);
            let perf_event: PerfEvent = PerfEvent(event);
            let ret = pg.add_event(perf_event);
            assert!(ret == true); // Should always be able to add an event to an empty group
            groups.push(pg);
        }
    }

    //println!("{:?}", groups);
    groups
}

pub fn profile(output_path: &Path, cmd: Vec<&str>, env: Vec<(String, String)>, record: bool) {
    create_out_directory(output_path);
    check_for_perf();
    let ret = check_for_perf_permissions() || check_for_disabled_nmi_watchdog() || check_for_perf_paranoia();
    if !ret {
        std::process::exit(3);
    }

    assert!(cmd.len() >= 1);
    let mut perf_log = PathBuf::new();
    perf_log.push(output_path);
    perf_log.push("perf.csv");

    let mut wtr = csv::Writer::from_file(perf_log).unwrap();
    let r = wtr.encode(("command", "event_names", "perf_events", "breakpoints", "datafile", "perf_command"));
    assert!(r.is_ok());

    let event_groups = schedule_events(get_events());

    let mut pb = ProgressBar::new(event_groups.len() as u64);
    for group in event_groups {
        let idx = pb.inc();

        let mut event_names: Vec<&'static str> = group.get_event_names();
        let counters: Vec<String> = group.get_perf_config_strings();

        let mut perf = Command::new("perf");
        let mut record_path = PathBuf::new();
        let mut filename: String;
        if !record {
            let mut perf = perf.arg("stat").arg("-aA").arg("-I 250").arg("-x ;");
            for &(ref key, ref value) in env.iter() {
                perf = perf.env(key, value);
            }
            record_path.push(output_path);
            filename = format!("{}_stat.csv", idx);
            record_path.push(&filename);
        }
        else {
            let mut perf = perf.arg("record").arg("--group").arg("-F 4").arg("-a").arg("--raw-samples");
            record_path.push(output_path);
            filename = format!("{}_perf.data", idx);
            record_path.push(&filename);
        }

        let executed_cmd = execute_perf(&mut perf, &cmd, &counters, record_path.as_path());
        let r = wtr.encode(vec![cmd.join(" "), event_names.join(","), counters.join(","), String::new(), filename, executed_cmd]);
        assert!(r.is_ok());

        let r = wtr.flush();
        assert!(r.is_ok());
    }

}

fn check_for_perf() {
    match Command::new("perf").output() {
        Ok(out) => {
            if out.status.code() != Some(1) {
                error!("'perf' seems to have some problems?");
                debug!("perf exit status was: {}", out.status);
                error!("{}", String::from_utf8_lossy(&out.stderr));
                error!("You may require a restart after fixing this so `/sys/bus/event_source/devices` is updated!");
                std::process::exit(2);
            }
        },
        Err(_) => {
            error!("'perf' does not seem to be executable? You may need to install it (Ubuntu: `sudo apt-get install linux-tools-common`).");
            error!("You may require a restart after fixing this so `/sys/bus/event_source/devices` is updated!");
            std::process::exit(2);
        }
    }
}

fn check_for_perf_permissions() -> bool {
    let path = Path::new("/proc/sys/kernel/kptr_restrict");
    let mut file = File::open(path).expect("kptr_restrict file does not exist?");
    let mut s = String::new();

    match file.read_to_string(&mut s) {
        Ok(_) =>  {
            match s.trim() {
                "1" => {
                    error!("kptr restriction is enabled. You can either run autoperf as root or do:");
                    error!("\tsudo sh -c 'echo 0 >> {}'", path.display());
                    error!("to disable.");
                    return false;
                }
                "0" => {
                    //debug!("kptr_restrict is already disabled (good).");
                }
                _ => {
                    warn!("Unkown content read from '{}': {}. Proceeding anyways...", path.display(), s.trim());
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

fn check_for_disabled_nmi_watchdog() -> bool {
    let path = Path::new("/proc/sys/kernel/nmi_watchdog");
    let mut file = File::open(path).expect("nmi_watchdog file does not exist?");
    let mut s = String::new();

    match file.read_to_string(&mut s) {
        Ok(_) =>  {
            match s.trim() {
                "1" => {
                    error!("nmi_watchdog is enabled. This can lead to counters not read (<not counted>). Execute");
                    error!("\tsudo sh -c 'echo 0 > {}'", path.display());
                    error!("to disable.");
                    return false;
                }
                "0" => {
                    //debug!("nmi_watchdog is already disabled (good).");
                }
                _ => {
                    warn!("Unkown content read from '{}': {}. Proceeding anyways...", path.display(), s.trim());
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


fn check_for_perf_paranoia() -> bool {
    let path = Path::new("/proc/sys/kernel/perf_event_paranoid");
    let mut file = File::open(path).expect("perf_event_paranoid file does not exist?");
    let mut s = String::new();

    return match file.read_to_string(&mut s) {
        Ok(_) =>  {
            let digit = i64::from_str(s.trim()).unwrap_or_else(|op| {
                warn!("Unkown content read from '{}': {}. Proceeding anyways...", path.display(), s.trim());
                1
            });

            if digit >= 0 {
                error!("perf_event_paranoid is enabled. This means we can't collect system wide stats. Execute");
                error!("\tsudo sh -c 'echo -1 > {}'", path.display());
                error!("to disable.");
                return false;
            }
            return true;
        }

        Err(why) => {
            error!("Couldn't read {}: {}", path.display(), why.description());
            std::process::exit(4);
        }
    };
}
