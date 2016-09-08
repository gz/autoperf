use std;

use std::fs::File;
use std::io::prelude::*;
use std::process::Command;
use std::error::Error;
use std::path::Path;
use std::path::PathBuf;
use csv;

use pbr::ProgressBar;
use x86::shared::perfcnt::{core_counters, uncore_counters};
use x86::shared::perfcnt::intel::description::{IntelPerformanceCounterDescription, Tuple, Counter};
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
}

fn perf_record(cmd: &Vec<&str>, counters: &Vec<String>, datafile: &Path) {
    assert!(cmd.len() >= 1);
    let mut perf = Command::new("perf");
    let mut perf = perf.arg("record").arg("-o").arg(datafile.as_os_str());
    let mut perf = perf.arg("--raw-samples");
    let mut perf = perf.arg("--group");
    let mut perf = perf.arg(counters.join(" "));
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

#[derive(Debug)]
struct PerfEvent(&'static IntelPerformanceCounterDescription);

struct PerfEventConfigIter<'a> {
    event: &'a PerfEvent,
    curr: usize,
}

impl<'a> Iterator for PerfEventConfigIter<'a> {

    type Item = Vec<String>;

    fn next(&mut self) -> Option<Vec<String>> {
        self.curr += 1;

        if self.curr == 1 {
            return Some(self.event.perf_args(false));
        }

        if self.curr == 2 && self.event.is_offcore() {
            return Some(self.event.perf_args(true));
        }

        None
    }
}

impl PerfEvent {

    /// Iterator over all possible event configurations.
    pub fn perf_configs(&self) -> PerfEventConfigIter {
        PerfEventConfigIter{ event: self, curr: 0 }
    }

    /// Is this event an offcore event?
    pub fn is_offcore(&self) -> bool {
        match self.0.event_code {
            Tuple::One(_) => false,
            Tuple::Two(_, _) => {
                assert!(self.0.event_name.contains("OFFCORE"));
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
    fn perf_args(&self, try_alternative: bool) -> Vec<String> {

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

        let mut args = Vec::new();

        match self.0.event_code {
            Tuple::One(ev) => args.push(format!("event=0x{:x}", ev)),
            Tuple::Two(e1, e2) => {
                if !try_alternative {
                    args.push(format!("event=0x{:x}", e1))
                }
                else {
                    args.push(format!("event=0x{:x}", e2))
                }
            }
        };

        match self.0.umask {
            Tuple::One(mask) => args.push(format!("umask=0x{:x}", mask)),
            Tuple::Two(m1, m2) => {
                if !try_alternative {
                    args.push(format!("umask=0x{:x}", m1))
                }
                else {
                    args.push(format!("umask=0x{:x}", m2))
                }
            }
        };

        if self.0.counter_mask != 0 {
            args.push(format!("cmask=0x{:x}", self.0.counter_mask));
        }

        if self.0.offcore {
            args.push(format!("offcore_rsp=0x{:x}", self.0.msr_value));
        }

        if self.0.invert {
            args.push(String::from("inv=1"));
        }

        if self.0.edge_detect {
            args.push(String::from("edge=1"));
        }

        args
    }
}

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

    /// Returns how many offcore counters are in the group.
    fn offcore_counters(&self) -> usize {
        self.events.iter().filter(|e| {
            e.is_offcore()
        }).count()
    }

    /// Try to add an event to an event group.
    ///
    /// Returns true if the event can be added to the group, false if we would be Unable
    /// to measure the event in the same group (given the PMU limitations).
    ///
    /// Things we consider right now:
    /// * Can't have more than two offcore events because we only have two MSRs to measure them.
    /// * Some events can only use some counters.
    pub fn add_event(&mut self, event: PerfEvent) -> bool {
        if self.events.len() >= self.size {
            false
        }
        else {
            // 1. Can't measure more than two offcore counters:
            if event.is_offcore() && self.offcore_counters() == 2 {
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
                //panic!("Wow, this actually triggers?");
                return false;
            }

            self.events.push(event);
            true
        }
    }

    /// Find the right config to use for every event in the group.
    ///
    /// * We need to make sure we use the correct config if we have two offcore events in the same group.
    pub fn get_perf_config(&self) -> Vec<Vec<String>> {
        let mut configs: Vec<Vec<String>> = Vec::with_capacity(self.size);
        let mut second_offcore = false; // Have we already added one offcore event?

        for event in self.events.iter() {
            configs.push(match second_offcore && event.is_offcore() {
                false => event.perf_configs().next().unwrap(), // Ok, always has at least one config
                true => event.perf_configs().nth(1).unwrap() // Ok, as offcore implies two configs
            });
            if event.is_offcore() {
                second_offcore = true;
            }
        }

        configs
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

    groups
}

pub fn profile(output_path: &Path, cmd: Vec<&str>) {
    create_out_directory(output_path);
    check_for_perf();
    check_for_perf_permissions();

    assert!(cmd.len() >= 1);
    let mut perf_log = PathBuf::new();
    perf_log.push(output_path);
    perf_log.push("perf.csv");

    let mut wtr = csv::Writer::from_file(perf_log).unwrap();
    let r = wtr.encode(("command", "counters", "breakpoints", "datafile"));
    assert!(r.is_ok());

    let event_groups = schedule_events(get_events());

    let mut pb = ProgressBar::new(event_groups.len() as u64);
    for group in event_groups {
        let mut record_path = PathBuf::new();
        let idx = pb.inc();

        record_path.push(output_path);
        let filename = format!("{}_perf.data", idx);
        record_path.push(&filename);

        let mut counters: Vec<String> = Vec::new();
        for args in group.get_perf_config() {
            let arg_string = args.join(",");
            counters.push(format!("cpu/{}/", arg_string));
        }

        perf_record(&cmd, &counters, record_path.as_path());

        let r = wtr.encode(vec![cmd.join(" "), counters.join(" "), String::new(), filename]);
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
                std::process::exit(2);
            }
        },
        Err(_) => {
            error!("'perf' does not seem to be executable? You may need to install it (Ubuntu: `sudo apt-get install linux-tools-common`).");
            std::process::exit(2);
        }
    }
}

fn check_for_perf_permissions() {
    let path = Path::new("/proc/sys/kernel/kptr_restrict");
    let mut file = File::open(path).expect("kptr_restrict file does not exist?");
    let mut s = String::new();

    match file.read_to_string(&mut s) {
        Ok(_) =>  {
            match s.trim() {
                "1" => {
                    error!("kptr restriction is enabled. You can either run autoperf as root or do:");
                    error!("\tsudo sh -c \"echo 0 >> {}\"", path.display());
                    error!("to disable.");
                    std::process::exit(3);
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
}
