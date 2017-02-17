use std::io::Error;

use std::path::Path;
use std::path::PathBuf;

use x86::shared::perfcnt::intel::{EventDescription, Tuple, MSRIndex, Counter, PebsType};
use super::profile::{PerfEvent, MonitoringUnit, get_known_events, profile};

pub fn event_is_documented(events: &Vec<PerfEvent>,
                           unit: MonitoringUnit,
                           code: u8,
                           umask: u8)
                           -> bool {
    for event in events.iter() {
        if event.unit() == unit && event.uses_event_code(code) && event.uses_umask(umask) {
            return true;
        }
    }

    return false;
}


pub fn print_unknown_events() {
    let events = get_known_events();
    let pevents: Vec<PerfEvent> = events.into_iter().map(|e| PerfEvent(e)).collect();

    let units = vec![MonitoringUnit::CPU,
                     //MonitoringUnit::UBox,
                     //MonitoringUnit::CBox,
                     //MonitoringUnit::HA,
                     //MonitoringUnit::IMC,
                     //MonitoringUnit::PCU,
                     //MonitoringUnit::R2PCIe,
                     //MonitoringUnit::R3QPI,
                     //MonitoringUnit::QPI
    ];

    println!("Find events...");
    let mut events = Vec::new();

    for unit in units {
        for code in 0..100 {
            for umask in 0..10 {
                if event_is_documented(&pevents, unit, code, umask) {
                    //println!("Event documented {:?} {:?} {:?}", unit, code, umask);
                    continue;
                }
                    //println!("Undocumented event found: {:?} {:?} {:?}", unit, code, umask);

                    let e = EventDescription::new(
                               Tuple::One(code),
                               Tuple::One(umask),
                               "UNKNOWN_EVENT",
                               "unknown event",
                               None,
                               Counter::Programmable(15),
                               None,
                               None,
                               0,
                               MSRIndex::None,
                               0,
                               false,
                               0x0,
                               false,
                               false,
                               false,
                               PebsType::Regular,
                               false,
                               None,
                               false,
                               false,
                               None,
                               false,
                               None,
                               None,
                               false);
                    //let pe = PerfEvent(&e);
                    events.push(e);
                //} else {
                    //println!("Event documented {:?} {:?} {:?}", unit, code, umask);
                //}
            }
        }
    }

    let storage_location = PathBuf::from("out");
    profile(&storage_location, ".", vec![String::from("sleep"), String::from("2")], Vec::new(), Vec::new(), false, Some(events.iter().collect()))
}
