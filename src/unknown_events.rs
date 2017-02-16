use super::profile::{PerfEvent, MonitoringUnit, get_known_events};
use std::io::Error;

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
                     MonitoringUnit::UBox,
                     MonitoringUnit::CBox,
                     MonitoringUnit::HA,
                     MonitoringUnit::IMC,
                     MonitoringUnit::PCU,
                     MonitoringUnit::R2PCIe,
                     MonitoringUnit::R3QPI,
                     MonitoringUnit::QPI];

    println!("Find events...");
    for unit in units {
        for code in 0..255 {
            for umask in 0..255 {
                if !event_is_documented(&pevents, unit, code, umask) {
                    //println!("Undocumented event found: {:?} {:?} {:?}",
                    //         unit,
                    //         code,
                    //         umask);
                } else {
                    println!("Event documented {:?} {:?} {:?}", unit, code, umask);
                }
            }
        }
    }
}
