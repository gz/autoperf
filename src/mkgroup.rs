use std::collections::HashMap;

use std::path::Path;

use csv;
use phf::Map;

use super::profile::{MonitoringUnit, PerfEvent, PerfEventGroup};
use log::*;
use x86::perfcnt::intel::{events, EventDescription};

pub fn mkgroup(ranking_file: &Path) {
    let core_counter: &'static Map<&'static str, EventDescription<'static>> = &events().unwrap();
    let uncore_counter: &'static Map<&'static str, EventDescription<'static>> = &events().unwrap();

    let mut res = HashMap::with_capacity(11);
    res.insert(MonitoringUnit::CPU, 4);
    res.insert(MonitoringUnit::UBox, 2);
    res.insert(MonitoringUnit::CBox, 4);
    res.insert(MonitoringUnit::HA, 4);
    res.insert(MonitoringUnit::IMC, 4);
    res.insert(MonitoringUnit::IRP, 4);
    res.insert(MonitoringUnit::PCU, 4);
    res.insert(MonitoringUnit::QPI, 4);
    res.insert(MonitoringUnit::R2PCIe, 4);
    res.insert(MonitoringUnit::R3QPI, 2); // According to the manual this is 3 but then it multiplexes...
    res.insert(MonitoringUnit::QPI, 4); // Not in the manual?

    // Accuracy,Config,Error,Event,F1 score,Precision/Recall,Samples,Samples detail,Test App
    // Accuracy,Error,Event,F1 score,Precision,Recall,Samples Test 0,Samples Test 1,Samples Test Total,Samples Training 0,Samples Training 1,Samples Training Total,Tested Application,Training Configs
    type OutputRow = (
        f64,
        String,
        String,
        f64,
        f64,
        f64,
        String,
        String,
        String,
        String,
    );
    let mut rdr = csv::Reader::from_file(ranking_file)
        .unwrap()
        .has_headers(true);
    let mut events_added = HashMap::with_capacity(25);

    let mut group = PerfEventGroup::new(&res);

    for row in rdr.decode() {
        let (_, _, feature_name, _, _, _, _, _, _, _): OutputRow = row.unwrap();
        // println!("{:?}", feature_name);
        let splits: Vec<&str> = feature_name.splitn(2, ".").collect();
        let event_name = String::from(splits[1]);
        let feature_name = String::from(feature_name.clone());

        let maybe_e: Option<&'static EventDescription> = core_counter.get(event_name.as_str());

        // If we already measure the event, just return it (in case a feature shows up with AVG. and
        // STD.)
        if events_added.contains_key(&event_name) {
            println!("{}", feature_name);
        } else {
            // Otherwise, let's see if we can still add it to the group:
            match maybe_e {
                Some(event) => match group.add_event(PerfEvent(event)) {
                    Ok(()) => {
                        events_added.insert(event_name, true);
                        println!("{}", feature_name);
                    }
                    Err(e) => info!(
                        "Unable to add event: '{}' to {:?} because of '{}'",
                        event_name, event.unit, e
                    ),
                },
                None => {
                    let maybe_ue: Option<&'static EventDescription> =
                        uncore_counter.get(event_name.as_str());
                    match maybe_ue {
                        Some(uncore_event) => match group.add_event(PerfEvent(uncore_event)) {
                            Ok(()) => {
                                events_added.insert(event_name, true);
                                println!("{}", feature_name);
                            }
                            Err(e) => info!(
                                "Unable to add event: '{}' to {:?} because of '{}'",
                                event_name, uncore_event.unit, e
                            ),
                        },
                        None => {
                            // panic!("Didn't find event {} in data set?", event_name);
                        }
                    }
                }
            };
        }
    }
}
