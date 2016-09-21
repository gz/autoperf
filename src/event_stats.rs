extern crate x86;
extern crate phf;

use std::collections::HashMap;
use x86::shared::perfcnt;
use x86::shared::perfcnt::intel::description::IntelPerformanceCounterDescription;
use phf::Map;

fn edit_distance(a: &str, b: &str) -> i32 {
    let len_a = a.chars().count();
    let len_b = b.chars().count();

    let row: Vec<i32> = vec![0; len_b + 1];
    let mut matrix: Vec<Vec<i32>> = vec![row; len_a + 1];

    let chars_a: Vec<char> = a.chars().collect();
    let chars_b: Vec<char> = b.chars().collect();

    for i in 0..len_a {
        matrix[i+1][0] = (i+1) as i32;
    }
    for i in 0..len_b {
        matrix[0][i+1] = (i+1) as i32;
    }

    for i in 0..len_a {
        for j in 0..len_b {
            let ind: i32 = if chars_a[i] == chars_b[j] { 0 } else { 1 };

            let min = vec![
                matrix[i][j+1] + 1,
                matrix[i+1][j] + 1,
                matrix[i][j] + ind
            ].into_iter().min().unwrap();

            matrix[i+1][j+1] = if min == 0 { 0 } else { min };
        }
    }
    matrix[len_a][len_b]
}

type EventMap = Map<&'static str, IntelPerformanceCounterDescription>;

// Find all events with the same name
fn common_event_names(a: Option<&'static EventMap>, b: Option<&'static EventMap>) -> usize {
    if a.is_none() || b.is_none() {
        return 0;
    }

    let a_map = a.unwrap();
    let b_map = b.unwrap();

    let mut counter = 0;
    for (key, value) in a_map.entries() {
        if b_map.get(key).is_some() {
            counter += 1
        }
    }

    counter
}


// Find all events with the same name
fn common_event_desc_distance(a: Option<&'static EventMap>, b: Option<&'static EventMap>) {
    if a.is_none() || b.is_none() {
        return;
    }

    let a_map = a.unwrap();
    let b_map = b.unwrap();

    println!("event name, edit distance");
    for (key1, value1) in a_map.entries() {
        b_map.get(key1).map(|value2| {
            assert_eq!(value1.event_name, value2.event_name);
            println!("{},{}", value1.event_name, edit_distance(value1.brief_description, value2.brief_description));
        });
    }
}



pub fn main() {

    let mut key_to_name = HashMap::new();
    key_to_name.insert("GenuineIntel-6-2E","NehalemEX");
    key_to_name.insert("GenuineIntel-6-1E","NehalemEP");
    key_to_name.insert("GenuineIntel-6-2F","WestmereEX");
    key_to_name.insert("GenuineIntel-6-25","WestmereEP-SP");
    key_to_name.insert("GenuineIntel-6-2C","WestmereEP-DP");
    key_to_name.insert("GenuineIntel-6-37","Silvermont");
    key_to_name.insert("GenuineIntel-6-5C","Goldmont");
    key_to_name.insert("GenuineIntel-6-1C","Bonnell");
    key_to_name.insert("GenuineIntel-6-2A","SandyBridge");
    key_to_name.insert("GenuineIntel-6-2D","Jaketown");
    key_to_name.insert("GenuineIntel-6-3A","IvyBridge");
    key_to_name.insert("GenuineIntel-6-3E","IvyTown");
    key_to_name.insert("GenuineIntel-6-3C","Haswell");
    key_to_name.insert("GenuineIntel-6-3F","HaswellX");
    key_to_name.insert("GenuineIntel-6-3D","Broadwell");
    key_to_name.insert("GenuineIntel-6-4F","BroadwellX");
    key_to_name.insert("GenuineIntel-6-56","BroadwellDE");
    key_to_name.insert("GenuineIntel-6-4E","Skylake");
    key_to_name.insert("GenuineIntel-6-57","KnightsLanding");

    println!("architecture,core events,uncore events");
    for (key, name) in key_to_name.iter() {
        let core_counters = perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-core", key).as_str());
        let uncore_counters = perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-uncore", key).as_str());

        println!("{},{},{}", name, core_counters.map(|c| c.len()).unwrap_or(0), uncore_counters.map(|c| c.len()).unwrap_or(0)) ;
    }

    for (key1, name1) in key_to_name.iter() {
        for (key2, name2) in key_to_name.iter() {
            let core_counters1 = perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-core", key1).as_str());
            let uncore_counters1 = perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-uncore", key1).as_str());

            let core_counters2 = perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-core", key2).as_str());
            let uncore_counters2 = perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-uncore", key2).as_str());

            println!("Comparing {} vs. {}", name1, name2);
            println!("name1,name2,common core events,common uncore events,name1 core events, name2 core events, name1 uncore events, name2 uncore event");
            println!("{},{},{},{},{},{},{},{}",
                name1, name2,
                common_event_names(core_counters1, core_counters2),
                common_event_names(uncore_counters1, uncore_counters2),
                core_counters1.map(|c| c.len()).unwrap_or(0),
                uncore_counters1.map(|c| c.len()).unwrap_or(0),
                core_counters2.map(|c| c.len()).unwrap_or(0),
                uncore_counters2.map(|c| c.len()).unwrap_or(0)
            );

            println!("Edit distance of common events");
            common_event_desc_distance(core_counters1, core_counters2);
            common_event_desc_distance(uncore_counters1, uncore_counters2);

        }
    }


    //let counter_description = perfcnt::core_counters().unwrap().get("BR_INST_RETIRED.ALL_BRANCHES").unwrap();
    //println!("{:?}", counter_description);
}
