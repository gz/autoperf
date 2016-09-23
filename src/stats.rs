use std::io;
use std::io::prelude::*;
use std::fs;
use std::fs::File;
use std::path::Path;
use std::collections::HashMap;
use phf::Map;
use csv;

use x86::shared::perfcnt;
use x86::shared::perfcnt::intel::description::IntelPerformanceCounterDescription;

use super::util::*;

type EventMap = Map<&'static str, IntelPerformanceCounterDescription>;
type ArchitectureMap = HashMap<&'static str, (&'static str, &'static str, &'static str)>;

/// Saves the event count for all architectures to a file.
fn save_event_counts(key_to_name: &ArchitectureMap, csv_result: &Path) {
    let mut writer = csv::Writer::from_file(csv_result).unwrap();
    writer.encode(&["year", "architecture", "core events", "uncore events", "counters"]).unwrap();

    for (key, &(name, year, counters)) in key_to_name.iter() {
        let core_counters =
            perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-core", key).as_str());
        let uncore_counters =
            perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-uncore", key).as_str());


        let cc_count = core_counters.map(|c| c.len()).unwrap_or(0);
        let uc_count = uncore_counters.map(|c| c.len()).unwrap_or(0);
        writer.encode(&[year,
                      name,
                      cc_count.to_string().as_str(),
                      uc_count.to_string().as_str(),
                      counters])
            .unwrap();
    }
}

/// Given two EventMaps count all the shared (same event name key) events.
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

/// Does pairwise comparison of all architectures and saves their shared events to a file.
fn save_architecture_comparison(key_to_name: &ArchitectureMap, csv_result: &Path) {
    let mut writer = csv::Writer::from_file(csv_result).unwrap();
    writer.encode(&["name1",
                  "name2",
                  "common core events",
                  "common uncore events",
                  "name1 core events",
                  "name1 uncore events",
                  "name2 core events",
                  "name2 uncore event"])
        .unwrap();

    for (key1, &(name1, year1, _)) in key_to_name.iter() {
        for (key2, &(name2, year2, _)) in key_to_name.iter() {
            let core_counters1 =
                perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-core", key1).as_str());
            let uncore_counters1 =
                perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-uncore", key1).as_str());

            let core_counters2 =
                perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-core", key2).as_str());
            let uncore_counters2 =
                perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-uncore", key2).as_str());

            writer.encode(&[name1,
                            name2,
                            common_event_names(core_counters1, core_counters2)
                                .to_string()
                                .as_str(),
                            common_event_names(uncore_counters1, uncore_counters2)
                                .to_string()
                                .as_str(),
                            core_counters1.map(|c| c.len()).unwrap_or(0).to_string().as_str(),
                            uncore_counters1.map(|c| c.len()).unwrap_or(0).to_string().as_str(),
                            core_counters2.map(|c| c.len()).unwrap_or(0).to_string().as_str(),
                            uncore_counters2.map(|c| c.len()).unwrap_or(0).to_string().as_str()]);
        }
    }
}

/// Computes the Levenshtein edit distance of two strings.
fn edit_distance(a: &str, b: &str) -> i32 {
    let len_a = a.chars().count();
    let len_b = b.chars().count();

    let row: Vec<i32> = vec![0; len_b + 1];
    let mut matrix: Vec<Vec<i32>> = vec![row; len_a + 1];

    let chars_a: Vec<char> = a.chars().collect();
    let chars_b: Vec<char> = b.chars().collect();

    for i in 0..len_a {
        matrix[i + 1][0] = (i + 1) as i32;
    }
    for i in 0..len_b {
        matrix[0][i + 1] = (i + 1) as i32;
    }

    for i in 0..len_a {
        for j in 0..len_b {
            let ind: i32 = if chars_a[i] == chars_b[j] { 0 } else { 1 };

            let min = vec![matrix[i][j + 1] + 1, matrix[i + 1][j] + 1, matrix[i][j] + ind]
                .into_iter()
                .min()
                .unwrap();

            matrix[i + 1][j + 1] = if min == 0 { 0 } else { min };
        }
    }
    matrix[len_a][len_b]
}

/// Computes the edit distance of the event description for common events shared in 'a' and 'b'.
fn common_event_desc_distance(writer: &mut csv::Writer<File>,
                              a: Option<&'static EventMap>,
                              b: Option<&'static EventMap>,
                              uncore: bool)
                              -> csv::Result<()> {
    if a.is_none() || b.is_none() {
        return Ok(());
    }

    let a_map = a.unwrap();
    let b_map = b.unwrap();

    for (key1, value1) in a_map.entries() {
        match b_map.get(key1) {
            Some(value2) => {
                assert_eq!(value1.event_name, value2.event_name);
                let ed = edit_distance(value1.brief_description, value2.brief_description)
                    .to_string();
                let uncore_str = if uncore { "true" } else { "false" };

                try!(writer.encode(&[value1.event_name,
                                     ed.as_str(),
                                     uncore_str,
                                     value1.brief_description,
                                     value2.brief_description]))
            }
            None => {
                // Ignore event names that are not shared in both architectures
            }
        }
    }

    Ok(())
}

/// Does a pairwise comparison of all architectures by computing edit distances of shared events.
fn save_edit_distances(key_to_name: &ArchitectureMap, output_dir: &Path) {

    for (key1, &(name1, year1, _)) in key_to_name.iter() {
        for (key2, &(name2, year2, _)) in key_to_name.iter() {

            let mut csv_result = output_dir.to_path_buf();
            csv_result.push(format!("editdist_{}-vs-{}.csv", name1, name2));

            let mut writer = csv::Writer::from_file(csv_result).unwrap();
            writer.encode(&["event name", "edit distance", "uncore", "desc1", "desc2"])
                .unwrap();

            let core_counters1 =
                perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-core", key1).as_str());
            let uncore_counters1 =
                perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-uncore", key1).as_str());

            let core_counters2 =
                perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-core", key2).as_str());
            let uncore_counters2 =
                perfcnt::intel::counters::COUNTER_MAP.get(format!("{}-uncore", key2).as_str());

            common_event_desc_distance(&mut writer, core_counters1, core_counters2, false);
            common_event_desc_distance(&mut writer, uncore_counters1, uncore_counters2, true);
        }
    }
}

/// Generate all the stats about Intel events and save them to a file.
pub fn stats(output_path: &Path) {
    mkdir(output_path);

    let mut key_to_name = HashMap::new();
    key_to_name.insert("GenuineIntel-6-1C", ("Bonnell", "2008", "4"));
    key_to_name.insert("GenuineIntel-6-1E", ("NehalemEP", "2009", "4"));
    key_to_name.insert("GenuineIntel-6-2E", ("NehalemEX", "2010", "4"));
    key_to_name.insert("GenuineIntel-6-2F", ("WestmereEX", "2011", "4"));
    key_to_name.insert("GenuineIntel-6-25", ("WestmereEP-SP", "2010", "4"));
    key_to_name.insert("GenuineIntel-6-2C", ("WestmereEP-DP", "2010", "4"));
    key_to_name.insert("GenuineIntel-6-37", ("Silvermont", "2013", "8"));
    key_to_name.insert("GenuineIntel-6-5C", ("Goldmont", "2016", "8"));
    key_to_name.insert("GenuineIntel-6-2A", ("SandyBridge", "2011", "8"));
    key_to_name.insert("GenuineIntel-6-2D", ("Jaketown", "2011", "8"));
    key_to_name.insert("GenuineIntel-6-3A", ("IvyBridge", "2012", "8"));
    key_to_name.insert("GenuineIntel-6-3E", ("IvyBridgeEP", "2014", "8"));
    key_to_name.insert("GenuineIntel-6-3C", ("Haswell", "2013", "8"));
    key_to_name.insert("GenuineIntel-6-3F", ("HaswellX", "2014", "8"));
    key_to_name.insert("GenuineIntel-6-3D", ("Broadwell", "2014", "8"));
    key_to_name.insert("GenuineIntel-6-4F", ("BroadwellX", "2016", "8"));
    key_to_name.insert("GenuineIntel-6-56", ("BroadwellDE", "2015", "8"));
    key_to_name.insert("GenuineIntel-6-4E", ("Skylake", "2015", "8"));
    key_to_name.insert("GenuineIntel-6-57", ("KnightsLanding", "2016", "4"));

    let mut csv_result_file = output_path.to_path_buf();
    csv_result_file.push("events.csv");
    save_event_counts(&key_to_name, csv_result_file.as_path());

    let mut csv_result_file = output_path.to_path_buf();
    csv_result_file.push("architecture_comparison.csv");
    save_architecture_comparison(&key_to_name, csv_result_file.as_path());

    save_edit_distances(&key_to_name, output_path);
}
