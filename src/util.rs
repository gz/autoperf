use std::fs;
use std::path::PathBuf;
use std::path::Path;

pub fn mkdir(out_dir: &Path) {
    if !out_dir.exists() {
        fs::create_dir(out_dir).expect("Can't create directory");
    }
}
