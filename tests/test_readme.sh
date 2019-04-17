#!/usr/bin/bash
# We run all commands of the README.md file and hope it works
set -ex
export RUST_BACKTRACE=1
export RUST_LOG='trace'

sudo apt-get install likwid cpuid hwloc numactl util-linux

curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env

git clone https://github.com/gz/autoperf.git

cd autoperf
cargo build --release
./target/release/autoperf --help

sudo sh -c 'echo 0 >> /proc/sys/kernel/kptr_restrict'
sudo sh -c 'echo 0 > /proc/sys/kernel/nmi_watchdog'
sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'

cargo run --release -- stats stats_out
cargo run --release -- profile -d echo test

mkdir pairings
cat <<EOT >> pairings/manifest.toml
[experiment]
configurations = ["L3-SMT", "L3-SMT-cores"]

[programA]
name = "gcc"
binary = "gcc"
arguments = ["-j", "4", "out.c", "-o", "out"]

[programB]
name = "objdump"
binary = "objdump"
arguments = ["--disassemble", "/bin/true"]

[programC]
name = "cat"
binary = "cat"
arguments = ["/var/log/messages"]
env = { LC_ALL = "C" }
use_watch_repeat = true
EOT
cargo run --release -- pair -d ./pairings

