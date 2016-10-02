# How-to use

To profile an application use:
`$ cargo run -- profile myprogram -args1`
This will put a `results.csv` file in your default (result) output directory which can be parsed by Python.

To generate a CSV result file from intermediate source files run:
`$ cargo run -- extract <path>`

To compute pairwise placements of programs:
`$ cargo run -- profile <manifest path>`
TODO: explain manifest

# Install dependencies

Rust:
```
$ curl https://sh.rustup.rs -sSf | sh
$ cd autoperf
$ rustup override set nightly
```

Python:
```
$ sudo pip install pandas numpy ascii_graph scipy
```

# Stuff not documented in perf
 * PCU has umask which is supposed to be and'ed with event= attribute (from pmu-tools ucevent.py)
 * Intel Unit to perf device translation (libpfm4 source code and ucevent.py)
 * /sys/bus does not expose how many counters a device has
 * cbox to core mapping is not readable from /sys


# TODO
 * Write run.toml
 * Generate better (the no-SMT stuff is not no SMT) and whole machine deployments
 * Integrate more apps
 * Filter out offcore stuff
 * Scale out analysis
 * Don't re-run if a folder with results already exists

# Related projects
 git clone git://git.code.sf.net/p/perfmon2/perfmon2 perfmon2-perfmon2
 git clone git://git.code.sf.net/p/perfmon2/libpfm4 perfmon2-libpfm4
