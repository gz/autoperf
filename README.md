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

Dependencies:
```
$ sudo apt-get install likwid cpuid hwloc numactl util-linux
```

Rust:
```
$ curl https://sh.rustup.rs -sSf | sh
$ cd autoperf
$ rustup override set nightly
```

Python:
```
$ sudo pip install pandas numpy ascii_graph scipy toml pydotplus sklearn tabulate
```

# Workflow

python scripts/pair/runtimes.py
python scripts/pair/extract_all.py
python scripts/pair/matrix_all.py

# Stuff not documented in perf
  * PCU has umask which is supposed to be and'ed with event= attribute (from pmu-tools ucevent.py)
  * Intel Unit to perf device translation (libpfm4 source code and ucevent.py)
  * /sys/bus does not expose how many counters a device has
  * cbox to core mapping is not readable from /sys

# Deployments
  * L1-SMT: Programs are placed on a single core, each gets one hyper-thread.
  * L3-SMT: Programs are placed on a single socket, applications each gets one hyper-thread interleaved (i.e., cores are shared between apps).
  * L3-SMT-cores: Programs are placed on a single socket, applications get a full core (i.e., hyper-threads are not shared between apps).
  * L3-cores: Programs are placed on a single socket, use a core per application but leave the other hyper-thread idle.
  * Full-L3: Use the whole machine, program allocated an entire L3/socket, program threads allocate an entire core (hyper-threads are left idle).
  * Full-SMT-L3: Use the whole machines, programs allocate an entire L3/socket (use hyper-threads).
  * Full-cores: Use the whole machine, programs use cores from all sockets interleaved (hyper-threads are left idle).
  * Full-SMT-cores: Use the whole machine, programs use cores from all sockets interleaved (hyper-threads are used).

# TODO
  * Create one big SVM matrix, evaluate with one program removed [uncore: shared, exclusive, all]
  * Put plots, SVM accuracty in the paper
  * Add figure to paper [events]
  * Add eval program descriptions / data size to paper
  * Add machine description to paper
  * put heatmaps them in the paper
Not today:
  * Scale out analysis
  * Wait a bit before A starts (to warm-up b)
  - TODO: Should probably have return codes in perf.csv and check them as well!


# Related projects
  * git clone git://git.code.sf.net/p/perfmon2/perfmon2 perfmon2-perfmon2
  * git clone git://git.code.sf.net/p/perfmon2/libpfm4 perfmon2-libpfm4
