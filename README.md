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
$ sudo pip install pandas numpy ascii_graph scipy toml
```

# Stuff not documented in perf
 * PCU has umask which is supposed to be and'ed with event= attribute (from pmu-tools ucevent.py)
 * Intel Unit to perf device translation (libpfm4 source code and ucevent.py)
 * /sys/bus does not expose how many counters a device has
 * cbox to core mapping is not readable from /sys

# Deployments
  * L1-SMT: Programs are placed on one core, each gets one hyper-thread.
  * L3-SMT: Programs are placed on one socket, applications each gets one hyper-thread (i.e., cores are shared between apps).
  * L3-SMT-cores: Programs are placed on one socket, applications get a full core (i.e., hyper-threads are not shared between apps).
  * L3-cores: Programs are placed on one socket, use one core per app (the corresponding hyper-thread is left idle).
  * Full-sockets: Use the whole machine, programs allocate an entire socket.
  * Full-cores: Use the whole machine, programs use cores from all sockets interleaved (hyper-threads are left idle).
  * Full-SMT: Use the whole machine, programs use a hyper-thread from every core (interleaved, shared hyper-threads).

# TODO
 * Generate better (the no-SMT stuff is not no SMT) and whole machine deployments
Not today:
 * Integrate more apps (parsec, green marl breakpoints!, taskset cpumasks)
 * Filter out offcore stuff
 * Scale out analysis


# Related projects
 git clone git://git.code.sf.net/p/perfmon2/perfmon2 perfmon2-perfmon2
 git clone git://git.code.sf.net/p/perfmon2/libpfm4 perfmon2-libpfm4
