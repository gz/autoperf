# autoperf

autoperf vastly simplifies the instrumentation of programs with performance
counters on Intel machines. Rather than trying to learn how to measure every
event and manually programming event values in counter registers, you run your
program with autoperf which will repeatedly run your program until it has
measured every single performance event on your machine.

# How-to use

autoperf has a few commands, use `--help` to get a better overview of all the
options.

## Profiling

The **profile** command instruments a single program by running it multiple times
until every performance event is measured. For example,
```
$ autoperf profile echo test
```
will repeatedly run `echo test` while measuring events with performance
counters. After it is done you will find an `out` folder with many csv files
that contain measurements from individual runs.

## Aggregating results

To combine all those runs into a single CSV result file you can use the
**aggregate** command: 
```
$ autoperf aggregate ./out
``` 
This will do some sanity checking and produce a `results.csv` file including 
all repeated runs.

## Running pairwise combinations of programs

A more advanced feature is the pairwise instrumentation of programs.
Say you have a set of programs and you want to study their pairwise 
interactions with each other. You would first define a manifest like this:

```
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
```

Saving this as a file called `manifest.toml` and stored in a folder called `pairings`
you could call `autoperf` with the following arguments:
```
$ autoperf pair ./pairings
```
If you want to get a sense for what this command would be doing you can use the `-d` which just prints
the commands rather than launching any programs and doing any perf instrumentation.

### Manifest settings

The manifest format has quite a few configuration parameters. A full manifest file with
all possible configurations is shown in `./tests/pair/manifest.toml`.

* **configuration** is a list of possible mappings of the program to cores:
  * L1-SMT: Programs are placed on a single core, each gets one hyper-thread.
  * L3-SMT: Programs are placed on a single socket, applications each gets one hyper-thread interleaved (i.e., cores are shared between apps).
  * L3-SMT-cores: Programs are placed on a single socket, applications get a full core (i.e., hyper-threads are not shared between apps).
  * L3-cores: Programs are placed on a single socket, use a core per application but leave the other hyper-thread idle.
  * Full-L3: Use the whole machine, program allocated an entire L3/socket, program threads allocate an entire core (hyper-threads are left idle).
  * Full-SMT-L3: Use the whole machines, programs allocate an entire L3/socket (use hyper-threads).
  * Full-cores: Use the whole machine, programs use cores from all sockets interleaved (hyper-threads are left idle).
  * Full-SMT-cores: Use the whole machine, programs use cores from all sockets interleaved (hyper-threads are used).

# Installation

autoperf is known to work with Ubuntu 18.04 on Skylake and
IvyBridge/SandyBridge architectures. Other architectures should work too, but
may not work right out of the box. Please file a bug request. Currently we
require `perf` from the Linux project and a few other libraries that can be
installed using:

```
$ sudo apt-get install likwid cpuid hwloc numactl util-linux
```

You'll also need Rust which is best installed using rustup:
```
$ curl https://sh.rustup.rs -sSf | sh
```

autoperf is published on crates.io, so once you have rust and cargo, you can
install it like this:
```
$ cargo install autoperf
```

Or clone and build this repository:
```
$ git clone git@github.com:gz/autoperf.git
$ cd autoperf
$ cargo build --release
$ ./target/release/autoperf --help
```
