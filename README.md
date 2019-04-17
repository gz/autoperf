# autoperf

autoperf vastly simplifies the instrumentation of programs with performance
counters on Intel machines. Rather than trying to learn how to measure every
event and manually programming event values in counter registers, you can use
autoperf which will repeatedly run your program until it has
measured every single performance event on your machine.

# Installation

autoperf is known to work with Ubuntu 18.04 on Skylake and
IvyBridge/SandyBridge architectures. Other architectures should work too, but
may not work right out of the box. Please file a bug request. Currently we
require `perf` from the Linux project and a few other libraries that can be
installed using:

```
$ sudo apt-get install likwid cpuid hwloc numactl util-linux
```

To run some sample analysis scripts, you'll need these python3 libraries:
```
$ pip3 install ascii_graph matplotlib pandas
```

You'll also need Rust which is best installed using rustup:
```
$ curl https://sh.rustup.rs -sSf | sh -s -- -y
$ source $HOME/.cargo/env
```

autoperf is published on crates.io, so once you have rust and cargo, you can
install it like this:
```
$ cargo install autoperf
```

Or clone and build this repository:
```
$ git clone https://github.com/gz/autoperf.git
$ cd autoperf
$ cargo build --release
$ ./target/release/autoperf --help
```

autoperf uses perf internally to interface with Linux and the performance
counter hardware. perf recommends that the following settings are disabled.
Therefore, autoperf will check the values of those configurations and refuse to
start if they are not set to the values below:
```
sudo sh -c 'echo 0 >> /proc/sys/kernel/kptr_restrict'
sudo sh -c 'echo 0 > /proc/sys/kernel/nmi_watchdog'
sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'
```

# Quick and simple how-to

autoperf has a few commands, use `--help` to get a better overview of all the
options.

## Profiling

The **profile** command instruments a single program by running it multiple times
until every performance event is measured. For example,
```
$ autoperf profile sleep 2
```
will repeatedly run `sleep 2` while measuring events with performance
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

## Analyzing results

Now you have all the data, so you can start asking some questions. As an
example the following script tells you which events were correlated
when your program was running:

```
$ python3 analyze/profile/correlation.py ./out
$ open out/heatmap.png
```

For example, the output from the `sleep 2` command above looks like this (every dot represents the correlation between two measured performance events):
![Correlation Heatmap](/doc/correlation_heatmap.png)
