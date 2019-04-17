# autoperf

autoperf vastly simplifies the instrumentation of programs with performance
counters on Intel machines. Rather than trying to learn how to measure every
event and manually programming event values in counter registers or perf, you
can use autoperf which will repeatedly run your program until it has measured
every single performance event on your machine. autoperf tries to compute a
near-optimal schedule that maximizes the amount of events measured per run,
minimizes the total number of runs and avoids multiplexing of events on
counters.

# Installation

autoperf is known to work with Ubuntu 18.04 on Skylake and
IvyBridge/SandyBridge architectures. All Intel architectures should work,
please file a bug request if they don't. autoperf builds on `perf` from the
Linux project and a few other libraries that can be installed using:

```
$ sudo apt-get install likwid cpuid hwloc numactl util-linux
```

To run the example analysis scripts, you'll need these python3 libraries:
```
$ pip3 install ascii_graph matplotlib pandas argparse numpy
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

Or clone and build the repository yourself:
```
$ git clone https://github.com/gz/autoperf.git
$ cd autoperf
$ cargo build --release
$ ./target/release/autoperf --help
```

autoperf uses perf internally to interface with Linux and the performance
counter hardware. perf recommends that the following settings are disabled.
Therefore, autoperf will check the values of those configurations and refuse to
start if they are not set like below:
```
sudo sh -c 'echo 0 >> /proc/sys/kernel/kptr_restrict'
sudo sh -c 'echo 0 > /proc/sys/kernel/nmi_watchdog'
sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'
```

# Usage

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
This will do some sanity checking and produce a `results.csv` file which contains 
all the measured data.

## Analyze results

Now you have all the data, so you can start asking some questions. As an
example the following script tells you which events were correlated
when your program was running:

```
$ python3 analyze/profile/correlation.py ./out
$ open out/correlation_heatmap.png
```

For example, visualizing event correlation for the profiled `sleep 2` command
above looks like this (every dot represents the correlation between two
measured performance events, this was done on a Skylake machine which had
around 1.7k non-zero event measurement):
![Correlation Heatmap](/doc/correlation_heatmap.png)

You can look at individual events too:
```
python3 analyze/profile/event_detail.py --resultdir ./out --features AVG.OFFCORE_RESPONSE.ALL_RFO.L3_MISS.REMOTE_HIT_FORWARD
```
![Plot events](/doc/perf_event_plot.png)

## What do I use this for?

autoperf allows you to quickly gather lots of performance (training) data and
reason about it quantitaively. For example, we initially developed autoperf to
build ML classifiers that the Barrelfish scheduler could use for detecting
application slowdown and make better scheduling decisions. autoperf meant that
we needed no domain knowledge about events, aside from how to measure them they
could be treated as a black-box.

You can read more about our experiments here:

* https://dl.acm.org/citation.cfm?id=2967360.2967375 
* https://www.research-collection.ethz.ch/handle/20.500.11850/155854

Of course autoperf can greatly simplify your life in many other scenarios too:
 * Find out what performance events are relevant for your workload
 * Analyzing and finding performance issues in your code
 * Find the best classifiers to detect hardware exploits (side channels/Spectre/Meltdown etc.)
 * ...
