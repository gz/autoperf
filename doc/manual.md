# autoperf

User manual, currently still under construction.

## profile -- measure all the things

## aggregate -- combine results

## stats -- generate some stats about all events

## search -- finding undocumented events

## pair -- profiling pairwise combinations of programs

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

After saving this as a file called `manifest.toml` in a folder called
`pairings` you could call `autoperf` with the following arguments:

```
$ autoperf pair ./pairings
```

This essentially does what the profile command does, but for every individual
program defined in the manifest. In addition, it does even more profile
commands for programA while continously running programB or programC in the
background (once this is done it does the same for programB and programC).

If this is confusing and you want to get first hand experience of what we would
really be running here you can also pass the `-d` argument to the pair
sub-command. In this case, autoperf just prints a plan of what it would be
doing, rather than launching any programs.

### Manifest settings

The manifest format has a few configuration parameters. A full manifest file with
all possible configurations and documentation in the comments is shown in 
`./tests/pair/manifest.toml`.

* **configuration** is a list of possible mappings of the program to cores:
  * L1-SMT: Programs are placed on a single core, each gets one hyper-thread.
  * L3-SMT: Programs are placed on a single socket, applications each gets one hyper-thread interleaved (i.e., cores are shared between apps).
  * L3-SMT-cores: Programs are placed on a single socket, applications get a full core (i.e., hyper-threads are not shared between apps).
  * L3-cores: Programs are placed on a single socket, use a core per application but leave the other hyper-thread idle.
  * Full-L3: Use the whole machine, program allocated an entire L3/socket, program threads allocate an entire core (hyper-threads are left idle).
  * Full-SMT-L3: Use the whole machines, programs allocate an entire L3/socket (use hyper-threads).
  * Full-cores: Use the whole machine, programs use cores from all sockets interleaved (hyper-threads are left idle).
  * Full-SMT-cores: Use the whole machine, programs use cores from all sockets interleaved (hyper-threads are used).
