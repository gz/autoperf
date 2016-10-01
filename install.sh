#!/usr/bin/bash

sudo pip install --upgrade pip
sudo pip install pandas numpy ascii_graph
curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly -y
source $HOME/.cargo/env
cd autoperf
rustup override set nightly
cargo run --release -- stats
cargo run --release -- extract -i out


route add default gw 10.110.4.4 eno1

route add default gw 10.110.4.97 eno1

Thhis worked:
Kernel IP routing table
Destination     Gateway         Genmask         Flags Metric Ref    Use Iface
default         10.110.4.1      0.0.0.0         UG    0      0        0 ens2f0
default         *               0.0.0.0         U     1003   0        0 eno2
default         *               0.0.0.0         U     1005   0        0 eno3
default         *               0.0.0.0         U     1007   0        0 eno4
default         *               0.0.0.0         U     1009   0        0 ens2f1
10.110.4.0      *               255.255.252.0   U     0      0        0 eno1
10.110.4.0      *               255.255.252.0   U     0      0        0 ens2f0
link-local      *               255.255.0.0     U     0      0        0 eno4
link-local      *               255.255.0.0     U     0      0        0 eno3
link-local      *               255.255.0.0     U     0      0        0 eno2
link-local      *               255.255.0.0     U     0      0        0 ens2f1

ubuntu@ubuntu:~$ ifconfig ens2f0
ens2f0    Link encap:Ethernet  HWaddr 00:1e:67:9f:45:06
          inet addr:10.110.5.12  Bcast:10.110.7.255  Mask:255.255.252.0
          inet6 addr: fe80::21e:67ff:fe9f:4506/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:1149761 errors:0 dropped:0 overruns:0 frame:0
          TX packets:209234 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:1675672387 (1.6 GB)  TX bytes:14403294 (14.4 MB)
