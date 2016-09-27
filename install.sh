#!/usr/bin/bash

sudo pip install --upgrade pip
sudo pip install pandas numpy ascii_graph
curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly -y
source $HOME/.cargo/env
cd autoperf
rustup override set nightly
cargo run --release -- stats
cargo run --release -- extract -i out
