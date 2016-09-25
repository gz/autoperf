#!/usr/bin/bash

#git clone https://github.com/gz/autoperf.git

pip install --upgrade pip
sudo pip install pandas numpy ascii_graph
curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env

cd autoperf
rustup override set nightly


