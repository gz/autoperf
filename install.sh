#!/usr/bin/bash
git clone git@github.com:gz/autoperf.git
curl https://sh.rustup.rs -sSf | sh
rustup override set nightly
sudo pip install pandas numpy ascii_graph


