# How-to use

To profile an application use:
`$ cargo run -- profile myprogram -args1`

To generate a CSV result file from intermediate source files run:
`$ cargo run -- extract`

This will put a `results.csv` file in your default (result) output directory which can be parsed by Python.



# Install dependencies

Rust:
```
$ curl https://sh.rustup.rs -sSf | sh
$ cd autoperf
$ rustup override set nightly
```


Python:
```
$ sudo pip install pandas numpy
```
