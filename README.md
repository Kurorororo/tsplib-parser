# Parser for the TSPLIB format in Rust

[![crates.io](https://img.shields.io/crates/v/tsplib-parser)](https://crates.io/crates/tsplib-parser)
[![minimum rustc 1.76](https://img.shields.io/badge/rustc-1.76+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Warning: This crate is work in progress and is not comprehensively tested!**

In addition to [the specifications by TSPLIB95](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf), the following features are supported:

- Maximum distance (`DISTANCE`) and service time (`SERVICE_TIME`) for some instances in [CVRPLIB](http://vrp.galgos.inf.puc-rio.br/index.php/en/).
- Multi-dimensional demands (`DEMAND_DIMENSION`) for [multi-commodity pickup-and-delivery TSP (m-PDTSP)](https://hhperez.webs.ull.es/PDsite/).

## Examples

```rust
use std::env;
use tsplib_parser::Instance;

fn main() {
    let mut args = env::args();
    args.next().unwrap();
    let input = args.next().unwrap();

    println!("Loading instance from file: {}", input);
    let instance = Instance::load(&input).unwrap();
    println!("{:?}", instance);

    let matrix = instance.get_full_distance_matrix().unwrap();
    println!("{:?}", matrix);
}
```
