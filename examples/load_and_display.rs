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
