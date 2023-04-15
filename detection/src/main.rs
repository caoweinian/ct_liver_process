use clap::Parser;
use detection::Cli;

fn main() {
    let mut cmd: Cli = Cli::parse();
    cmd.run_program();
}
