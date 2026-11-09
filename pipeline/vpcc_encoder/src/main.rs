mod cli;
mod debug;
mod logging;
mod metrics;
mod pipeline;
mod report;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = cli::parse_args()?;
    pipeline::run(config)
}
