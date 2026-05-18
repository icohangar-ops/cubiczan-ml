//! SwarmFi Perps — Binary entry point.
//!
//! Run a demo swarm analysis on any market.

use swarmfi_perps::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let market = if args.len() > 1 { &args[1] } else { "BTC-USD" };

    println!("SwarmFi Perps — Swarm Analysis");
    println!("Market: {}", market);
    println!();

    // Generate mock data (offline mode)
    let data = pipeline::generate_mock_market_data(market);

    // Run full analysis
    let result = pipeline::run_swarm_analysis(market, &data, None);

    // Render report
    let report = pipeline::render_report(&result);
    println!("{}", report);
}
