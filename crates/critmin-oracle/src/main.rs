//! CritMin Oracle CLI — Rust binary entry point.
//!
//! Usage:
//!   critmin-oracle demo     — Run demo pipeline with mock data
//!   critmin-oracle live     — Run live pipeline with real APIs (requires feature "live")

use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let banner = r#"
    ╔══════════════════════════════════════════════════════╗
    ║   ███╗   ███╗ █████╗ ███████╗████████╗██╗  ██╗      ║
    ║   ████╗ ████║██╔══██╗██╔════╝╚══██╔══╝██║  ██║      ║
    ║   ██╔████╔██║███████║███████╗   ██║   ███████║      ║
    ║   ██║╚██╔╝██║██╔══██║╚════██║   ██║   ██╔══██║      ║
    ║   ██║ ╚═╝ ██║██║  ██║███████║   ██║   ██║  ██║      ║
    ║   ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝      ║
    ║        On-Chain Oracle for Critical Minerals          ║
    ╚══════════════════════════════════════════════════════╝
    "#;

    println!("{}", banner);

    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("demo");
    match mode {
        "demo" => {
            println!("  Mode: DEMO (Mock Data)");
            println!("  Time: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
            let output = critmin_oracle_lib::run_demo_pipeline();
            critmin_oracle_lib::print_pipeline_results(&output);

            // Save results to JSON
            let json = serde_json::to_string_pretty(&output).unwrap();
            let filename = format!(
                "results-{}.json",
                chrono::Utc::now().format("%Y%m%d-%H%M%S")
            );
            let mut file = std::fs::File::create(&filename).expect("Failed to create output file");
            file.write_all(json.as_bytes()).unwrap();
            println!("\n  Results saved to: {}", filename);
        }
        "live" => {
            #[cfg(feature = "live")]
            {
                println!("  Mode: LIVE (Real APIs)");
                println!("  Live mode requires FRED_API_KEY and ALPHA_VANTAGE_KEY env vars.");
                println!("  Compile with: cargo run --features live -- live");
            }
            #[cfg(not(feature = "live"))]
            {
                println!("  ERROR: Live mode requires the 'live' feature.");
                println!("  Compile with: cargo run --features live -- live");
                std::process::exit(1);
            }
        }
        "--help" | "-h" | "help" => {
            println!("  Usage:");
            println!("    critmin-oracle demo     Run demo pipeline with mock data");
            println!("    critmin-oracle live     Run live pipeline with real API data");
            println!("    critmin-oracle help     Show this help message");
        }
        _ => {
            eprintln!("  Unknown mode: '{}'. Use 'demo' or 'live'.", mode);
            std::process::exit(1);
        }
    }

    println!("\n  Pipeline complete!");
}
