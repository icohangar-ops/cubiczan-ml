//! `cz-serve` — CLI binary for the Superserve.ai persistent sandbox API.
//!
//! Provides human-friendly command-line access to sandbox and template
//! management, command execution, and convenience operations.
//!
//! # Usage
//!
//! ```text
//! cz-serve <command> [options]
//!
//! Commands:
//!   sandbox create [--name NAME] [--template TEMPLATE_ID] [--timeout SECS] [--metadata KEY=VAL]...
//!   sandbox list [--metadata KEY=VAL]...
//!   sandbox get <SANDBOX_ID>
//!   sandbox pause <SANDBOX_ID>
//!   sandbox resume <SANDBOX_ID>
//!   sandbox delete <SANDBOX_ID>
//!   sandbox exec <SANDBOX_ID> <COMMAND> [--dir DIR] [--timeout SECS]
//!
//!   template create --name NAME [--base-image IMAGE] [--cpu MILLIS] [--memory MB] [--disk MB]
//!   template list
//!   template get <TEMPLATE_ID>
//!   template rebuild <TEMPLATE_ID>
//!
//!   health
//!
//!   run <TEMPLATE_ID> <COMMAND> [--name NAME] [--timeout SECS]
//! ```

use std::collections::HashMap;
use std::env;
use std::process;

use cubiczan_superserve::{
    CreateSandboxRequest, CreateTemplateRequest, ExecRequest, SuperserveClient,
    TemplateResources,
};

// ---------------------------------------------------------------------------
// Arg parsing helpers
// ---------------------------------------------------------------------------

/// Parse `--metadata KEY=VAL` pairs from the args iterator.
/// Consumes and returns arguments that start with `--metadata`.
fn parse_metadata_args(args: &[String], start_idx: usize) -> (HashMap<String, String>, usize) {
    let mut metadata = HashMap::new();
    let mut i = start_idx;
    while i < args.len() && args[i] == "--metadata" {
        if i + 1 < args.len() {
            let kv = &args[i + 1];
            if let Some(eq_pos) = kv.find('=') {
                let key = kv[..eq_pos].to_string();
                let val = kv[eq_pos + 1..].to_string();
                metadata.insert(key, val);
            }
            i += 2;
        } else {
            // Missing value for --metadata
            eprintln!("error: --metadata requires a KEY=VAL argument");
            process::exit(2);
        }
    }
    (metadata, i)
}

/// Parse a `--flag VALUE` pair, returning (Some(value), new_index) or (None, old_index).
fn parse_flag_value(args: &[String], idx: usize, flag: &str) -> (Option<String>, usize) {
    if idx < args.len() && args[idx] == flag {
        if idx + 1 < args.len() {
            (Some(args[idx + 1].clone()), idx + 2)
        } else {
            eprintln!("error: {} requires a value", flag);
            process::exit(2);
        }
    } else {
        (None, idx)
    }
}

/// Parse a `--flag VALUE` pair as u64.
fn parse_flag_u64(args: &[String], idx: usize, flag: &str) -> (Option<u64>, usize) {
    if idx < args.len() && args[idx] == flag {
        if idx + 1 < args.len() {
            match args[idx + 1].parse::<u64>() {
                Ok(v) => return (Some(v), idx + 2),
                Err(_) => {
                    eprintln!("error: {} must be a number, got '{}'", flag, args[idx + 1]);
                    process::exit(2);
                }
            }
        } else {
            eprintln!("error: {} requires a value", flag);
            process::exit(2);
        }
    }
    (None, idx)
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

fn cmd_sandbox_create(client: &SuperserveClient, args: &[String]) {
    let mut name = String::new();
    let mut template_id: Option<String> = None;
    let mut timeout: Option<u64> = None;
    let mut metadata = HashMap::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--name" => {
                if i + 1 >= args.len() {
                    eprintln!("error: --name requires a value");
                    process::exit(2);
                }
                name = args[i + 1].clone();
                i += 2;
            }
            "--template" => {
                let (val, new_i) = parse_flag_value(args, i, "--template");
                template_id = val;
                i = new_i;
            }
            "--timeout" => {
                let (val, new_i) = parse_flag_u64(args, i, "--timeout");
                timeout = val;
                i = new_i;
            }
            "--metadata" => {
                let (meta, new_i) = parse_metadata_args(args, i);
                metadata = meta;
                i = new_i;
            }
            other => {
                eprintln!("error: unexpected argument '{}' for 'sandbox create'", other);
                process::exit(2);
            }
        }
    }

    if name.is_empty() {
        eprintln!("error: --name is required for 'sandbox create'");
        process::exit(2);
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut req = CreateSandboxRequest::new(&name);
    if let Some(tid) = template_id {
        req = req.template_id(tid);
    }
    if let Some(secs) = timeout {
        req = req.timeout_seconds(secs);
    }
    if !metadata.is_empty() {
        req = req.metadata(metadata);
    }

    match rt.block_on(client.create_sandbox(&req)) {
        Ok(sandbox) => {
            println!("Sandbox created successfully");
            println!("  ID:     {}", sandbox.id);
            println!("  Name:   {}", sandbox.name);
            println!("  Status: {}", sandbox.status);
            if let Some(ref tmpl) = sandbox.template_id {
                println!("  Template: {}", tmpl);
            }
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_sandbox_list(client: &SuperserveClient, args: &[String]) {
    let mut i = 0;
    let mut metadata = HashMap::new();
    while i < args.len() {
        if args[i] == "--metadata" {
            let (meta, new_i) = parse_metadata_args(args, i);
            metadata = meta;
            i = new_i;
        } else {
            eprintln!("error: unexpected argument '{}' for 'sandbox list'", args[i]);
            process::exit(2);
        }
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(client.list_sandboxes(&metadata)) {
        Ok(sandboxes) => {
            if sandboxes.is_empty() {
                println!("No sandboxes found.");
                return;
            }
            println!("{:<38} {:<20} {:<10} {}", "ID", "NAME", "STATUS", "CREATED");
            println!("{}", "-".repeat(90));
            for sb in &sandboxes {
                println!(
                    "{:<38} {:<20} {:<10} {}",
                    sb.id, sb.name, sb.status, sb.created_at
                );
            }
            println!("\n{} sandbox(es)", sandboxes.len());
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_sandbox_get(client: &SuperserveClient, args: &[String]) {
    if args.is_empty() {
        eprintln!("error: sandbox get requires <SANDBOX_ID>");
        process::exit(2);
    }
    let sandbox_id = &args[0];

    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(client.get_sandbox(sandbox_id)) {
        Ok(sb) => {
            println!("ID:     {}", sb.id);
            println!("Name:   {}", sb.name);
            println!("Status: {}", sb.status);
            println!("Created: {}", sb.created_at);
            if let Some(ref tmpl) = sb.template_id {
                println!("Template: {}", tmpl);
            }
            if let Some(ref token) = sb.access_token {
                println!("Access Token: {}", token);
            }
            if let Some(secs) = sb.timeout_seconds {
                println!("Timeout: {}s", secs);
            }
            if !sb.metadata.is_empty() {
                println!("Metadata:");
                for (k, v) in &sb.metadata {
                    println!("  {} = {}", k, v);
                }
            }
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_sandbox_pause(client: &SuperserveClient, args: &[String]) {
    if args.is_empty() {
        eprintln!("error: sandbox pause requires <SANDBOX_ID>");
        process::exit(2);
    }
    let sandbox_id = &args[0];

    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(client.pause_sandbox(sandbox_id)) {
        Ok(sb) => {
            println!("Sandbox {} paused", sb.id);
            println!("  Status: {}", sb.status);
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_sandbox_resume(client: &SuperserveClient, args: &[String]) {
    if args.is_empty() {
        eprintln!("error: sandbox resume requires <SANDBOX_ID>");
        process::exit(2);
    }
    let sandbox_id = &args[0];

    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(client.resume_sandbox(sandbox_id)) {
        Ok(sb) => {
            println!("Sandbox {} resumed", sb.id);
            println!("  Status: {}", sb.status);
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_sandbox_delete(client: &SuperserveClient, args: &[String]) {
    if args.is_empty() {
        eprintln!("error: sandbox delete requires <SANDBOX_ID>");
        process::exit(2);
    }
    let sandbox_id = &args[0];

    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(client.delete_sandbox(sandbox_id)) {
        Ok(sb) => {
            println!("Sandbox {} deleted", sb.id);
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_sandbox_exec(client: &SuperserveClient, args: &[String]) {
    if args.len() < 2 {
        eprintln!("error: sandbox exec requires <SANDBOX_ID> <COMMAND>");
        process::exit(2);
    }
    let sandbox_id = &args[0];
    let command = &args[1];

    let mut working_dir: Option<String> = None;
    let mut timeout: Option<u64> = None;
    let mut i = 2;

    while i < args.len() {
        match args[i].as_str() {
            "--dir" => {
                let (val, new_i) = parse_flag_value(args, i, "--dir");
                working_dir = val;
                i = new_i;
            }
            "--timeout" => {
                let (val, new_i) = parse_flag_u64(args, i, "--timeout");
                timeout = val;
                i = new_i;
            }
            other => {
                eprintln!("error: unexpected argument '{}' for 'sandbox exec'", other);
                process::exit(2);
            }
        }
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut req = ExecRequest::new(command);
    if let Some(dir) = working_dir {
        req = req.working_dir(dir);
    }
    if let Some(secs) = timeout {
        req = req.timeout_s(secs);
    }

    match rt.block_on(client.exec(sandbox_id, &req)) {
        Ok(result) => {
            if !result.stdout.is_empty() {
                println!("{}", result.stdout.trim_end());
            }
            if !result.stderr.is_empty() {
                eprintln!("{}", result.stderr.trim_end());
            }
            println!("[exit code: {}]", result.exit_code);
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_template_create(client: &SuperserveClient, args: &[String]) {
    let mut name = String::new();
    let mut base_image: Option<String> = None;
    let mut cpu: Option<u64> = None;
    let mut memory: Option<u64> = None;
    let mut disk: Option<u64> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--name" => {
                if i + 1 >= args.len() {
                    eprintln!("error: --name requires a value");
                    process::exit(2);
                }
                name = args[i + 1].clone();
                i += 2;
            }
            "--base-image" => {
                let (val, new_i) = parse_flag_value(args, i, "--base-image");
                base_image = val;
                i = new_i;
            }
            "--cpu" => {
                let (val, new_i) = parse_flag_u64(args, i, "--cpu");
                cpu = val;
                i = new_i;
            }
            "--memory" => {
                let (val, new_i) = parse_flag_u64(args, i, "--memory");
                memory = val;
                i = new_i;
            }
            "--disk" => {
                let (val, new_i) = parse_flag_u64(args, i, "--disk");
                disk = val;
                i = new_i;
            }
            other => {
                eprintln!("error: unexpected argument '{}' for 'template create'", other);
                process::exit(2);
            }
        }
    }

    if name.is_empty() {
        eprintln!("error: --name is required for 'template create'");
        process::exit(2);
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut req = CreateTemplateRequest::new(&name);
    if let Some(img) = base_image {
        req = req.base_image(img);
    }
    if cpu.is_some() || memory.is_some() || disk.is_some() {
        let resources = TemplateResources::new(
            cpu.unwrap_or(2000) as u32,
            memory.unwrap_or(2048) as u32,
            disk.unwrap_or(4096) as u32,
        );
        req = req.resources(resources);
    }

    match rt.block_on(client.create_template(&req)) {
        Ok(template) => {
            println!("Template created successfully");
            println!("  ID:     {}", template.id);
            println!("  Name:   {}", template.name);
            println!("  Status: {}", template.status);
            println!("  Image:  {}", template.base_image);
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_template_list(client: &SuperserveClient, _args: &[String]) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(client.list_templates()) {
        Ok(templates) => {
            if templates.is_empty() {
                println!("No templates found.");
                return;
            }
            println!("{:<38} {:<20} {:<10} {}", "ID", "NAME", "STATUS", "CREATED");
            println!("{}", "-".repeat(90));
            for tmpl in &templates {
                println!(
                    "{:<38} {:<20} {:<10} {}",
                    tmpl.id, tmpl.name, tmpl.status, tmpl.created_at
                );
            }
            println!("\n{} template(s)", templates.len());
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_template_get(client: &SuperserveClient, args: &[String]) {
    if args.is_empty() {
        eprintln!("error: template get requires <TEMPLATE_ID>");
        process::exit(2);
    }
    let template_id = &args[0];

    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(client.get_template(template_id)) {
        Ok(tmpl) => {
            println!("ID:     {}", tmpl.id);
            println!("Name:   {}", tmpl.name);
            println!("Status: {}", tmpl.status);
            println!("Image:  {}", tmpl.base_image);
            println!("Created: {}", tmpl.created_at);
            println!(
                "Resources: cpu={}ms, memory={}MB, disk={}MB",
                tmpl.resources.cpu_millis, tmpl.resources.memory_mb, tmpl.resources.disk_mb
            );
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_template_rebuild(client: &SuperserveClient, args: &[String]) {
    if args.is_empty() {
        eprintln!("error: template rebuild requires <TEMPLATE_ID>");
        process::exit(2);
    }
    let template_id = &args[0];

    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(client.rebuild_template(template_id)) {
        Ok(build) => {
            println!("Template rebuild triggered");
            println!("  Build ID: {}", build.id);
            println!("  Status:   {}", build.status);
            println!("  Created:  {}", build.created_at);
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_health(client: &SuperserveClient) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(client.health()) {
        Ok(health) => {
            if health.ok {
                println!("API is healthy");
            } else {
                println!("API is unhealthy");
                process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_run(client: &SuperserveClient, args: &[String]) {
    if args.len() < 2 {
        eprintln!("error: run requires <TEMPLATE_ID> <COMMAND>");
        process::exit(2);
    }
    let template_id = &args[0];
    let command = &args[1];

    let mut name: Option<String> = None;
    let mut timeout: Option<u64> = None;
    let mut i = 2;

    while i < args.len() {
        match args[i].as_str() {
            "--name" => {
                let (val, new_i) = parse_flag_value(args, i, "--name");
                name = val;
                i = new_i;
            }
            "--timeout" => {
                let (val, new_i) = parse_flag_u64(args, i, "--timeout");
                timeout = val;
                i = new_i;
            }
            other => {
                eprintln!("error: unexpected argument '{}' for 'run'", other);
                process::exit(2);
            }
        }
    }

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Create sandbox
    let sandbox_name = name.unwrap_or_else(|| {
        format!("run-{}", chrono::Utc::now().timestamp())
    });

    let mut create_req = CreateSandboxRequest::new(&sandbox_name)
        .template_id(template_id);
    if let Some(secs) = timeout {
        create_req = create_req.timeout_seconds(secs);
    }

    let sandbox = match rt.block_on(client.create_sandbox(&create_req)) {
        Ok(sb) => sb,
        Err(e) => {
            eprintln!("error: failed to create sandbox: {}", e);
            process::exit(1);
        }
    };

    println!("Created sandbox {} ({})", sandbox.id, sandbox.name);

    // Execute command
    let mut exec_req = ExecRequest::new(command);
    if let Some(secs) = timeout {
        exec_req = exec_req.timeout_s(secs);
    }

    let result = match rt.block_on(client.exec(&sandbox.id, &exec_req)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: execution failed: {}", e);
            // Try to clean up
            let _ = rt.block_on(client.delete_sandbox(&sandbox.id));
            process::exit(1);
        }
    };

    // Print result
    if !result.stdout.is_empty() {
        println!("{}", result.stdout.trim_end());
    }
    if !result.stderr.is_empty() {
        eprintln!("{}", result.stderr.trim_end());
    }
    println!("[exit code: {}]", result.exit_code);

    // Delete sandbox (best-effort)
    match rt.block_on(client.delete_sandbox(&sandbox.id)) {
        Ok(_) => println!("Deleted sandbox {}", sandbox.id),
        Err(e) => eprintln!("warning: failed to delete sandbox {}: {}", sandbox.id, e),
    }

    // Exit with the command's exit code
    if result.exit_code != 0 {
        process::exit(result.exit_code);
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

fn print_usage() -> ! {
    eprintln!("cz-serve — CLI for Superserve.ai persistent sandbox API");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  cz-serve <command> [options]");
    eprintln!();
    eprintln!("COMMANDS:");
    eprintln!("  sandbox create [--name NAME] [--template TEMPLATE_ID] [--timeout SECS] [--metadata KEY=VAL]...");
    eprintln!("  sandbox list [--metadata KEY=VAL]...");
    eprintln!("  sandbox get <SANDBOX_ID>");
    eprintln!("  sandbox pause <SANDBOX_ID>");
    eprintln!("  sandbox resume <SANDBOX_ID>");
    eprintln!("  sandbox delete <SANDBOX_ID>");
    eprintln!("  sandbox exec <SANDBOX_ID> <COMMAND> [--dir DIR] [--timeout SECS]");
    eprintln!();
    eprintln!("  template create --name NAME [--base-image IMAGE] [--cpu MILLIS] [--memory MB] [--disk MB]");
    eprintln!("  template list");
    eprintln!("  template get <TEMPLATE_ID>");
    eprintln!("  template rebuild <TEMPLATE_ID>");
    eprintln!();
    eprintln!("  health");
    eprintln!();
    eprintln!("  run <TEMPLATE_ID> <COMMAND> [--name NAME] [--timeout SECS]");
    eprintln!();
    eprintln!("ENVIRONMENT:");
    eprintln!("  SUPERSERVE_API_KEY   Required. API key for Superserve.ai");
    eprintln!();
    eprintln!("EXIT CODES:");
    eprintln!("  0  Success");
    eprintln!("  1  Error");
    eprintln!("  2  Invalid arguments");
    process::exit(2);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
    }

    let command = &args[1];
    let sub_args = &args[2..];

    // Check for API key
    if env::var("SUPERSERVE_API_KEY").is_err() {
        eprintln!("error: SUPERSERVE_API_KEY environment variable is not set");
        process::exit(1);
    }

    let client = match SuperserveClient::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    };

    match command.as_str() {
        "sandbox" => {
            if sub_args.is_empty() {
                eprintln!("error: 'sandbox' requires a subcommand (create, list, get, pause, resume, delete, exec)");
                process::exit(2);
            }
            let sub = &sub_args[0];
            let rest = &sub_args[1..];
            match sub.as_str() {
                "create" => cmd_sandbox_create(&client, rest),
                "list" => cmd_sandbox_list(&client, rest),
                "get" => cmd_sandbox_get(&client, rest),
                "pause" => cmd_sandbox_pause(&client, rest),
                "resume" => cmd_sandbox_resume(&client, rest),
                "delete" => cmd_sandbox_delete(&client, rest),
                "exec" => cmd_sandbox_exec(&client, rest),
                other => {
                    eprintln!("error: unknown sandbox subcommand '{}'", other);
                    eprintln!("       valid subcommands: create, list, get, pause, resume, delete, exec");
                    process::exit(2);
                }
            }
        }
        "template" => {
            if sub_args.is_empty() {
                eprintln!("error: 'template' requires a subcommand (create, list, get, rebuild)");
                process::exit(2);
            }
            let sub = &sub_args[0];
            let rest = &sub_args[1..];
            match sub.as_str() {
                "create" => cmd_template_create(&client, rest),
                "list" => cmd_template_list(&client, rest),
                "get" => cmd_template_get(&client, rest),
                "rebuild" => cmd_template_rebuild(&client, rest),
                other => {
                    eprintln!("error: unknown template subcommand '{}'", other);
                    eprintln!("       valid subcommands: create, list, get, rebuild");
                    process::exit(2);
                }
            }
        }
        "health" => cmd_health(&client),
        "run" => cmd_run(&client, sub_args),
        "--help" | "-h" | "help" => print_usage(),
        other => {
            eprintln!("error: unknown command '{}'", other);
            eprintln!("       run 'cz-serve help' for usage information");
            process::exit(2);
        }
    }
}
