//! Deterministic Server-vs-Script classification for detected runnables.
//!
//! The safe default is Script: "Start all servers" must never auto-run a
//! one-shot, and a destructive task (deploy/migrate/clean) misfiled as a server
//! would be dangerous. Only unambiguous long-running signals — a known server
//! runner, a genuine watch/serve/reload flag, or a strong server name — promote
//! a runnable to Server.

/// Whether a runnable is a long-running server or a one-shot script.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Category {
    Server,
    Script,
}

/// One-shot / destructive names that must never be auto-started as servers.
const HARD_SCRIPT: &[&str] = &[
    "deploy", "release", "publish", "ship", "migrate", "migration", "migrations", "seed", "reset",
    "rollback", "drop", "destroy", "teardown", "clean", "clobber", "purge", "prune", "install",
    "uninstall", "reinstall", "setup", "bootstrap", "prepare", "provision", "codegen", "scaffold",
    "generate", "gen", "package", "dist", "pack",
];

/// Unambiguous long-running name tokens.
const STRONG_SERVER_NAME: &[&str] = &[
    "dev", "start", "serve", "server", "watch", "preview", "storybook", "hot", "live", "hmr",
    "tunnel", "proxy", "daemon", "livereload", "hotreload", "devserver",
];

/// One-shot name tokens (weaker than the runner/flag rules above).
const SOFT_SCRIPT_NAME: &[&str] = &[
    "test", "tests", "build", "compile", "lint", "fmt", "format", "check", "checks", "typecheck",
    "types", "ci", "coverage", "cov", "e2e", "bench", "benchmark", "doc", "docs", "clippy", "audit",
    "analyze", "validate", "verify", "export",
];

/// Binaries that always imply a long-running process.
const UNCONDITIONAL_RUNNERS: &[&str] = &[
    "nodemon",
    "ts-node-dev",
    "webpack-dev-server",
    "uvicorn",
    "gunicorn",
    "hypercorn",
    "daphne",
    "granian",
    "runserver",
    "air",
    "reflex",
    "wgo",
    "watchexec",
    "entr",
    "livekit-server",
    "nginx",
    "postgres",
    "redis-server",
    "mongod",
    "mailhog",
    "browser-sync",
    "live-server",
    "http-server",
    "json-server",
    "cargo-watch",
];

/// Classify a runnable as Server or Script. The safe default is Script.
pub(crate) fn classify(label: &str, full_command: &str) -> Category {
    let label_lower = label.to_ascii_lowercase();
    let command_lower = full_command.to_ascii_lowercase();
    let label_tokens: Vec<&str> = label_lower
        .split(|c: char| matches!(c, ':' | '-' | '_' | '/' | '.' | ' '))
        .filter(|s| !s.is_empty())
        .collect();

    // 1. Hard one-shot / destructive names — never a server.
    if label_tokens.iter().any(|t| HARD_SCRIPT.contains(t)) {
        return Category::Script;
    }
    // 2. A known server runner in the resolved command.
    if command_implies_server(&command_lower) {
        return Category::Server;
    }
    // 3. An explicit long-running flag (--watch/--reload/--serve/...).
    if command_has_server_flag(&command_lower) {
        return Category::Server;
    }
    // 4. A strong server name token (dev/serve/watch/...).
    if label_tokens.iter().any(|t| STRONG_SERVER_NAME.contains(t)) {
        return Category::Server;
    }
    // 5. A soft one-shot name token (test/build/lint/...).
    if label_tokens.iter().any(|t| SOFT_SCRIPT_NAME.contains(t)) {
        return Category::Script;
    }
    Category::Script
}

fn command_tokens(command: &str) -> Vec<&str> {
    command
        .split(|c: char| {
            c.is_whitespace() || matches!(c, ';' | '|' | '&' | '"' | '\'' | '(' | ')' | '`')
        })
        .filter(|s| !s.is_empty())
        .collect()
}

fn command_implies_server(command: &str) -> bool {
    let tokens = command_tokens(command);
    if tokens.iter().any(|t| UNCONDITIONAL_RUNNERS.contains(t)) {
        return true;
    }
    for (i, &token) in tokens.iter().enumerate() {
        let sub = tokens
            .get(i + 1..)
            .and_then(|rest| rest.iter().copied().find(|a| !a.starts_with('-')));
        let is_server = match token {
            "vite" => sub != Some("build") && sub != Some("optimize"),
            "next" => matches!(sub, Some("dev") | Some("start")),
            "nuxt" | "nuxi" => sub == Some("dev"),
            "astro" => matches!(sub, Some("dev") | Some("preview")),
            "ng" => sub == Some("serve"),
            "vue-cli-service" => sub == Some("serve"),
            "wrangler" => sub == Some("dev"),
            "gatsby" => sub == Some("develop"),
            "expo" => sub == Some("start"),
            "rails" | "bin/rails" => matches!(sub, Some("server") | Some("s")),
            "artisan" => sub == Some("serve"),
            "flask" => tokens.contains(&"run"),
            "django-admin" | "manage.py" | "./manage.py" => sub == Some("runserver"),
            "webpack" | "rspack" => sub == Some("serve"),
            "remix" | "snowpack" => sub == Some("dev"),
            "storybook" | "start-storybook" => matches!(sub, Some("dev") | Some("start") | None),
            "parcel" => sub != Some("build"),
            "tsc" => tokens.iter().any(|t| *t == "-w" || *t == "--watch"),
            "cargo" => sub == Some("watch"),
            "caddy" => sub == Some("run"),
            "vercel" | "netlify" => sub == Some("dev"),
            "supabase" => sub == Some("start"),
            _ => false,
        };
        if is_server {
            return true;
        }
    }
    false
}

fn command_has_server_flag(command: &str) -> bool {
    // Only unambiguous long-running flags. `--host`/`--dev` are intentionally
    // excluded: they appear on one-shot client commands (`psql --host=…`,
    // `yarn add x --dev`), and genuine dev servers that bind a host are already
    // promoted by `command_implies_server`.
    const FLAGS: &[&str] = &[
        "--watch",
        "--reload",
        "--hot",
        "--hmr",
        "--serve",
        "--live",
        "--tunnel",
        "--hot-reload",
        "--livereload",
    ];
    const WATCH_TOOLS: &[&str] = &[
        "tsc",
        "jest",
        "vitest",
        "rollup",
        "esbuild",
        "sass",
        "tailwindcss",
        "cargo-watch",
        "webpack",
        "postcss",
    ];
    let tokens: Vec<&str> = command.split_whitespace().collect();
    for &token in &tokens {
        let head = token.split('=').next().unwrap_or(token);
        if FLAGS.contains(&head) {
            return true;
        }
    }
    if tokens.contains(&"-w") && tokens.iter().any(|t| WATCH_TOOLS.contains(t)) {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_servers_and_scripts() {
        assert_eq!(classify("dev", "vite"), Category::Server);
        assert_eq!(classify("dev:portal", "next dev"), Category::Server);
        assert_eq!(classify("build", "vite build"), Category::Script);
        assert_eq!(classify("test", "jest"), Category::Script);
        // A watcher named "test" is still long-running.
        assert_eq!(classify("test:watch", "jest --watch"), Category::Server);
        assert_eq!(classify("test:watch", ""), Category::Server);
        // Destructive one-shots must never be servers.
        assert_eq!(classify("deploy", "serve dist"), Category::Script);
        assert_eq!(classify("migrate", "prisma migrate deploy"), Category::Script);
        // Opaque command with a server runner in the body.
        assert_eq!(classify("api", "nodemon server.js"), Category::Server);
        // `next build` is not a server despite the runner.
        assert_eq!(classify("compile", "next build"), Category::Script);
        // Ambiguous name + opaque command defaults to Script (safe).
        assert_eq!(classify("worker", "node worker.js"), Category::Script);
        // Client `--host`/`--dev` flags must not promote a one-shot to Server.
        assert_eq!(classify("db-shell", "psql --host=db"), Category::Script);
        assert_eq!(classify("backup", "pg_dump --host=db mydb"), Category::Script);
        assert_eq!(classify("deps", "yarn add x --dev"), Category::Script);
    }

    #[test]
    fn cargo_run_stays_a_script_by_default() {
        assert_eq!(classify("cargo run", "cargo run"), Category::Script);
        assert_eq!(classify("cargo watch", "cargo watch -x run"), Category::Server);
    }
}
