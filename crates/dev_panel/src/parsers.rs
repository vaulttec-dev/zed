//! Pure, panic-free parsers for the manifest formats the Dev panel understands.
//! Every parser isolates failure to itself (a malformed file yields nothing),
//! so one bad manifest can never abort the whole scan.

use std::collections::BTreeMap;

use collections::HashSet;
use serde::Deserialize;

/// Parse `package.json`, returning the declared package manager (if any) and
/// each `(script name, script body)` pair.
pub(crate) fn parse_package_scripts(text: &str) -> (Option<&'static str>, Vec<(String, String)>) {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        return (None, Vec::new());
    };
    let package_manager = value
        .get("packageManager")
        .and_then(|v| v.as_str())
        .map(package_manager_from_field);
    let scripts = value
        .get("scripts")
        .and_then(|s| s.as_object())
        .map(|obj| {
            obj.iter()
                .map(|(name, body)| (name.clone(), body.as_str().unwrap_or_default().to_string()))
                .collect()
        })
        .unwrap_or_default();
    (package_manager, scripts)
}

fn package_manager_from_field(field: &str) -> &'static str {
    if field.starts_with("pnpm") {
        "pnpm"
    } else if field.starts_with("yarn") {
        "yarn"
    } else if field.starts_with("bun") {
        "bun"
    } else {
        "npm"
    }
}

/// Parse `Cargo.toml`, returning `(label, cargo args)` for each runnable binary.
/// A pure `[workspace]`/virtual manifest yields nothing.
pub(crate) fn parse_cargo_runnables(text: &str) -> Vec<(String, Vec<String>)> {
    #[derive(Deserialize)]
    struct CargoManifest {
        package: Option<CargoPackage>,
        #[serde(default)]
        bin: Vec<CargoBin>,
    }
    #[derive(Deserialize)]
    struct CargoPackage {
        #[allow(dead_code)]
        name: Option<String>,
    }
    #[derive(Deserialize)]
    struct CargoBin {
        name: Option<String>,
    }

    let Ok(manifest) = toml::from_str::<CargoManifest>(text) else {
        return Vec::new();
    };
    if manifest.package.is_none() {
        return Vec::new();
    }
    let bins: Vec<String> = manifest.bin.into_iter().filter_map(|b| b.name).collect();
    if bins.is_empty() {
        return vec![("cargo run".to_string(), vec!["run".to_string()])];
    }
    bins.into_iter()
        .map(|name| {
            (
                format!("cargo run --bin {name}"),
                vec!["run".to_string(), "--bin".to_string(), name],
            )
        })
        .collect()
}

/// Extract `(target, first recipe line)` pairs from a Makefile.
pub(crate) fn parse_makefile_targets(text: &str) -> Vec<(String, Option<String>)> {
    let mut out: Vec<(String, Option<String>)> = Vec::new();
    let mut seen = HashSet::default();
    let mut pending: Option<usize> = None;
    for line in text.lines() {
        // Recipe lines are indented; capture the first for the pending target.
        if line.starts_with([' ', '\t']) {
            if let Some(idx) = pending.take() {
                let recipe = line.trim();
                let recipe = recipe.strip_prefix(['@', '-', '+']).unwrap_or(recipe);
                if let Some((_, slot)) = out.get_mut(idx) {
                    if slot.is_none() && !recipe.is_empty() {
                        *slot = Some(recipe.to_string());
                    }
                }
            }
            continue;
        }
        pending = None;
        let Some(idx) = line.find(':') else {
            continue;
        };
        if line[idx..].starts_with(":=") {
            continue;
        }
        let name = line[..idx].trim();
        if name.is_empty()
            || name.starts_with('.')
            || name.contains(['%', '$', '=', '/', ' ', '\t'])
        {
            continue;
        }
        if seen.insert(name.to_string()) {
            out.push((name.to_string(), None));
            pending = Some(out.len() - 1);
        }
    }
    out
}

/// Extract recipe names from a `justfile`.
pub(crate) fn parse_justfile_recipes(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::default();
    const KEYWORDS: &[&str] = &["set", "export", "alias", "import", "mod"];
    for line in text.lines() {
        if line.starts_with([' ', '\t']) {
            continue;
        }
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('[') {
            continue;
        }
        let Some(idx) = line.find(':') else {
            continue;
        };
        // Assignments and settings use `:=` (e.g. `x := y`, `set shell := [..]`).
        if line[idx..].starts_with(":=") {
            continue;
        }
        let head = &line[..idx];
        let Some(mut name) = head.split_whitespace().next() else {
            continue;
        };
        name = name.strip_prefix('@').unwrap_or(name);
        if KEYWORDS.contains(&name) {
            continue;
        }
        if name.is_empty()
            || name
                .chars()
                .any(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
        {
            continue;
        }
        if seen.insert(name.to_string()) {
            out.push(name.to_string());
        }
    }
    out
}

/// Textual form of a YAML mapping key, coercing scalars (a key like `80` or
/// `true` is a valid, if unusual, name) rather than silently dropping it.
fn yaml_key_to_string(key: &serde_yaml::Value) -> Option<String> {
    match key {
        serde_yaml::Value::String(s) => Some(s.clone()),
        serde_yaml::Value::Number(n) => Some(n.to_string()),
        serde_yaml::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Extract task names from a `Taskfile.yml`.
pub(crate) fn parse_taskfile_tasks(text: &str) -> Vec<String> {
    let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(text) else {
        return Vec::new();
    };
    value
        .get("tasks")
        .and_then(|t| t.as_mapping())
        .map(|map| {
            map.iter()
                .filter(|(_, val)| {
                    val.as_mapping()
                        .and_then(|m| m.get("internal"))
                        .and_then(|v| v.as_bool())
                        != Some(true)
                })
                .filter_map(|(key, _)| yaml_key_to_string(key))
                .collect()
        })
        .unwrap_or_default()
}

/// Parse `Procfile` lines into `(process name, command)` pairs.
pub(crate) fn parse_procfile(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let (name, command) = line.split_once(':')?;
            let (name, command) = (name.trim(), command.trim());
            if name.is_empty() || command.is_empty() {
                return None;
            }
            Some((name.to_string(), command.to_string()))
        })
        .collect()
}

/// Extract user-facing script names from `composer.json`, skipping lifecycle hooks.
pub(crate) fn parse_composer_scripts(text: &str) -> Vec<String> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        return Vec::new();
    };
    value
        .get("scripts")
        .and_then(|s| s.as_object())
        .map(|obj| {
            obj.keys()
                .filter(|name| {
                    !name.starts_with("pre-")
                        && !name.starts_with("post-")
                        && name.as_str() != "command"
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

/// Extract console-script names from `pyproject.toml`; also report whether the
/// project uses Poetry.
pub(crate) fn parse_pyproject_scripts(text: &str) -> (bool, Vec<String>) {
    #[derive(Deserialize)]
    struct PyProject {
        project: Option<PyProjectProject>,
        tool: Option<PyTool>,
    }
    #[derive(Deserialize)]
    struct PyProjectProject {
        scripts: Option<BTreeMap<String, toml::Value>>,
    }
    #[derive(Deserialize)]
    struct PyTool {
        poetry: Option<PyPoetry>,
    }
    #[derive(Deserialize)]
    struct PyPoetry {
        scripts: Option<BTreeMap<String, toml::Value>>,
    }

    let Ok(parsed) = toml::from_str::<PyProject>(text) else {
        return (false, Vec::new());
    };
    let has_poetry = parsed.tool.as_ref().and_then(|t| t.poetry.as_ref()).is_some();
    let mut names = Vec::new();
    if let Some(scripts) = parsed.project.and_then(|p| p.scripts) {
        names.extend(scripts.into_keys());
    }
    if let Some(scripts) = parsed.tool.and_then(|t| t.poetry).and_then(|p| p.scripts) {
        names.extend(scripts.into_keys());
    }
    (has_poetry, names)
}

pub(crate) struct ZedTask {
    pub label: String,
    pub command: String,
    pub args: Vec<String>,
    pub tags: Vec<String>,
}

/// Parse `.zed/tasks.json` (lenient JSON — comments and trailing commas allowed).
pub(crate) fn parse_zed_tasks(text: &str) -> Vec<ZedTask> {
    #[derive(Deserialize)]
    struct ZedTaskDef {
        label: String,
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        tags: Vec<String>,
    }
    serde_json_lenient::from_str::<Vec<ZedTaskDef>>(text)
        .map(|tasks| {
            tasks
                .into_iter()
                .map(|t| ZedTask {
                    label: t.label,
                    command: t.command,
                    args: t.args,
                    tags: t.tags,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Extract the `services:` keys from a docker-compose document.
pub(crate) fn parse_compose_services(text: &str) -> Vec<String> {
    let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(text) else {
        return Vec::new();
    };
    value
        .get("services")
        .and_then(|s| s.as_mapping())
        .map(|map| map.keys().filter_map(yaml_key_to_string).collect())
        .unwrap_or_default()
}

#[derive(Deserialize)]
struct ComposePsRow {
    #[serde(rename = "Service")]
    service: Option<String>,
    #[serde(rename = "State")]
    state: Option<String>,
}

/// Parse `docker compose ps --format json` output into the running-service set.
pub(crate) fn parse_running_services(stdout: &[u8]) -> HashSet<String> {
    let text = String::from_utf8_lossy(stdout);
    let mut running = HashSet::default();
    let mut consider = |row: ComposePsRow| {
        if let (Some(service), Some(state)) = (row.service, row.state) {
            let state = state.to_ascii_lowercase();
            if state.contains("running") || state.starts_with("up") {
                running.insert(service);
            }
        }
    };
    if let Ok(rows) = serde_json::from_str::<Vec<ComposePsRow>>(text.trim()) {
        rows.into_iter().for_each(&mut consider);
    } else {
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            if let Ok(row) = serde_json::from_str::<ComposePsRow>(line) {
                consider(row);
            }
        }
    }
    running
}

/// Whether a Go source file declares `package main`.
pub(crate) fn go_is_main_package(text: &str) -> bool {
    let mut in_block_comment = false;
    for line in text.lines() {
        let mut trimmed = line.trim();
        if in_block_comment {
            match trimmed.find("*/") {
                Some(idx) => {
                    trimmed = trimmed[idx + 2..].trim_start();
                    in_block_comment = false;
                }
                None => continue,
            }
        }
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if trimmed.starts_with("/*") {
            match trimmed.find("*/") {
                Some(idx) => trimmed = trimmed[idx + 2..].trim_start(),
                None => {
                    in_block_comment = true;
                    continue;
                }
            }
            if trimmed.is_empty() {
                continue;
            }
        }
        if trimmed == "package main" {
            return true;
        }
        if let Some(rest) = trimmed.strip_prefix("package") {
            if rest.starts_with(char::is_whitespace) {
                return rest.split_whitespace().next() == Some("main");
            }
        }
    }
    false
}

/// Whether a bare Python entry file looks like a web server.
pub(crate) fn python_entry_is_server(text: &str) -> bool {
    const MARKERS: &[&str] = &[
        "flask(",
        "fastapi(",
        "app.run(",
        "uvicorn",
        "aiohttp",
        "sanic(",
        "tornado",
        "hypercorn",
        "gunicorn",
    ];
    let lower = text.to_ascii_lowercase();
    MARKERS.iter().any(|m| lower.contains(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scripts_and_package_manager() {
        let json = r#"{ "packageManager": "pnpm@11.13.0",
            "scripts": { "dev": "vite", "build": "vite build" } }"#;
        let (pm, scripts) = parse_package_scripts(json);
        assert_eq!(pm, Some("pnpm"));
        assert!(scripts.iter().any(|(n, b)| n == "dev" && b == "vite"));
        assert!(scripts.iter().any(|(n, b)| n == "build" && b == "vite build"));
    }

    #[test]
    fn parses_cargo_bins() {
        let bins = parse_cargo_runnables(
            "[package]\nname = \"x\"\n[[bin]]\nname = \"a\"\n[[bin]]\nname = \"b\"\n",
        );
        assert_eq!(bins.len(), 2);
        assert_eq!(bins[0].1, vec!["run", "--bin", "a"]);
        let plain = parse_cargo_runnables("[package]\nname = \"x\"\n");
        assert_eq!(plain, vec![("cargo run".to_string(), vec!["run".to_string()])]);
        let workspace = parse_cargo_runnables("[workspace]\nmembers = []\n");
        assert!(workspace.is_empty());
    }

    #[test]
    fn parses_makefile_targets_skipping_specials() {
        let mk = "\
.PHONY: all\n\
all: lint test\n\
run-dev:\n\t./run --watch\n\
build-apk:\n\tflutter build\n\
VAR := value\n\
%.o: %.c\n";
        let targets = parse_makefile_targets(mk);
        let names: Vec<&str> = targets.iter().map(|(t, _)| t.as_str()).collect();
        assert!(names.contains(&"all"));
        assert!(names.contains(&"run-dev"));
        assert!(names.contains(&"build-apk"));
        assert!(!names.iter().any(|t| t.starts_with('.')));
        assert!(!names.iter().any(|t| t.contains('%')));
        assert!(!names.contains(&"VAR"));
        let run_dev = targets.iter().find(|(t, _)| t == "run-dev").unwrap();
        assert_eq!(run_dev.1.as_deref(), Some("./run --watch"));
    }

    #[test]
    fn parses_justfile_recipes() {
        let jf = "set shell := [\"bash\"]\n# a comment\n[private]\nbuild arg=\"x\":\n\tcargo build\n@quiet:\n\techo hi\ndev: build\n\tcargo run\n";
        let recipes = parse_justfile_recipes(jf);
        assert!(recipes.contains(&"build".to_string()));
        assert!(recipes.contains(&"quiet".to_string()));
        assert!(recipes.contains(&"dev".to_string()));
        assert!(!recipes.contains(&"set".to_string()));
    }

    #[test]
    fn parses_taskfile_tasks() {
        let yaml = "version: '3'\ntasks:\n  build:\n    cmds: [go build]\n  serve:\n    cmds: [go run .]\n  _hidden:\n    internal: true\n";
        let mut tasks = parse_taskfile_tasks(yaml);
        tasks.sort();
        assert_eq!(tasks, vec!["build", "serve"]);
    }

    #[test]
    fn parses_procfile_entries() {
        let pf = "web: cargo run --package collab serve all\n# comment\nworker: sh worker.sh\n";
        let entries = parse_procfile(pf);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "web");
        assert_eq!(entries[0].1, "cargo run --package collab serve all");
    }

    #[test]
    fn parses_composer_scripts_skipping_hooks() {
        let json = r#"{ "scripts": { "serve": "php -S localhost:8000", "post-install-cmd": "x", "test": "phpunit" } }"#;
        let mut scripts = parse_composer_scripts(json);
        scripts.sort();
        assert_eq!(scripts, vec!["serve", "test"]);
    }

    #[test]
    fn parses_pyproject_scripts() {
        let toml = "[project]\nname = \"x\"\n[project.scripts]\napp = \"x:main\"\n[tool.poetry.scripts]\nworker = \"x:worker\"\n";
        let (has_poetry, mut names) = parse_pyproject_scripts(toml);
        names.sort();
        assert!(has_poetry);
        assert_eq!(names, vec!["app", "worker"]);
    }

    #[test]
    fn parses_zed_tasks_with_tags() {
        let tasks = "[\n  { \"label\": \"clippy\", \"command\": \"./script/clippy\", \"args\": [], },\n  { \"label\": \"srv\", \"command\": \"cargo\", \"args\": [\"run\"], \"tags\": [\"dev:server\"] },\n]";
        let parsed = parse_zed_tasks(tasks);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[1].tags, vec!["dev:server".to_string()]);
    }

    #[test]
    fn parses_compose_services() {
        let yaml = "services:\n  postgres:\n    image: postgres\n  minio:\n    image: minio\n";
        let mut services = parse_compose_services(yaml);
        services.sort();
        assert_eq!(services, vec!["minio", "postgres"]);
    }

    #[test]
    fn coerces_non_string_service_keys() {
        // An unquoted numeric service name is a YAML number key, not a string.
        let yaml = "services:\n  8080:\n    image: proxy\n  web:\n    image: web\n";
        let mut services = parse_compose_services(yaml);
        services.sort();
        assert_eq!(services, vec!["8080", "web"]);
    }

    #[test]
    fn parses_running_services() {
        let out = br#"{"Service":"postgres","State":"running"}
{"Service":"minio","State":"exited"}"#;
        let running = parse_running_services(out);
        assert!(running.contains("postgres"));
        assert!(!running.contains("minio"));
    }

    #[test]
    fn detects_go_main_package() {
        assert!(go_is_main_package("//go:build linux\npackage main\n\nfunc main() {}\n"));
        assert!(go_is_main_package("/* header */ package main\n"));
        assert!(!go_is_main_package("package server\n"));
    }
}
