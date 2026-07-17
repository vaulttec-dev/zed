//! Tree-wide manifest detection: turns the files in a project's worktrees into
//! the [`Runnable`]s and [`Container`]s the Dev panel shows.
//!
//! The panel collects [`Candidate`]s on the foreground (reading the worktree
//! snapshot), then [`scan_candidates`] loads and parses them off-thread.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use collections::{HashMap, HashSet};
use gpui::SharedString;
use project::Fs;

use crate::classification::{Category, classify};
use crate::parsers::{
    go_is_main_package, parse_cargo_runnables, parse_compose_services, parse_composer_scripts,
    parse_justfile_recipes, parse_makefile_targets, parse_package_scripts, parse_procfile,
    parse_pyproject_scripts, parse_taskfile_tasks, parse_zed_tasks, python_entry_is_server,
};

/// Directories never scanned for manifests even if not git-ignored.
pub(crate) const SKIP_DIRS: &[&str] = &[
    "node_modules",
    "target",
    "dist",
    "build",
    ".venv",
    "venv",
    "vendor",
    ".git",
    "__pycache__",
    ".next",
    ".turbo",
    "testdata",
];

/// Maximum manifest nesting depth (root file = 1 component).
pub(crate) const MAX_MANIFEST_DEPTH: usize = 8;

/// A detected runnable command from any source.
#[derive(Clone)]
pub(crate) struct Runnable {
    /// Stable unique id, e.g. `npm:apps/web:dev`, `make:.:run`, `proc::web`.
    pub id: String,
    /// Display label (dir-prefixed only when it would otherwise collide).
    pub label: SharedString,
    /// Executable to run.
    pub program: String,
    /// Arguments.
    pub args: Vec<String>,
    /// Absolute working directory (the manifest's directory).
    pub cwd: Arc<Path>,
    pub category: Category,
}

/// A detected docker-compose service.
#[derive(Clone)]
pub(crate) struct Container {
    /// Stable unique id: `compose:{dir}:{service}` (dir-qualified so services
    /// with the same name across compose files never collide).
    pub id: String,
    pub service: String,
    pub label: SharedString,
    /// Absolute path of the compose file this service belongs to.
    pub compose_file: Arc<Path>,
    /// The compose file's directory — the working dir for `docker compose`.
    pub dir: Arc<Path>,
    /// Compose file name, relative to `dir` (used with `-f` for non-canonical
    /// files that docker's auto-discovery would miss).
    pub file_name: String,
    /// Whether the file must be passed explicitly via `-f`.
    pub explicit: bool,
}

#[derive(Clone)]
pub(crate) enum CandidateKind {
    PackageJson,
    CargoToml,
    Makefile,
    Justfile,
    Taskfile,
    Procfile,
    Composer,
    Csproj,
    Pyproject,
    ManagePy,
    PyEntry,
    GoMain,
    ZedTasks,
    Compose { canonical: bool },
}

#[derive(Clone)]
pub(crate) struct Candidate {
    pub kind: CandidateKind,
    pub abs: PathBuf,
    /// Absolute directory of the manifest — the runnable's working dir and the
    /// basis for its globally-unique id (relative paths would collide across
    /// worktrees, e.g. two roots both yielding `""`).
    pub dir: Arc<Path>,
    pub file_name: String,
}

/// Sibling markers collected during the foreground pass (keyed by absolute dir).
#[derive(Default)]
pub(crate) struct ScanIndex {
    package_manager: HashMap<PathBuf, &'static str>,
    pub poetry_dirs: HashSet<PathBuf>,
    pub pyproject_dirs: HashSet<PathBuf>,
    pub managepy_dirs: HashSet<PathBuf>,
}

impl ScanIndex {
    pub fn set_pm(&mut self, dir: &Path, pm: &'static str) {
        self.package_manager.entry(dir.to_path_buf()).or_insert(pm);
    }
}

pub(crate) fn candidate_kind(
    file_name: &str,
    extension: Option<&str>,
    parent_file_name: Option<&str>,
    parent_has_cmd: bool,
) -> Option<CandidateKind> {
    match file_name {
        "package.json" => return Some(CandidateKind::PackageJson),
        "Cargo.toml" => return Some(CandidateKind::CargoToml),
        "Makefile" | "makefile" | "GNUmakefile" | "BSDmakefile" => {
            return Some(CandidateKind::Makefile);
        }
        "composer.json" => return Some(CandidateKind::Composer),
        "pyproject.toml" => return Some(CandidateKind::Pyproject),
        "manage.py" => return Some(CandidateKind::ManagePy),
        "main.py" | "app.py" => return Some(CandidateKind::PyEntry),
        "main.go" => return Some(CandidateKind::GoMain),
        "tasks.json" => {
            if parent_file_name == Some(".zed") {
                return Some(CandidateKind::ZedTasks);
            }
        }
        _ => {}
    }
    if file_name.eq_ignore_ascii_case("justfile")
        || file_name.trim_start_matches('.').eq_ignore_ascii_case("justfile")
    {
        return Some(CandidateKind::Justfile);
    }
    if let Some(canonical) = compose_kind(file_name) {
        return Some(CandidateKind::Compose { canonical });
    }
    if file_name.starts_with("Procfile") {
        return Some(CandidateKind::Procfile);
    }
    let lower = file_name.to_ascii_lowercase();
    if lower.starts_with("taskfile.") && matches!(extension, Some("yml") | Some("yaml")) {
        return Some(CandidateKind::Taskfile);
    }
    if matches!(extension, Some("csproj") | Some("fsproj")) {
        return Some(CandidateKind::Csproj);
    }
    // A Go `main` package can also live under `cmd/<name>/` without `main.go`.
    if extension == Some("go") && !file_name.ends_with("_test.go") && parent_has_cmd {
        return Some(CandidateKind::GoMain);
    }
    None
}

/// `Some(true)` for canonical compose files (docker auto-discovers), `Some(false)`
/// for extra files that need an explicit `-f`.
fn compose_kind(file_name: &str) -> Option<bool> {
    let lower = file_name.to_ascii_lowercase();
    if !(lower.ends_with(".yml") || lower.ends_with(".yaml")) {
        return None;
    }
    const CANONICAL: &[&str] = &[
        "compose.yaml",
        "compose.yml",
        "docker-compose.yaml",
        "docker-compose.yml",
    ];
    if CANONICAL.contains(&lower.as_str()) {
        return Some(true);
    }
    if lower.starts_with("compose.") || lower.starts_with("docker-compose.") {
        return Some(false);
    }
    None
}

/// Intermediate detection result, before label disambiguation.
struct Detected {
    id: String,
    base: String,
    dir_seg: String,
    program: String,
    args: Vec<String>,
    cwd: Arc<Path>,
    category: Category,
}

pub(crate) async fn scan_candidates(
    fs: Arc<dyn Fs>,
    candidates: Vec<Candidate>,
    index: ScanIndex,
) -> (Vec<Runnable>, Vec<Container>) {
    let mut detected: Vec<Detected> = Vec::new();
    let mut go_dirs: HashSet<PathBuf> = HashSet::default();
    let mut containers: Vec<Container> = Vec::new();
    let mut container_ids: HashSet<String> = HashSet::default();

    // Canonical compose files first so a base file wins over an override
    // sharing a dir+service.
    let mut candidates = candidates;
    candidates.sort_by_key(|c| matches!(c.kind, CandidateKind::Compose { canonical: false }));

    for candidate in candidates {
        // Ids are keyed on the absolute directory so identically-named manifests
        // in different worktrees stay distinct; `dir_seg` is the folder name used
        // only for display disambiguation.
        let id_dir = candidate.dir.to_string_lossy();
        let dir_seg = dir_segment(&candidate.dir);
        match &candidate.kind {
            CandidateKind::PackageJson => {
                let Ok(text) = fs.load(&candidate.abs).await else {
                    continue;
                };
                let (pm_field, scripts) = parse_package_scripts(&text);
                let pm = pm_field
                    .or_else(|| index.package_manager.get(candidate.dir.as_ref()).copied())
                    .unwrap_or("npm");
                for (name, body) in scripts {
                    detected.push(Detected {
                        id: format!("npm:{}:{}", id_dir, name),
                        category: classify(&name, &body),
                        base: name.clone(),
                        dir_seg: dir_seg.clone(),
                        program: pm.to_string(),
                        args: vec!["run".into(), name],
                        cwd: candidate.dir.clone(),
                    });
                }
            }
            CandidateKind::CargoToml => {
                let Ok(text) = fs.load(&candidate.abs).await else {
                    continue;
                };
                for (base, args) in parse_cargo_runnables(&text) {
                    detected.push(Detected {
                        id: format!("cargo:{}:{}", id_dir, base),
                        category: classify(&base, &format!("cargo {}", args.join(" "))),
                        base,
                        dir_seg: dir_seg.clone(),
                        program: "cargo".into(),
                        args,
                        cwd: candidate.dir.clone(),
                    });
                }
            }
            CandidateKind::Makefile => {
                let Ok(text) = fs.load(&candidate.abs).await else {
                    continue;
                };
                for (target, recipe) in parse_makefile_targets(&text) {
                    let command = recipe.unwrap_or_else(|| format!("make {target}"));
                    detected.push(Detected {
                        id: format!("make:{}:{}", id_dir, target),
                        category: classify(&target, &command),
                        base: format!("make {target}"),
                        dir_seg: dir_seg.clone(),
                        program: "make".into(),
                        args: vec![target],
                        cwd: candidate.dir.clone(),
                    });
                }
            }
            CandidateKind::Justfile => {
                let Ok(text) = fs.load(&candidate.abs).await else {
                    continue;
                };
                for recipe in parse_justfile_recipes(&text) {
                    detected.push(Detected {
                        id: format!("just:{}:{}", id_dir, recipe),
                        category: classify(&recipe, &recipe),
                        base: format!("just {recipe}"),
                        dir_seg: dir_seg.clone(),
                        program: "just".into(),
                        args: vec![recipe],
                        cwd: candidate.dir.clone(),
                    });
                }
            }
            CandidateKind::Taskfile => {
                let Ok(text) = fs.load(&candidate.abs).await else {
                    continue;
                };
                for name in parse_taskfile_tasks(&text) {
                    detected.push(Detected {
                        id: format!("task:{}:{}", id_dir, name),
                        category: classify(&name, &name),
                        base: format!("task {name}"),
                        dir_seg: dir_seg.clone(),
                        program: "task".into(),
                        args: vec![name],
                        cwd: candidate.dir.clone(),
                    });
                }
            }
            CandidateKind::Procfile => {
                let Ok(text) = fs.load(&candidate.abs).await else {
                    continue;
                };
                let suffix = candidate
                    .file_name
                    .strip_prefix("Procfile.")
                    .map(|s| s.to_string());
                for (name, command) in parse_procfile(&text) {
                    let id_suffix = suffix.as_deref().unwrap_or("");
                    let base = match &suffix {
                        Some(s) => format!("{name} ({s})"),
                        None => name.clone(),
                    };
                    detected.push(Detected {
                        id: format!("proc:{}:{id_suffix}:{name}", id_dir),
                        // Procfile entries are long-running processes by contract.
                        category: Category::Server,
                        base,
                        dir_seg: dir_seg.clone(),
                        program: "sh".into(),
                        args: vec!["-lc".into(), command],
                        cwd: candidate.dir.clone(),
                    });
                }
            }
            CandidateKind::Composer => {
                let Ok(text) = fs.load(&candidate.abs).await else {
                    continue;
                };
                for name in parse_composer_scripts(&text) {
                    detected.push(Detected {
                        id: format!("composer:{}:{}", id_dir, name),
                        category: classify(&name, &name),
                        base: name.clone(),
                        dir_seg: dir_seg.clone(),
                        program: "composer".into(),
                        args: vec!["run".into(), name],
                        cwd: candidate.dir.clone(),
                    });
                }
            }
            CandidateKind::Csproj => {
                let base = candidate
                    .file_name
                    .rsplit_once('.')
                    .map(|(stem, _)| stem.to_string())
                    .unwrap_or_else(|| candidate.file_name.clone());
                detected.push(Detected {
                    id: format!("dotnet:{}:{}", id_dir, base),
                    category: classify(&base, "dotnet run"),
                    base: format!("dotnet run ({base})"),
                    dir_seg: dir_seg.clone(),
                    program: "dotnet".into(),
                    args: vec!["run".into(), "--project".into(), candidate.file_name.clone()],
                    cwd: candidate.dir.clone(),
                });
            }
            CandidateKind::Pyproject => {
                let Ok(text) = fs.load(&candidate.abs).await else {
                    continue;
                };
                let (has_poetry, scripts) = parse_pyproject_scripts(&text);
                let use_poetry = has_poetry || index.poetry_dirs.contains(candidate.dir.as_ref());
                for name in scripts {
                    let (program, args) = if use_poetry {
                        ("poetry".to_string(), vec!["run".into(), name.clone()])
                    } else {
                        (python_program().to_string(), vec![name.clone()])
                    };
                    detected.push(Detected {
                        id: format!("py:{}:{}", id_dir, name),
                        category: classify(&name, &name),
                        base: name.clone(),
                        dir_seg: dir_seg.clone(),
                        program,
                        args,
                        cwd: candidate.dir.clone(),
                    });
                }
            }
            CandidateKind::ManagePy => {
                let Ok(text) = fs.load(&candidate.abs).await else {
                    continue;
                };
                if !text.to_ascii_lowercase().contains("django") {
                    continue;
                }
                for (sub, category) in [
                    ("runserver", Category::Server),
                    ("migrate", Category::Script),
                ] {
                    detected.push(Detected {
                        id: format!("django:{}:{sub}", id_dir),
                        category,
                        base: format!("manage.py {sub}"),
                        dir_seg: dir_seg.clone(),
                        program: python_program().into(),
                        args: vec!["manage.py".into(), sub.into()],
                        cwd: candidate.dir.clone(),
                    });
                }
            }
            CandidateKind::PyEntry => {
                // Only a real entrypoint when no packaging manifest governs the dir.
                if index.pyproject_dirs.contains(candidate.dir.as_ref())
                    || index.managepy_dirs.contains(candidate.dir.as_ref())
                {
                    continue;
                }
                let Ok(text) = fs.load(&candidate.abs).await else {
                    continue;
                };
                let category = if python_entry_is_server(&text) {
                    Category::Server
                } else {
                    Category::Script
                };
                detected.push(Detected {
                    id: format!("py-entry:{}:{}", id_dir, candidate.file_name),
                    category,
                    base: format!("python {}", candidate.file_name),
                    dir_seg: dir_seg.clone(),
                    program: python_program().into(),
                    args: vec![candidate.file_name.clone()],
                    cwd: candidate.dir.clone(),
                });
            }
            CandidateKind::GoMain => {
                if !go_dirs.insert(candidate.dir.to_path_buf()) {
                    continue;
                }
                let Ok(text) = fs.load(&candidate.abs).await else {
                    go_dirs.remove(candidate.dir.as_ref());
                    continue;
                };
                if !go_is_main_package(&text) {
                    go_dirs.remove(candidate.dir.as_ref());
                    continue;
                }
                let name = dir_seg.clone();
                let base = if name.is_empty() {
                    "go run".to_string()
                } else {
                    format!("go run ({name})")
                };
                detected.push(Detected {
                    id: format!("go:{}", id_dir),
                    category: classify(&name, "go run ."),
                    base,
                    dir_seg: name,
                    program: "go".into(),
                    args: vec!["run".into(), ".".into()],
                    cwd: candidate.dir.clone(),
                });
            }
            CandidateKind::ZedTasks => {
                let Ok(text) = fs.load(&candidate.abs).await else {
                    continue;
                };
                for task in parse_zed_tasks(&text) {
                    let command = format!("{} {}", task.command, task.args.join(" "));
                    let category = if task.tags.iter().any(|t| t == "dev:server") {
                        Category::Server
                    } else if task.tags.iter().any(|t| t == "dev:script") {
                        Category::Script
                    } else {
                        classify(&task.label, &command)
                    };
                    detected.push(Detected {
                        id: format!("zed:{}:{}", id_dir, task.label),
                        category,
                        base: task.label,
                        dir_seg: dir_seg.clone(),
                        program: task.command,
                        args: task.args,
                        cwd: candidate.dir.clone(),
                    });
                }
            }
            CandidateKind::Compose { canonical } => {
                let Ok(text) = fs.load(&candidate.abs).await else {
                    continue;
                };
                let compose_file: Arc<Path> = Arc::from(candidate.abs.as_path());
                for service in parse_compose_services(&text) {
                    let id = format!("compose:{}:{}", id_dir, service);
                    if !container_ids.insert(id.clone()) {
                        continue;
                    }
                    containers.push(Container {
                        label: service.clone().into(),
                        id,
                        service,
                        compose_file: compose_file.clone(),
                        dir: candidate.dir.clone(),
                        file_name: candidate.file_name.clone(),
                        explicit: !canonical,
                    });
                }
            }
        }
    }

    let runnables = finalize_runnables(detected);
    (runnables, containers)
}

/// Deduplicate by id and disambiguate labels that collide across directories.
fn finalize_runnables(detected: Vec<Detected>) -> Vec<Runnable> {
    let mut seen = HashSet::default();
    let detected: Vec<Detected> = detected
        .into_iter()
        .filter(|d| seen.insert(d.id.clone()))
        .collect();

    let mut counts: HashMap<&str, usize> = HashMap::default();
    for d in &detected {
        *counts.entry(d.base.as_str()).or_insert(0) += 1;
    }

    detected
        .iter()
        .map(|d| {
            let collides = counts.get(d.base.as_str()).copied().unwrap_or(1) > 1;
            let label = if collides && !d.dir_seg.is_empty() {
                format!("{} ▸ {}", d.dir_seg, d.base)
            } else {
                d.base.clone()
            };
            Runnable {
                id: d.id.clone(),
                label: label.into(),
                program: d.program.clone(),
                args: d.args.clone(),
                cwd: d.cwd.clone(),
                category: d.category,
            }
        })
        .collect()
}

/// Final component of a directory path — the immediate folder name (for a
/// worktree root this is the project folder itself, which disambiguates
/// identically-named root scripts across multiple roots).
fn dir_segment(dir: &Path) -> String {
    dir.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn python_program() -> &'static str {
    if cfg!(windows) { "python" } else { "python3" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_compose_kind() {
        assert_eq!(compose_kind("compose.yml"), Some(true));
        assert_eq!(compose_kind("docker-compose.yaml"), Some(true));
        assert_eq!(compose_kind("docker-compose.override.yml"), Some(false));
        assert_eq!(compose_kind("compose.dev.yaml"), Some(false));
        assert_eq!(compose_kind("composer.json"), None);
        assert_eq!(compose_kind("settings.yml"), None);
    }

    #[test]
    fn disambiguates_colliding_labels() {
        let cwd: Arc<Path> = Arc::from(Path::new("/tmp"));
        let detected = vec![
            Detected {
                id: "npm:apps/web:dev".into(),
                base: "dev".into(),
                dir_seg: "web".into(),
                program: "npm".into(),
                args: vec![],
                cwd: cwd.clone(),
                category: Category::Server,
            },
            Detected {
                id: "npm:apps/api:dev".into(),
                base: "dev".into(),
                dir_seg: "api".into(),
                program: "npm".into(),
                args: vec![],
                cwd,
                category: Category::Server,
            },
        ];
        let runnables = finalize_runnables(detected);
        let labels: Vec<&str> = runnables.iter().map(|r| r.label.as_ref()).collect();
        assert!(labels.contains(&"web ▸ dev"));
        assert!(labels.contains(&"api ▸ dev"));
    }
}
