//! Dev Panel — a left-dock panel that auto-detects a project's runnable tasks
//! (dev servers and one-shot workflows) and Docker services, and lets you
//! start/stop them with one click.
//!
//! Detection is universal and file-native — it reads whatever the project
//! already has, across stacks:
//! - `package.json` scripts (Node)
//! - `Makefile` targets (any language)
//! - `Procfile` process entries (always servers)
//! - `.zed/tasks.json` native Zed tasks
//! - `Cargo.toml` (`cargo run`)
//! - `docker-compose` services (containers), with live status from `docker
//!   compose ps`
//!
//! Each runnable gets a heuristic **Server / Workflow** category that you can
//! override per row; the choice is remembered per project in Zed's local DB
//! (no file in the repo). Everything runs in Zed's own terminal panel.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use collections::{HashMap, HashSet};
use db::kvp::KeyValueStore;
use gpui::{
    Action, AnyElement, App, AsyncWindowContext, Context, Entity, EventEmitter, FocusHandle,
    Focusable, IntoElement, ParentElement, Pixels, Render, Styled, Task, WeakEntity, Window,
    actions, px,
};
use project::Project;
use serde::{Deserialize, Serialize};
use task::{RevealStrategy, SpawnInTerminal, TaskId};
use terminal::Terminal;
use terminal_view::terminal_panel::TerminalPanel;
use ui::{Tooltip, prelude::*};
use util::command::new_command;
use workspace::{
    Workspace,
    dock::{DockPosition, Panel, PanelEvent},
};

actions!(
    dev_panel,
    [
        /// Toggle focus on the Dev panel.
        ToggleFocus
    ]
);

/// Register the panel's toggle action on every workspace.
pub fn init(cx: &mut App) {
    cx.observe_new(|workspace: &mut Workspace, _, _| {
        workspace.register_action(|workspace, _: &ToggleFocus, window, cx| {
            workspace.toggle_panel_focus::<DevPanel>(window, cx);
        });
    })
    .detach();
}

/// Whether a runnable is a long-running server or a one-shot workflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Category {
    Server,
    Workflow,
}

impl Category {
    fn opposite(self) -> Self {
        match self {
            Category::Server => Category::Workflow,
            Category::Workflow => Category::Server,
        }
    }
}

/// A detected runnable command from any source.
#[derive(Clone)]
struct Runnable {
    /// Stable unique id, e.g. `npm:dev`, `make:run-dev`, `proc:web`, `zed:clippy`.
    id: String,
    /// Display label.
    label: String,
    /// Executable to run.
    program: String,
    /// Arguments.
    args: Vec<String>,
    /// Heuristic category before any user override.
    default_category: Category,
}

pub struct DevPanel {
    workspace: WeakEntity<Workspace>,
    project: Entity<Project>,
    focus_handle: FocusHandle,
    root: Option<Arc<Path>>,
    runnables: Vec<Runnable>,
    /// Per-runnable category overrides (id → category).
    overrides: HashMap<String, Category>,
    /// KV key under which `overrides` persists (per project root).
    override_key: Option<String>,
    containers: Vec<String>,
    container_running: HashSet<String>,
    /// Live terminals for started servers, keyed by runnable id.
    running: HashMap<String, WeakEntity<Terminal>>,
    pending_save: Task<()>,
    position: DockPosition,
    width: Option<Pixels>,
}

impl DevPanel {
    pub async fn load(
        workspace: WeakEntity<Workspace>,
        mut cx: AsyncWindowContext,
    ) -> Result<Entity<Self>> {
        // Load per-project category overrides before building the panel.
        let override_key = workspace
            .read_with(&cx, |workspace, cx| override_key(workspace, cx))
            .ok()
            .flatten();
        let overrides = match override_key.clone() {
            Some(key) => {
                let kvp = cx.update(|_, cx| KeyValueStore::global(cx))?;
                cx.background_spawn(async move { kvp.read_kvp(&key) })
                    .await
                    .ok()
                    .flatten()
                    .and_then(|s| serde_json::from_str::<HashMap<String, Category>>(&s).ok())
                    .unwrap_or_default()
            }
            None => HashMap::default(),
        };

        workspace.update_in(&mut cx, |workspace, window, cx| {
            Self::new(workspace, override_key, overrides, window, cx)
        })
    }

    fn new(
        workspace: &mut Workspace,
        override_key: Option<String>,
        overrides: HashMap<String, Category>,
        _window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> Entity<Self> {
        let project = workspace.project().clone();
        let weak = workspace.weak_handle();
        cx.new(|cx| {
            let focus_handle = cx.focus_handle();
            let root = project
                .read(cx)
                .visible_worktrees(cx)
                .next()
                .map(|wt| wt.read(cx).abs_path().clone());
            let mut panel = Self {
                workspace: weak,
                project,
                focus_handle,
                root,
                runnables: Vec::new(),
                overrides,
                override_key,
                containers: Vec::new(),
                container_running: HashSet::default(),
                running: HashMap::default(),
                pending_save: Task::ready(()),
                position: DockPosition::Left,
                width: None,
            };
            panel.detect(cx);
            panel
        })
    }

    // ─── detection ───────────────────────────────────────────────

    /// Re-scan all supported project files and rebuild the runnable list.
    fn detect(&mut self, cx: &mut Context<Self>) {
        let Some(root) = self.root.clone() else {
            return;
        };
        let fs = self.project.read(cx).fs().clone();
        cx.spawn(async move |this, cx| {
            let mut runnables = Vec::new();

            if let Ok(text) = fs.load(&root.join("package.json")).await {
                let (pm, scripts) = parse_package_json(&text);
                for name in scripts {
                    let category = default_category(&name);
                    runnables.push(Runnable {
                        id: format!("npm:{name}"),
                        label: name.clone(),
                        program: pm.clone(),
                        args: vec!["run".into(), name],
                        default_category: category,
                    });
                }
            }

            if let Ok(text) = fs.load(&root.join("Makefile")).await {
                for target in parse_makefile_targets(&text) {
                    let category = default_category(&target);
                    runnables.push(Runnable {
                        id: format!("make:{target}"),
                        label: format!("make {target}"),
                        program: "make".into(),
                        args: vec![target],
                        default_category: category,
                    });
                }
            }

            if let Ok(text) = fs.load(&root.join("Procfile")).await {
                for (name, command) in parse_procfile(&text) {
                    // Procfile entries are always long-running processes.
                    runnables.push(Runnable {
                        id: format!("proc:{name}"),
                        label: name,
                        program: "sh".into(),
                        args: vec!["-lc".into(), command],
                        default_category: Category::Server,
                    });
                }
            }

            if let Ok(text) = fs.load(&root.join(".zed/tasks.json")).await {
                for (label, command, args) in parse_zed_tasks(&text) {
                    let category = default_category(&label);
                    runnables.push(Runnable {
                        id: format!("zed:{label}"),
                        label,
                        program: command,
                        args,
                        default_category: category,
                    });
                }
            }

            if let Ok(text) = fs.load(&root.join("Cargo.toml")).await {
                if cargo_has_package(&text) {
                    runnables.push(Runnable {
                        id: "cargo:run".into(),
                        label: "cargo run".into(),
                        program: "cargo".into(),
                        args: vec!["run".into()],
                        default_category: Category::Server,
                    });
                }
            }

            dedup_runnables(&mut runnables);

            let mut containers = Vec::new();
            for name in COMPOSE_FILENAMES {
                if let Ok(text) = fs.load(&root.join(name)).await {
                    containers = parse_compose_services(&text);
                    break;
                }
            }

            this.update(cx, |this, cx| {
                this.runnables = runnables;
                this.containers = containers;
                this.refresh_container_status(cx);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Probe `docker compose ps` and update which services are running.
    fn refresh_container_status(&mut self, cx: &mut Context<Self>) {
        let Some(root) = self.root.clone() else {
            return;
        };
        cx.spawn(async move |this, cx| {
            let mut command = new_command("docker");
            command
                .args(["compose", "ps", "--format", "json"])
                .current_dir(&root)
                .env_remove("LD_LIBRARY_PATH")
                .env_remove("LD_PRELOAD");
            let running = match command.output().await {
                Ok(output) => parse_running_services(&output.stdout),
                Err(_) => HashSet::default(),
            };
            this.update(cx, |this, cx| {
                this.container_running = running;
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn schedule_status_refresh(&self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            cx.background_executor()
                .timer(Duration::from_secs(2))
                .await;
            this.update(cx, |this, cx| this.refresh_container_status(cx))
                .ok();
        })
        .detach();
    }

    // ─── category / classification ───────────────────────────────

    fn category_of(&self, runnable: &Runnable) -> Category {
        self.overrides
            .get(&runnable.id)
            .copied()
            .unwrap_or(runnable.default_category)
    }

    fn runnables_in(&self, category: Category) -> Vec<Runnable> {
        self.runnables
            .iter()
            .filter(|r| self.category_of(r) == category)
            .cloned()
            .collect()
    }

    /// Flip a runnable between Server and Workflow and persist the choice.
    fn reclassify(&mut self, id: String, cx: &mut Context<Self>) {
        let Some(runnable) = self.runnables.iter().find(|r| r.id == id) else {
            return;
        };
        let target = self.category_of(runnable).opposite();
        if target == runnable.default_category {
            self.overrides.remove(&id);
        } else {
            self.overrides.insert(id, target);
        }
        self.save_overrides(cx);
        cx.notify();
    }

    fn save_overrides(&mut self, cx: &mut Context<Self>) {
        let Some(key) = self.override_key.clone() else {
            return;
        };
        let Ok(value) = serde_json::to_string(&self.overrides) else {
            return;
        };
        let kvp = KeyValueStore::global(cx);
        self.pending_save = cx.background_spawn(async move {
            kvp.write_kvp(key, value).await.ok();
        });
    }

    // ─── run actions ─────────────────────────────────────────────

    fn start_runnable(&mut self, id: String, window: &mut Window, cx: &mut Context<Self>) {
        let Some(runnable) = self.runnables.iter().find(|r| r.id == id).cloned() else {
            return;
        };
        let Some(task) = self.spawn_terminal(
            &runnable.id,
            runnable.label.clone(),
            &runnable.program,
            runnable.args.clone(),
            window,
            cx,
        ) else {
            return;
        };
        cx.spawn(async move |this, cx| {
            if let Ok(handle) = task.await {
                this.update(cx, |this, cx| {
                    this.running.insert(id, handle);
                    cx.notify();
                })
                .ok();
            }
        })
        .detach();
    }

    fn stop_runnable(&mut self, id: String, cx: &mut Context<Self>) {
        if let Some(handle) = self.running.remove(&id) {
            if let Some(terminal) = handle.upgrade() {
                terminal.update(cx, |terminal, _| terminal.input(vec![0x03]));
            }
        }
        cx.notify();
    }

    fn toggle_runnable(&mut self, id: String, window: &mut Window, cx: &mut Context<Self>) {
        if self.is_running(&id) {
            self.stop_runnable(id, cx);
        } else {
            self.start_runnable(id, window, cx);
        }
    }

    fn start_all_servers(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        for runnable in self.runnables_in(Category::Server) {
            if !self.is_running(&runnable.id) {
                self.start_runnable(runnable.id, window, cx);
            }
        }
    }

    fn stop_all_servers(&mut self, cx: &mut Context<Self>) {
        for id in self.running.keys().cloned().collect::<Vec<_>>() {
            self.stop_runnable(id, cx);
        }
    }

    fn toggle_container(&mut self, service: String, window: &mut Window, cx: &mut Context<Self>) {
        let up = !self.container_running.contains(&service);
        self.container_action(service, up, window, cx);
    }

    fn container_action(
        &mut self,
        service: String,
        up: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let (verb, label) = if up { ("up", "up") } else { ("stop", "down") };
        let mut args = vec!["compose".to_string(), verb.to_string()];
        if up {
            args.push("-d".to_string());
        }
        args.push(service.clone());
        if let Some(task) = self.spawn_terminal(
            &format!("compose-{label}-{service}"),
            format!("docker {label}: {service}"),
            "docker",
            args,
            window,
            cx,
        ) {
            task.detach();
        }
        self.schedule_status_refresh(cx);
    }

    fn containers_all(&mut self, up: bool, window: &mut Window, cx: &mut Context<Self>) {
        let (verb, extra): (&str, Vec<String>) = if up {
            ("up", vec!["-d".into()])
        } else {
            ("stop", vec![])
        };
        let mut args = vec!["compose".to_string(), verb.to_string()];
        args.extend(extra);
        if let Some(task) = self.spawn_terminal(
            &format!("compose-all-{verb}"),
            format!("docker compose {verb} (all)"),
            "docker",
            args,
            window,
            cx,
        ) {
            task.detach();
        }
        self.schedule_status_refresh(cx);
    }

    fn spawn_terminal(
        &self,
        id: &str,
        label: String,
        program: &str,
        args: Vec<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<Task<Result<WeakEntity<Terminal>>>> {
        let root = self.root.clone()?;
        let workspace = self.workspace.upgrade()?;
        let terminal_panel = workspace.read(cx).panel::<TerminalPanel>(cx)?;
        let spec = SpawnInTerminal {
            id: TaskId(id.to_string()),
            full_label: label.clone(),
            label,
            command: Some(program.to_string()),
            args,
            cwd: Some(root.to_path_buf()),
            reveal: RevealStrategy::Always,
            ..Default::default()
        };
        Some(terminal_panel.update(cx, |panel, cx| panel.spawn_task(&spec, window, cx)))
    }

    fn is_running(&self, id: &str) -> bool {
        self.running
            .get(id)
            .and_then(|handle| handle.upgrade())
            .is_some()
    }

    // ─── rendering ───────────────────────────────────────────────

    fn render_runnable_row(&self, runnable: &Runnable, cx: &Context<Self>) -> AnyElement {
        let id = runnable.id.clone();
        let is_server = self.category_of(runnable) == Category::Server;
        let running = self.is_running(&id);
        let reclassify_id = id.clone();
        let toggle_id = id.clone();

        let action_button = if is_server {
            toggle_button(
                "run",
                running,
                cx.listener(move |this, _, window, cx| {
                    this.toggle_runnable(toggle_id.clone(), window, cx)
                }),
            )
        } else {
            IconButton::new("run", IconName::PlayOutlined)
                .icon_size(IconSize::Small)
                .tooltip(Tooltip::text("Run"))
                .on_click(cx.listener(move |this, _, window, cx| {
                    this.start_runnable(toggle_id.clone(), window, cx)
                }))
        };

        let move_tip = if is_server {
            "Move to Scripts"
        } else {
            "Move to Servers"
        };

        h_flex()
            .id(SharedString::from(format!("row-{id}")))
            .w_full()
            .justify_between()
            .py_0p5()
            .child(
                h_flex()
                    .gap_1p5()
                    .when(is_server, |el| el.child(status_dot(running)))
                    .child(Label::new(runnable.label.clone()).size(LabelSize::Small)),
            )
            .child(
                h_flex()
                    .gap_0p5()
                    .child(
                        IconButton::new("reclassify", IconName::ArrowCircle)
                            .icon_size(IconSize::XSmall)
                            .tooltip(Tooltip::text(move_tip))
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.reclassify(reclassify_id.clone(), cx)
                            })),
                    )
                    .child(action_button),
            )
            .into_any_element()
    }

    fn render_container_row(&self, service: &str, cx: &Context<Self>) -> AnyElement {
        let name = service.to_string();
        let running = self.container_running.contains(service);
        h_flex()
            .id(SharedString::from(format!("container-{name}")))
            .w_full()
            .justify_between()
            .py_0p5()
            .child(
                h_flex()
                    .gap_1p5()
                    .child(status_dot(running))
                    .child(Label::new(service.to_string()).size(LabelSize::Small)),
            )
            .child(toggle_button(
                "container",
                running,
                cx.listener(move |this, _, window, cx| {
                    this.toggle_container(name.clone(), window, cx)
                }),
            ))
            .into_any_element()
    }

    fn render_section(
        &self,
        title: &'static str,
        actions: Vec<AnyElement>,
        rows: Vec<AnyElement>,
    ) -> impl IntoElement {
        v_flex()
            .w_full()
            .gap_0p5()
            .child(
                h_flex()
                    .w_full()
                    .justify_between()
                    .child(
                        Label::new(title)
                            .size(LabelSize::XSmall)
                            .color(Color::Muted),
                    )
                    .child(h_flex().gap_0p5().children(actions)),
            )
            .children(rows)
    }
}

/// Green dot when running, muted dot otherwise.
fn status_dot(running: bool) -> impl IntoElement {
    Icon::new(IconName::Circle)
        .size(IconSize::XSmall)
        .color(if running { Color::Success } else { Color::Muted })
}

/// Single toggle button: play when stopped, stop when running.
fn toggle_button(
    prefix: &'static str,
    running: bool,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut App) + 'static,
) -> IconButton {
    let (icon, tip) = if running {
        (IconName::Stop, "Stop")
    } else {
        (IconName::PlayFilled, "Start")
    };
    IconButton::new(prefix, icon)
        .icon_size(IconSize::Small)
        .icon_color(if running { Color::Error } else { Color::Success })
        .tooltip(Tooltip::text(tip))
        .on_click(on_click)
}

fn mini_button(
    id: &'static str,
    icon: IconName,
    tip: &'static str,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut App) + 'static,
) -> AnyElement {
    IconButton::new(id, icon)
        .icon_size(IconSize::XSmall)
        .tooltip(Tooltip::text(tip))
        .on_click(on_click)
        .into_any_element()
}

impl Focusable for DevPanel {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<PanelEvent> for DevPanel {}

impl Render for DevPanel {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let server_rows: Vec<AnyElement> = self
            .runnables_in(Category::Server)
            .iter()
            .map(|r| self.render_runnable_row(r, cx))
            .collect();
        let workflow_rows: Vec<AnyElement> = self
            .runnables_in(Category::Workflow)
            .iter()
            .map(|r| self.render_runnable_row(r, cx))
            .collect();
        let container_rows: Vec<AnyElement> = self
            .containers
            .iter()
            .map(|s| self.render_container_row(s, cx))
            .collect();

        let server_actions = vec![
            mini_button(
                "start-all-servers",
                IconName::PlayFilled,
                "Start all servers",
                cx.listener(|this, _, window, cx| this.start_all_servers(window, cx)),
            ),
            mini_button(
                "stop-all-servers",
                IconName::Stop,
                "Stop all servers",
                cx.listener(|this, _, _, cx| this.stop_all_servers(cx)),
            ),
        ];
        let container_actions = vec![
            mini_button(
                "up-all-containers",
                IconName::PlayFilled,
                "Up all (compose up -d)",
                cx.listener(|this, _, window, cx| this.containers_all(true, window, cx)),
            ),
            mini_button(
                "down-all-containers",
                IconName::Stop,
                "Stop all (compose stop)",
                cx.listener(|this, _, window, cx| this.containers_all(false, window, cx)),
            ),
        ];

        v_flex()
            .key_context("DevPanel")
            .track_focus(&self.focus_handle)
            .size_full()
            .child(
                h_flex()
                    .flex_none()
                    .w_full()
                    .px_2()
                    .py_1()
                    .justify_between()
                    .child(Label::new("Dev").size(LabelSize::Default))
                    .child(
                        IconButton::new("refresh", IconName::RotateCw)
                            .icon_size(IconSize::Small)
                            .tooltip(Tooltip::text("Refresh"))
                            .on_click(cx.listener(|this, _, _, cx| this.detect(cx))),
                    ),
            )
            .child(
                v_flex()
                    .id("dev-panel-scroll")
                    .flex_1()
                    .overflow_y_scroll()
                    .px_2()
                    .pb_2()
                    .gap_3()
                    .child(self.render_section("SERVERS", server_actions, server_rows))
                    .child(self.render_section("CONTAINERS", container_actions, container_rows))
                    .child(self.render_section("SCRIPTS", Vec::new(), workflow_rows)),
            )
    }
}

impl Panel for DevPanel {
    fn persistent_name() -> &'static str {
        "DevPanel"
    }

    fn panel_key() -> &'static str {
        "DevPanel"
    }

    fn position(&self, _window: &Window, _cx: &App) -> DockPosition {
        self.position
    }

    fn position_is_valid(&self, position: DockPosition) -> bool {
        matches!(position, DockPosition::Left | DockPosition::Right)
    }

    fn set_position(
        &mut self,
        position: DockPosition,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.position = position;
        cx.notify();
    }

    fn default_size(&self, _window: &Window, _cx: &App) -> Pixels {
        self.width.unwrap_or_else(|| px(280.))
    }

    fn icon(&self, _window: &Window, _cx: &App) -> Option<IconName> {
        Some(IconName::Server)
    }

    fn icon_tooltip(&self, _window: &Window, _cx: &App) -> Option<&'static str> {
        Some("Dev Panel")
    }

    fn toggle_action(&self) -> Box<dyn Action> {
        Box::new(ToggleFocus)
    }

    fn activation_priority(&self) -> u32 {
        5
    }
}

// ─── detection helpers (pure, unit-tested) ───────────────────────

const COMPOSE_FILENAMES: &[&str] = &[
    "docker-compose.yaml",
    "docker-compose.yml",
    "compose.yaml",
    "compose.yml",
];

/// First path segment of a name matching these → treated as a server.
const SERVER_HINTS: &[&str] = &["dev", "start", "serve", "watch", "run"];

fn default_category(name: &str) -> Category {
    let head = name.split([':', '-', ' ']).next().unwrap_or(name);
    if SERVER_HINTS.contains(&head) {
        Category::Server
    } else {
        Category::Workflow
    }
}

fn dedup_runnables(runnables: &mut Vec<Runnable>) {
    let mut seen = HashSet::default();
    runnables.retain(|r| seen.insert(r.id.clone()));
}

/// Parse `package.json`, returning the package manager and script names.
fn parse_package_json(text: &str) -> (String, Vec<String>) {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        return ("npm".to_string(), Vec::new());
    };
    let pm = value
        .get("packageManager")
        .and_then(|v| v.as_str())
        .map(package_manager_from_field)
        .unwrap_or("npm")
        .to_string();
    let scripts = value
        .get("scripts")
        .and_then(|s| s.as_object())
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_default();
    (pm, scripts)
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

/// Extract non-special target names from a Makefile.
fn parse_makefile_targets(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::default();
    for line in text.lines() {
        // Recipe lines are indented; target lines start at column 0.
        if line.starts_with([' ', '\t']) {
            continue;
        }
        let Some(idx) = line.find(':') else {
            continue;
        };
        // Skip `NAME := value` variable assignments.
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
            out.push(name.to_string());
        }
    }
    out
}

/// Parse `Procfile` lines into `(process_name, command)` pairs.
fn parse_procfile(text: &str) -> Vec<(String, String)> {
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

/// Parse `.zed/tasks.json` (lenient JSON — trailing commas allowed) into
/// `(label, command, args)` tuples.
fn parse_zed_tasks(text: &str) -> Vec<(String, String, Vec<String>)> {
    #[derive(Deserialize)]
    struct ZedTaskDef {
        label: String,
        command: String,
        #[serde(default)]
        args: Vec<String>,
    }
    serde_json_lenient::from_str::<Vec<ZedTaskDef>>(text)
        .map(|tasks| {
            tasks
                .into_iter()
                .map(|t| (t.label, t.command, t.args))
                .collect()
        })
        .unwrap_or_default()
}

fn cargo_has_package(text: &str) -> bool {
    text.lines().any(|l| l.trim_start().starts_with("[package]"))
}

/// Extract the `services:` keys from a docker-compose YAML document.
fn parse_compose_services(text: &str) -> Vec<String> {
    let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(text) else {
        return Vec::new();
    };
    value
        .get("services")
        .and_then(|s| s.as_mapping())
        .map(|map| {
            map.keys()
                .filter_map(|k| k.as_str().map(String::from))
                .collect()
        })
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
fn parse_running_services(stdout: &[u8]) -> HashSet<String> {
    let text = String::from_utf8_lossy(stdout);
    let mut running = HashSet::default();
    let mut consider = |row: ComposePsRow| {
        if let (Some(service), Some(state)) = (row.service, row.state) {
            let s = state.to_ascii_lowercase();
            if s.contains("running") || s.starts_with("up") {
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

/// KV key for a workspace's category overrides, based on its first worktree.
fn override_key(workspace: &Workspace, cx: &App) -> Option<String> {
    let root = workspace
        .project()
        .read(cx)
        .visible_worktrees(cx)
        .next()?
        .read(cx)
        .abs_path()
        .clone();
    Some(format!("dev_panel::overrides::{}", root.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scripts_and_package_manager() {
        let json = r#"{ "packageManager": "pnpm@11.13.0",
            "scripts": { "dev": "x", "build": "y", "dev:portal": "z" } }"#;
        let (pm, mut scripts) = parse_package_json(json);
        scripts.sort();
        assert_eq!(pm, "pnpm");
        assert_eq!(scripts, vec!["build", "dev", "dev:portal"]);
    }

    #[test]
    fn categorizes_by_head_segment() {
        assert_eq!(default_category("dev"), Category::Server);
        assert_eq!(default_category("dev:portal"), Category::Server);
        assert_eq!(default_category("run-dev"), Category::Server);
        assert_eq!(default_category("watch"), Category::Server);
        assert_eq!(default_category("build"), Category::Workflow);
        assert_eq!(default_category("test:unit"), Category::Workflow);
    }

    #[test]
    fn parses_makefile_targets_skipping_specials() {
        let mk = "\
.PHONY: all\n\
all: lint test\n\
run-dev:\n\t./run\n\
build-apk:\n\tflutter build\n\
VAR := value\n\
%.o: %.c\n";
        let targets = parse_makefile_targets(mk);
        assert!(targets.contains(&"all".to_string()));
        assert!(targets.contains(&"run-dev".to_string()));
        assert!(targets.contains(&"build-apk".to_string()));
        assert!(!targets.iter().any(|t| t.starts_with('.')));
        assert!(!targets.iter().any(|t| t.contains('%')));
        assert!(!targets.contains(&"VAR".to_string()));
    }

    #[test]
    fn parses_procfile_entries() {
        let pf = "web: cargo run --package collab serve all\n# comment\nworker: sh worker.sh\n";
        let entries = parse_procfile(pf);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "web");
        assert_eq!(entries[0].1, "cargo run --package collab serve all");
        assert_eq!(entries[1].0, "worker");
    }

    #[test]
    fn parses_zed_tasks_with_trailing_commas() {
        let tasks = "[\n  { \"label\": \"clippy\", \"command\": \"./script/clippy\", \"args\": [], },\n  { \"label\": \"run\", \"command\": \"cargo\", \"args\": [\"run\"], },\n]";
        let parsed = parse_zed_tasks(tasks);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].0, "clippy");
        assert_eq!(parsed[1].1, "cargo");
        assert_eq!(parsed[1].2, vec!["run".to_string()]);
    }

    #[test]
    fn detects_cargo_package() {
        assert!(cargo_has_package("[package]\nname = \"x\"\n"));
        assert!(!cargo_has_package("[workspace]\nmembers = []\n"));
    }

    #[test]
    fn parses_compose_services() {
        let yaml = "services:\n  postgres:\n    image: postgres\n  minio:\n    image: minio\n";
        let mut services = parse_compose_services(yaml);
        services.sort();
        assert_eq!(services, vec!["minio", "postgres"]);
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
    fn category_serde_roundtrips_lowercase() {
        let mut map = HashMap::default();
        map.insert("npm:dev".to_string(), Category::Workflow);
        let json = serde_json::to_string(&map).unwrap();
        assert!(json.contains("\"workflow\""));
        let back: HashMap<String, Category> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.get("npm:dev"), Some(&Category::Workflow));
    }
}
