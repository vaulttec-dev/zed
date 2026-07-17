//! Dev Panel — a left-dock panel that auto-detects a project's runnable tasks
//! (dev servers and one-shot scripts) and Docker services, and lets you
//! start/stop them with one click.
//!
//! Detection is file-native and tree-wide: it walks every visible worktree
//! (respecting `.gitignore`) and reads whatever manifests the project already
//! has, in any subdirectory, across stacks:
//! - `package.json` scripts (Node, per package — monorepos included)
//! - `Cargo.toml` (`cargo run`, one per `[[bin]]`)
//! - Go `main` packages (`go run .`)
//! - `pyproject.toml` scripts, Django `manage.py`, bare `main.py`/`app.py`
//! - `Makefile` targets, `justfile` recipes, `Taskfile.yml` tasks
//! - `Procfile`/`Procfile.*` process entries (always servers)
//! - `composer.json` scripts (PHP), `*.csproj`/`*.fsproj` (.NET)
//! - `.zed/tasks.json` native Zed tasks
//! - `docker-compose`/`compose` services (containers), with live status from
//!   `docker compose ps`
//!
//! The list refreshes itself: it rescans (debounced) whenever the project's
//! files change, so there is no manual refresh. Each runnable is classified as
//! **Server** or **Script** automatically (see [`classification`]). For the rare
//! genuine miss, the escape hatch is declarative and checked-in: tag a task in
//! `.zed/tasks.json` with `dev:server` or `dev:script`. Everything runs in Zed's
//! own terminal panel.

mod classification;
mod detection;
mod parsers;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use collections::{HashMap, HashSet};
use gpui::{
    Action, AnyElement, App, AsyncWindowContext, Context, Entity, EventEmitter, FocusHandle,
    Focusable, IntoElement, ParentElement, Pixels, Render, Styled, Subscription, Task, WeakEntity,
    Window, actions, px,
};
use project::Project;
use task::{RevealStrategy, SpawnInTerminal, TaskId};
use terminal::Terminal;
use terminal_view::terminal_panel::TerminalPanel;
use ui::{Tooltip, prelude::*};
use util::command::new_command;
use workspace::{
    Workspace,
    dock::{DockPosition, Panel, PanelEvent},
};

use classification::Category;
use detection::{
    Candidate, Container, MAX_MANIFEST_DEPTH, Runnable, SKIP_DIRS, ScanIndex, candidate_kind,
    scan_candidates,
};
use parsers::parse_running_services;

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

pub struct DevPanel {
    workspace: WeakEntity<Workspace>,
    project: Entity<Project>,
    focus_handle: FocusHandle,
    runnables: Vec<Runnable>,
    containers: Vec<Container>,
    /// Ids of containers currently reported as running.
    container_running: HashSet<String>,
    /// Live terminals for started servers, keyed by runnable id.
    running: HashMap<String, WeakEntity<Terminal>>,
    /// Ids whose start task is in flight (the terminal handle hasn't resolved
    /// yet), so a rapid second click can't launch a duplicate.
    starting: HashSet<String>,
    /// Coalesces file-change events into a single rescan.
    debounce_task: Task<()>,
    /// The in-flight tree scan (cancelled if a newer one starts).
    scan_task: Task<()>,
    /// The in-flight container-status probe.
    status_task: Task<()>,
    /// The periodic container-status poll.
    poll_task: Task<()>,
    _subscriptions: Vec<Subscription>,
    position: DockPosition,
    width: Option<Pixels>,
}

impl DevPanel {
    pub async fn load(
        workspace: WeakEntity<Workspace>,
        mut cx: AsyncWindowContext,
    ) -> Result<Entity<Self>> {
        workspace.update_in(&mut cx, |workspace, window, cx| Self::new(workspace, window, cx))
    }

    fn new(
        workspace: &mut Workspace,
        _window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> Entity<Self> {
        let project = workspace.project().clone();
        let weak = workspace.weak_handle();
        cx.new(|cx| {
            let focus_handle = cx.focus_handle();
            let subscription = cx.subscribe(
                &project,
                |this: &mut Self, _project, event: &project::Event, cx| {
                    if matches!(
                        event,
                        project::Event::WorktreeUpdatedEntries(..)
                            | project::Event::WorktreeAdded(..)
                            | project::Event::WorktreeRemoved(..)
                    ) {
                        this.schedule_detect(cx);
                    }
                },
            );
            let mut panel = Self {
                workspace: weak,
                project,
                focus_handle,
                runnables: Vec::new(),
                containers: Vec::new(),
                container_running: HashSet::default(),
                running: HashMap::default(),
                starting: HashSet::default(),
                debounce_task: Task::ready(()),
                scan_task: Task::ready(()),
                status_task: Task::ready(()),
                poll_task: Task::ready(()),
                _subscriptions: vec![subscription],
                position: DockPosition::Left,
                width: None,
            };
            panel.detect(cx);
            panel.start_status_polling(cx);
            panel
        })
    }

    // ─── detection ───────────────────────────────────────────────

    /// Debounced rescan, so a burst of file-change events triggers one scan.
    fn schedule_detect(&mut self, cx: &mut Context<Self>) {
        self.debounce_task = cx.spawn(async move |this, cx| {
            cx.background_executor()
                .timer(Duration::from_millis(500))
                .await;
            this.update(cx, |this, cx| this.detect(cx)).ok();
        });
    }

    /// Walk every visible worktree and rebuild the runnable/container lists.
    fn detect(&mut self, cx: &mut Context<Self>) {
        let worktrees: Vec<_> = self.project.read(cx).visible_worktrees(cx).collect();
        let mut candidates: Vec<Candidate> = Vec::new();
        let mut scan = ScanIndex::default();

        for worktree in worktrees {
            let snapshot = worktree.read(cx);
            let root = snapshot.abs_path().to_path_buf();
            for entry in snapshot.files(false, 0) {
                let path = &entry.path;
                if path.components().count() > MAX_MANIFEST_DEPTH {
                    continue;
                }
                if path.components().any(|c| SKIP_DIRS.contains(&c)) {
                    continue;
                }
                let Some(file_name) = path.file_name() else {
                    continue;
                };
                let extension = path.extension();
                let dir_abs: Arc<Path> = path
                    .parent()
                    .map(|p| Arc::from(snapshot.absolutize(p).as_path()))
                    .unwrap_or_else(|| Arc::from(root.as_path()));

                // Index sibling markers used later for classification decisions.
                match file_name {
                    "pnpm-lock.yaml" => scan.set_pm(&dir_abs, "pnpm"),
                    "yarn.lock" => scan.set_pm(&dir_abs, "yarn"),
                    "bun.lockb" | "bun.lock" => scan.set_pm(&dir_abs, "bun"),
                    "package-lock.json" => scan.set_pm(&dir_abs, "npm"),
                    "poetry.lock" => {
                        scan.poetry_dirs.insert(dir_abs.to_path_buf());
                    }
                    "pyproject.toml" => {
                        scan.pyproject_dirs.insert(dir_abs.to_path_buf());
                    }
                    "manage.py" => {
                        scan.managepy_dirs.insert(dir_abs.to_path_buf());
                    }
                    _ => {}
                }

                let parent_file_name = path.parent().and_then(|p| p.file_name());
                let parent_has_cmd = path
                    .parent()
                    .is_some_and(|p| p.components().any(|c| c == "cmd"));
                let kind = candidate_kind(file_name, extension, parent_file_name, parent_has_cmd);
                if let Some(kind) = kind {
                    candidates.push(Candidate {
                        kind,
                        abs: snapshot.absolutize(path),
                        dir: dir_abs,
                        file_name: file_name.to_string(),
                    });
                }
            }
        }

        let fs = self.project.read(cx).fs().clone();
        self.scan_task = cx.spawn(async move |this, cx| {
            let (runnables, containers) = scan_candidates(fs, candidates, scan).await;
            this.update(cx, |this, cx| {
                this.runnables = runnables;
                this.containers = containers;
                // Drop status for containers that no longer exist.
                let live: HashSet<String> = this.containers.iter().map(|c| c.id.clone()).collect();
                this.container_running.retain(|id| live.contains(id));
                this.refresh_container_status(cx);
                cx.notify();
            })
            .ok();
        });
    }

    /// Distinct compose invocations `(dir, explicit_file_name)` to probe/act on.
    fn compose_invocations(&self) -> Vec<(Arc<Path>, Option<String>)> {
        let mut seen = HashSet::default();
        let mut out = Vec::new();
        for container in &self.containers {
            if seen.insert(container.compose_file.to_path_buf()) {
                let explicit = container.explicit.then(|| container.file_name.clone());
                out.push((container.dir.clone(), explicit));
            }
        }
        out
    }

    /// Probe `docker compose ps` for every compose file and update status.
    fn refresh_container_status(&mut self, cx: &mut Context<Self>) {
        // Group each container's (id, service) under its compose file.
        let mut groups: HashMap<PathBuf, (Arc<Path>, Option<String>, Vec<(String, String)>)> =
            HashMap::default();
        for container in &self.containers {
            let entry = groups
                .entry(container.compose_file.to_path_buf())
                .or_insert_with(|| {
                    (
                        container.dir.clone(),
                        container.explicit.then(|| container.file_name.clone()),
                        Vec::new(),
                    )
                });
            entry.2.push((container.id.clone(), container.service.clone()));
        }

        if groups.is_empty() {
            self.container_running.clear();
            return;
        }
        let jobs: Vec<(Arc<Path>, Option<String>, Vec<(String, String)>)> =
            groups.into_values().collect();

        self.status_task = cx.spawn(async move |this, cx| {
            let mut running_ids = HashSet::default();
            for (dir, explicit, rows) in jobs {
                // No explicit `-p`: we let docker infer the project name from the
                // working dir (its default), so status stays consistent with
                // stacks started outside the panel. The trade-off is that two
                // compose dirs sharing a basename would share a project — a rare
                // case docker itself has the same way.
                let mut command = new_command("docker");
                command.arg("compose");
                if let Some(file) = &explicit {
                    command.args(["-f", file]);
                }
                command
                    .args(["ps", "--format", "json"])
                    .current_dir(&*dir)
                    .env_remove("LD_LIBRARY_PATH")
                    .env_remove("LD_PRELOAD");
                if let Ok(output) = command.output().await {
                    let running_services = parse_running_services(&output.stdout);
                    for (id, service) in rows {
                        if running_services.contains(&service) {
                            running_ids.insert(id);
                        }
                    }
                }
            }
            this.update(cx, |this, cx| {
                this.container_running = running_ids;
                cx.notify();
            })
            .ok();
        });
    }

    /// Poll container status every few seconds while the panel is alive.
    fn start_status_polling(&mut self, cx: &mut Context<Self>) {
        self.poll_task = cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_secs(5))
                    .await;
                if this
                    .update(cx, |this, cx| {
                        if !this.containers.is_empty() {
                            this.refresh_container_status(cx);
                        }
                    })
                    .is_err()
                {
                    break;
                }
            }
        });
    }

    fn schedule_status_refresh(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            cx.background_executor()
                .timer(Duration::from_secs(2))
                .await;
            this.update(cx, |this, cx| this.refresh_container_status(cx))
                .ok();
        })
        .detach();
    }

    // ─── categories ──────────────────────────────────────────────

    fn runnables_in(&self, category: Category) -> Vec<Runnable> {
        self.runnables
            .iter()
            .filter(|r| r.category == category)
            .cloned()
            .collect()
    }

    // ─── run actions ─────────────────────────────────────────────

    fn start_runnable(&mut self, id: String, window: &mut Window, cx: &mut Context<Self>) {
        if self.starting.contains(&id) {
            return;
        }
        let Some(runnable) = self.runnables.iter().find(|r| r.id == id).cloned() else {
            return;
        };
        let Some(task) = self.spawn_terminal(
            &runnable.id,
            runnable.label.to_string(),
            &runnable.program,
            runnable.args.clone(),
            &runnable.cwd,
            window,
            cx,
        ) else {
            return;
        };
        self.starting.insert(id.clone());
        cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| {
                this.starting.remove(&id);
                if let Ok(handle) = result {
                    this.running.insert(id, handle);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn stop_runnable(&mut self, id: String, cx: &mut Context<Self>) {
        self.starting.remove(&id);
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

    fn toggle_container(&mut self, id: String, window: &mut Window, cx: &mut Context<Self>) {
        let up = !self.container_running.contains(&id);
        let Some(container) = self.containers.iter().find(|c| c.id == id).cloned() else {
            return;
        };
        self.container_action(&container, up, window, cx);
    }

    fn container_action(
        &mut self,
        container: &Container,
        up: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let (verb, label) = if up { ("up", "up") } else { ("stop", "down") };
        let mut args = vec!["compose".to_string()];
        if container.explicit {
            args.push("-f".into());
            args.push(container.file_name.clone());
        }
        args.push(verb.into());
        if up {
            args.push("-d".into());
        }
        args.push(container.service.clone());
        if let Some(task) = self.spawn_terminal(
            &format!("compose-{label}-{}", container.id),
            format!("docker {label}: {}", container.service),
            "docker",
            args,
            &container.dir,
            window,
            cx,
        ) {
            task.detach();
        }
        self.schedule_status_refresh(cx);
    }

    fn containers_all(&mut self, up: bool, window: &mut Window, cx: &mut Context<Self>) {
        let verb = if up { "up" } else { "stop" };
        for (dir, explicit) in self.compose_invocations() {
            let mut args = vec!["compose".to_string()];
            if let Some(file) = &explicit {
                args.push("-f".into());
                args.push(file.clone());
            }
            args.push(verb.into());
            if up {
                args.push("-d".into());
            }
            if let Some(task) = self.spawn_terminal(
                &format!("compose-all-{verb}-{}", dir.display()),
                format!("docker compose {verb}"),
                "docker",
                args,
                &dir,
                window,
                cx,
            ) {
                task.detach();
            }
        }
        self.schedule_status_refresh(cx);
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_terminal(
        &self,
        id: &str,
        label: String,
        program: &str,
        args: Vec<String>,
        cwd: &Path,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<Task<Result<WeakEntity<Terminal>>>> {
        let workspace = self.workspace.upgrade()?;
        let terminal_panel = workspace.read(cx).panel::<TerminalPanel>(cx)?;
        let spec = SpawnInTerminal {
            id: TaskId(id.to_string()),
            full_label: label.clone(),
            label,
            command: Some(program.to_string()),
            args,
            cwd: Some(cwd.to_path_buf()),
            reveal: RevealStrategy::Always,
            ..Default::default()
        };
        Some(terminal_panel.update(cx, |panel, cx| panel.spawn_task(&spec, window, cx)))
    }

    fn is_running(&self, id: &str) -> bool {
        self.starting.contains(id)
            || self
                .running
                .get(id)
                .and_then(|handle| handle.upgrade())
                .is_some()
    }

    // ─── rendering ───────────────────────────────────────────────

    fn render_runnable_row(&self, runnable: &Runnable, cx: &Context<Self>) -> AnyElement {
        let id = runnable.id.clone();
        let is_server = runnable.category == Category::Server;
        let running = self.is_running(&id);
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
            .child(action_button)
            .into_any_element()
    }

    fn render_container_row(&self, container: &Container, cx: &Context<Self>) -> AnyElement {
        let id = container.id.clone();
        let running = self.container_running.contains(&id);
        h_flex()
            .id(SharedString::from(format!("container-{id}")))
            .w_full()
            .justify_between()
            .py_0p5()
            .child(
                h_flex()
                    .gap_1p5()
                    .child(status_dot(running))
                    .child(Label::new(container.label.clone()).size(LabelSize::Small)),
            )
            .child(toggle_button(
                "container",
                running,
                cx.listener(move |this, _, window, cx| {
                    this.toggle_container(id.clone(), window, cx)
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
        let script_rows: Vec<AnyElement> = self
            .runnables_in(Category::Script)
            .iter()
            .map(|r| self.render_runnable_row(r, cx))
            .collect();
        let container_rows: Vec<AnyElement> = self
            .containers
            .iter()
            .map(|c| self.render_container_row(c, cx))
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
                    .child(Label::new("Dev").size(LabelSize::Default)),
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
                    .child(self.render_section("SCRIPTS", Vec::new(), script_rows)),
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
