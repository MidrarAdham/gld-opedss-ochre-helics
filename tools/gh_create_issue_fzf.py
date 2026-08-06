#!/usr/bin/env python3
"""
Create a GitHub issue, add it to a Projects v2 board, or browse the issues already
on a board and change their custom field values.

Requires: gh CLI, fzf, and an authenticated GitHub session (`gh auth status`).

Examples
--------
Create an issue on the org repo, add it to project #18, set status/sprint/estimate:

    python3 tools/gh_create_issue.py \\
        --repo PortlandStatePowerLab/midrar-gld-opedss-ochre-helics-2025 \\
        --project 18 --project-owner PortlandStatePowerLab \\
        --title "Refactor the OLS module" \\
        --body "Split fitting and prediction into separate classes." \\
        --status "Backlog" --sprint "Sprint 3" --estimate "two hours" \\
        --label enhancement --assignee MidrarAdham

List the field/option names available on a board (useful before setting values):

    python3 tools/gh_create_issue.py --project 18 --project-owner PortlandStatePowerLab --list-fields

Run it with no arguments to choose between creating an issue and viewing/editing
existing project issues. Use --interactive to go directly to issue creation:

    python3 tools/gh_create_issue.py
"""
import argparse
import json
import shutil
import subprocess
import sys


SUPPORTED_FIELD_TYPES = (
    "ProjectV2SingleSelectField",
    "ProjectV2IterationField",
    "ProjectV2Field",
)


def run_gh(args, input_text=None):
    result = subprocess.run(
        ["gh"] + args, capture_output=True, text=True, input=input_text
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def get_project_id(owner, number):
    out = run_gh(["project", "view", str(number), "--owner", owner, "--format", "json"])
    return json.loads(out)["id"]


def get_fields(owner, number):
    """Returns {field_name: {"id": ..., "type": ..., "options": {name: id}, "iterations": {name: id}}}"""
    out = run_gh(["project", "field-list", str(number), "--owner", owner, "--format", "json"])
    fields = {}
    for f in json.loads(out)["fields"]:
        entry = {"id": f["id"], "type": f.get("type", "")}
        entry["options"] = {o["name"]: o["id"] for o in f.get("options", [])}
        entry["iterations"] = {}
        if entry["type"] == "ProjectV2IterationField":
            gql = run_gh([
                "api", "graphql", "-f",
                f'query={{ node(id: "{f["id"]}") {{ ... on ProjectV2IterationField '
                f'{{ configuration {{ iterations {{ id title }} completedIterations {{ id title }} }} }} }} }}'
            ])
            config = json.loads(gql)["data"]["node"]["configuration"]
            for it in config["iterations"] + config["completedIterations"]:
                entry["iterations"][it["title"]] = it["id"]
        fields[f["name"]] = entry
    return fields


def list_projects(owner):
    out = run_gh(["project", "list", "--owner", owner, "--format", "json"])
    return json.loads(out)["projects"]


def list_project_items(owner, number, limit=200, field_names=None):
    """Return project items as emitted by ``gh project item-list``."""
    args = [
        "project", "item-list", str(number), "--owner", owner,
        "--format", "json", "--limit", str(limit),
    ]
    for name in field_names or []:
        args += ["--field", name]
    out = run_gh(args)
    return json.loads(out).get("items", [])


def list_repos(owner, limit=50):
    out = run_gh(["repo", "list", owner, "--json", "nameWithOwner", "--limit", str(limit)])
    return [r["nameWithOwner"] for r in json.loads(out)]


def list_labels(repo):
    out = run_gh(["label", "list", "--repo", repo, "--json", "name", "--limit", "100"])
    return [l["name"] for l in json.loads(out)]


def create_issue(repo, title, body, labels, assignees):
    args = ["issue", "create", "--repo", repo, "--title", title, "--body", body or ""]
    for l in labels or []:
        args += ["--label", l]
    for a in assignees or []:
        args += ["--assignee", a]
    url = run_gh(args)
    return url.strip()


def add_to_project(project_number, project_owner, issue_url):
    out = run_gh([
        "project", "item-add", str(project_number),
        "--owner", project_owner, "--url", issue_url, "--format", "json",
    ])
    return json.loads(out)["id"]


def set_field(item_id, project_id, field, value):
    args = ["project", "item-edit", "--id", item_id, "--project-id", project_id, "--field-id", field["id"]]
    if field["type"] == "ProjectV2SingleSelectField":
        if value not in field["options"]:
            raise ValueError(f"Unknown option {value!r}. Choices: {list(field['options'])}")
        args += ["--single-select-option-id", field["options"][value]]
    elif field["type"] == "ProjectV2IterationField":
        if value not in field["iterations"]:
            raise ValueError(f"Unknown iteration {value!r}. Choices: {list(field['iterations'])}")
        args += ["--iteration-id", field["iterations"][value]]
    elif field["type"] == "ProjectV2Field":
        # plain text/number/date field - try text first
        args += ["--text", value]
    else:
        raise ValueError(f"Unsupported field type: {field['type']}")
    run_gh(args)


def clear_field(item_id, project_id, field):
    run_gh([
        "project", "item-edit", "--id", item_id,
        "--project-id", project_id, "--field-id", field["id"], "--clear",
    ])


def ensure_command(command):
    """Exit with a clear message when a required command is unavailable."""
    if shutil.which(command) is None:
        sys.exit(
            f"Required command {command!r} was not found in PATH. "
            f"Install it before running this script."
        )


def run_fzf(options, prompt, multi=False):
    """Run fzf and return the selected value(s). Escape means no selection."""
    if not options:
        return [] if multi else None

    args = [
        "fzf",
        "--prompt", f"{prompt}> ",
        "--height", "40%",
        "--layout", "reverse",
        "--border",
    ]
    if multi:
        args.append("--multi")

    result = subprocess.run(
        args,
        input="\n".join(options) + "\n",
        text=True,
        stdout=subprocess.PIPE,
    )

    if result.returncode in (1, 130):
        return [] if multi else None
    if result.returncode != 0:
        raise RuntimeError(f"fzf failed with exit code {result.returncode}")

    selected = [line for line in result.stdout.splitlines() if line]
    return selected if multi else (selected[0] if selected else None)


def choose(prompt, options, allow_skip=True, allow_free_text=False):
    """Single-choice fzf menu. Returns the chosen string or None."""
    if not options:
        return input(f"{prompt} (free text): ").strip() or None

    selected = run_fzf(options, prompt)
    if selected is not None:
        return selected

    if not allow_skip:
        print("A selection is required.")
        return choose(prompt, options, allow_skip=False, allow_free_text=allow_free_text)

    if allow_free_text:
        return input(f"{prompt} (free text, blank to skip): ").strip() or None

    return None


def choose_multi(prompt, options):
    """Multi-choice fzf menu. Tab selects multiple entries; Escape skips."""
    return run_fzf(options, prompt, multi=True)


def ask(prompt, default=None, required=False):
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        if not raw and required:
            print("This is required.")
            continue
        return raw


def choose_owner_and_project():
    """Interactively select a project owner and one of their projects."""
    owner = choose(
        "Project owner",
        ["MidrarAdham", "PortlandStatePowerLab"],
        allow_free_text=True,
    ) or ask("Project owner (user or org login)", required=True)

    projects = list_projects(owner)
    if not projects:
        sys.exit(f"No projects found for owner {owner!r}.")
    labels = [
        f"#{p['number']} {p['title']} ({p['items']['totalCount']} items)"
        for p in projects
    ]
    picked = choose("\nPick a project:", labels, allow_skip=False)
    return owner, projects[labels.index(picked)]


def item_content(item):
    """Normalize the content object from different gh CLI versions."""
    return item.get("content") or {}


def item_label(item):
    content = item_content(item)
    title = content.get("title") or item.get("title") or "(untitled draft)"
    number = content.get("number")
    repo = content.get("repository") or ""
    prefix = f"{repo}#{number}" if repo and number is not None else repo
    return f"{prefix}  {title}" if prefix else title


def current_item_fields(item):
    """Extract field values included by gh, excluding content metadata."""
    ignored = {"id", "content", "title", "type", "url", "repository", "number", "body"}
    values = {}
    for name, value in item.items():
        if name not in ignored and value not in (None, "", []):
            values[name] = value
    return values


def interactive_edit_items():
    ensure_command("gh")
    ensure_command("fzf")
    print("=== View and edit GitHub project issues ===\n")

    owner, project = choose_owner_and_project()
    fields = get_fields(owner, project["number"])
    editable = {name: field for name, field in fields.items()
                if field["type"] in SUPPORTED_FIELD_TYPES}
    if not editable:
        print("This project has no editable custom fields.")
        return
    project_id = project["id"]

    while True:
        items = [item for item in list_project_items(
                    owner, project["number"], field_names=editable)
                 if item_content(item).get("type") == "Issue"]
        if not items:
            print("No issues were found in this project.")
            return

        labels = [item_label(item) for item in items]
        picked = choose("\nIssue (Escape to finish)", labels)
        if picked is None:
            return
        item = items[labels.index(picked)]
        content = item_content(item)

        print(f"\n{item_label(item)}")
        if content.get("url"):
            print(content["url"])
        if content.get("body"):
            print(f"\n{content['body']}")
        values = current_item_fields(item)
        print("\nCurrent fields:")
        if values:
            for name, value in values.items():
                print(f"  {name}: {value}")
        else:
            print("  (none)")

        while True:
            field_name = choose("\nField to change (Escape for issue list)", list(editable))
            if field_name is None:
                break
            field = editable[field_name]
            choices = list(field["options"]) or list(field["iterations"])
            if choices:
                choice = choose(
                    f"{field_name} (Escape to cancel)",
                    choices + ["<clear value>"],
                )
            else:
                choice = ask(f"{field_name} (blank cancels)", default="") or None
            if choice is None:
                continue
            if choice == "<clear value>":
                clear_field(item["id"], project_id, field)
                print(f"  Cleared {field_name}")
            else:
                set_field(item["id"], project_id, field, choice)
                print(f"  {field_name} = {choice}")


def interactive_main():
    ensure_command("gh")
    ensure_command("fzf")

    print("=== Create a GitHub issue (interactive with fzf) ===\n")

    owner_guess, project = choose_owner_and_project()
    project_number = project["number"]

    repos = list_repos(owner_guess)
    repo = choose("\nPick a repo for the issue:", repos, allow_skip=False, allow_free_text=True)

    print()
    title = ask("Issue title", required=True)
    body = ask("Issue body (optional)", default="")

    labels = choose_multi("\nLabels", list_labels(repo))
    assignees_raw = ask("\nAssignees (comma-separated logins, optional)", default="")
    assignees = [a.strip() for a in assignees_raw.split(",") if a.strip()]

    fields = get_fields(owner_guess, project_number)
    settable = {name: f for name, f in fields.items()
                if f["type"] in ("ProjectV2SingleSelectField", "ProjectV2IterationField")}

    to_set = {}
    print("\n=== Board fields ===")
    for name, f in settable.items():
        opts = list(f["options"]) or list(f["iterations"])
        val = choose(f"\n{name}", opts)
        if val:
            to_set[name] = val

    print("\n=== Summary ===")
    print(f"Repo:      {repo}")
    print(f"Title:     {title}")
    print(f"Project:   #{project_number} ({project['title']}) owned by {owner_guess}")
    print(f"Labels:    {labels or '(none)'}")
    print(f"Assignees: {assignees or '(none)'}")
    for name, val in to_set.items():
        print(f"{name}: {val}")
    if ask("\nCreate this issue? (y/n)", default="y").lower() not in ("y", "yes"):
        print("Aborted.")
        return

    issue_url = create_issue(repo, title, body, labels, assignees)
    print(f"\nCreated: {issue_url}")

    project_id = project["id"]
    item_id = add_to_project(project_number, owner_guess, issue_url)
    print(f"Added to project #{project_number}")

    for name, val in to_set.items():
        set_field(item_id, project_id, fields[name], val)
        print(f"  {name} = {val}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--interactive", action="store_true", help="select values with fzf instead of using flags")
    p.add_argument("--edit", action="store_true",
                   help="interactively view project issues and edit their fields")
    p.add_argument("--repo", help="owner/repo to create the issue in (required unless --list-fields)")
    p.add_argument("--title", help="issue title")
    p.add_argument("--body", default="", help="issue body")
    p.add_argument("--label", action="append", dest="labels", help="repeatable")
    p.add_argument("--assignee", action="append", dest="assignees", help="repeatable")
    p.add_argument("--project", type=int, help="project number, e.g. 18")
    p.add_argument("--project-owner", help="project owner login (user or org)")
    p.add_argument("--status", help="Status field value, e.g. Backlog")
    p.add_argument("--sprint", help="Sprint/iteration field value, e.g. 'Sprint 3'")
    p.add_argument("--estimate", help="Estimated Time field value, e.g. 'two hours'")
    p.add_argument("--field", action="append", default=[], metavar="NAME=VALUE",
                    help="set any other board field by name, repeatable")
    p.add_argument("--list-fields", action="store_true", help="print available fields/options and exit")
    args = p.parse_args()

    if args.edit:
        interactive_edit_items()
        return

    if len(sys.argv) == 1:
        ensure_command("gh")
        ensure_command("fzf")
        action = choose("What would you like to do?", [
            "Create a new issue",
            "View and edit project issues",
        ], allow_skip=False)
        if action == "View and edit project issues":
            interactive_edit_items()
        else:
            interactive_main()
        return

    if args.interactive:
        interactive_main()
        return

    if not args.project or not args.project_owner:
        p.error("--project and --project-owner are required (or run with no arguments for --interactive)")

    fields = get_fields(args.project_owner, args.project)

    if args.list_fields:
        for name, f in fields.items():
            choices = list(f["options"]) or list(f["iterations"]) or "(free text)"
            print(f"{name} [{f['type']}]: {choices}")
        return

    if not args.repo or not args.title:
        p.error("--repo and --title are required unless --list-fields is given")

    issue_url = create_issue(args.repo, args.title, args.body, args.labels, args.assignees)
    print(f"Created: {issue_url}")

    project_id = get_project_id(args.project_owner, args.project)
    item_id = add_to_project(args.project, args.project_owner, issue_url)
    print(f"Added to project #{args.project}")

    to_set = {}
    if args.status:
        to_set["Status"] = args.status
    if args.sprint:
        to_set["Sprint"] = args.sprint
    if args.estimate:
        to_set["Estimated Time"] = args.estimate
    for kv in args.field:
        name, _, value = kv.partition("=")
        to_set[name] = value

    for name, value in to_set.items():
        if name not in fields:
            print(f"  WARNING: no field named {name!r} on this board, skipping", file=sys.stderr)
            continue
        set_field(item_id, project_id, fields[name], value)
        print(f"  {name} = {value}")


if __name__ == "__main__":
    main()
