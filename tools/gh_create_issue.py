#!/usr/bin/env python3
"""
Create a GitHub issue, add it to a Projects v2 board, and set custom field values
(Status, Sprint, Estimated Time, or any other single-select/iteration/text/number
field configured on the board) in one shot.

Requires: gh CLI, authenticated (`gh auth status`).

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

Or just run it with no arguments (or --interactive) to be walked through owner, project,
repo, title, body, labels, assignees, and every board field via numbered menus:

    python3 tools/gh_create_issue.py
"""
import argparse
import json
import subprocess
import sys


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


def choose(prompt, options, allow_skip=True, allow_free_text=False):
    """Numbered single-choice menu. Returns the chosen string, or None if skipped."""
    if not options:
        return input(f"{prompt} (free text): ").strip() or None
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    if allow_skip:
        print("  0. (skip)")
    while True:
        raw = input("> ").strip()
        if allow_skip and raw in ("", "0"):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if allow_free_text and raw:
            return raw
        print("Invalid choice, try again.")


def choose_multi(prompt, options):
    """Comma-separated numbered multi-choice menu. Returns a list (possibly empty)."""
    if not options:
        return []
    print(f"{prompt} (comma-separated numbers, blank for none)")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    raw = input("> ").strip()
    if not raw:
        return []
    picks = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(options):
            picks.append(options[int(part) - 1])
    return picks


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


def interactive_main():
    print("=== Create a GitHub issue (interactive) ===\n")

    owner_guess = choose("Project owner", ["MidrarAdham", "PortlandStatePowerLab"], allow_free_text=True) \
        or ask("Project owner (user or org login)", required=True)

    projects = list_projects(owner_guess)
    if not projects:
        sys.exit(f"No projects found for owner {owner_guess!r}.")
    proj_labels = [f"#{p['number']} {p['title']} ({p['items']['totalCount']} items)" for p in projects]
    picked = choose("\nPick a project:", proj_labels, allow_skip=False)
    project = projects[proj_labels.index(picked)]
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
    p.add_argument("-i", "--interactive", action="store_true", help="walk through prompts instead of using flags")
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

    if args.interactive or len(sys.argv) == 1:
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
