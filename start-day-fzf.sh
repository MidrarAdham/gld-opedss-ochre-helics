#!/usr/bin/env bash
set -euo pipefail

PROJECT_OWNER="PortlandStatePowerLab"
PROJECT_NUMBER="18"

# Optional: allow overriding the default base branch from the command line.
# Example: ./start-day-fzf.sh dev
BASE_BRANCH="${1:-}"

echo "=== Issues in Sandbox and Queue ==="

# Make sure required commands exist before doing any work.
for cmd in gh jq fzf git; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: '$cmd' is required but is not installed or not in PATH." >&2
    exit 1
  fi
done

issues=$(gh project item-list "$PROJECT_NUMBER" \
  --owner "$PROJECT_OWNER" \
  --limit 500 \
  --format json | jq -r '
    .items[]
    | select(.content.type == "Issue")
    | (
        .status
        // (.fieldValues[]? | select(.field.name == "Status") | .name)
        // (.fieldValues[]? | select(.field.name == "Status") | .value)
        // ""
      ) as $status
    | select($status == "Sandbox" or $status == "Queue" or $status == "Backlog")
    | [.content.number, $status, .content.title, .content.url]
    | @tsv
  ')

if [ -z "$issues" ]; then
  echo "No issues found in Sandbox or Queue."
  exit 0
fi

# Use fzf to interactively select an issue.
# Columns are: issue number, project status, title, URL.
selected=$(echo "$issues" | fzf \
  --delimiter=$'\t' \
  --with-nth=1,2,3 \
  --header=$'ISSUE\tSTATUS\tTITLE' \
  --prompt="Pick issue: " \
  --height=80% \
  --border \
  --preview='echo {} | awk -F "\t" "{print \"Issue: #\" \$1 \"\nStatus: \" \$2 \"\nTitle: \" \$3 \"\nURL: \" \$4}"' \
  --preview-window=down:6:wrap)

if [ -z "$selected" ]; then
  echo "Cancelled."
  exit 0
fi

issue_num=$(echo "$selected" | cut -f1)
status=$(echo "$selected" | cut -f2)
title=$(echo "$selected" | cut -f3)
url=$(echo "$selected" | cut -f4)

slug=$(echo "$title" \
  | tr "[:upper:]" "[:lower:]" \
  | sed "s/[^a-z0-9]/-/g" \
  | sed "s/-\{2,\}/-/g" \
  | sed "s/^-//; s/-$//" \
  | cut -c1-60)

branch_name="issue-${issue_num}-${slug}"

echo ""
echo "Selected issue: #$issue_num [$status] $title"
echo "Issue URL: $url"
echo "Creating linked branch: $branch_name..."

if [ -n "$BASE_BRANCH" ]; then
  gh issue develop "$issue_num" --checkout --name "$branch_name" --base "$BASE_BRANCH"
else
  gh issue develop "$issue_num" --checkout --name "$branch_name"
fi

echo ""
echo "Workspace ready! Happy coding."
