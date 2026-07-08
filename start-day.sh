#!/usr/bin/env bash

PROJECT_OWNER="PortlandStatePowerLab"
PROJECT_NUMBER="18"

echo "=== Issues in Sandbox and Queue ==="

issues=$(gh project item-list 18 \
  --owner PortlandStatePowerLab \
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
    | select($status == "Sandbox" or $status == "Queue")
    | [.content.number, $status, .content.title, .content.url]
    | @tsv
  ')


if [ -z "$issues" ]; then
  echo "No issues found in Sandbox or Queue."
  exit 0
fi

printf "%-8s %-10s %s\n" "ISSUE" "COLUMN" "TITLE"
printf "%-8s %-10s %s\n" "-----" "------" "-----"

echo "$issues" | while IFS=$'\t' read -r number status title url; do
  printf "#%-7s %-10s %s\n" "$number" "$status" "$title"
done

echo ""
read -p "Enter the Issue Number you want to work on: " issue_num

if [ -z "$issue_num" ]; then
  echo "Cancelled."
  exit 0
fi

selected=$(echo "$issues" | awk -F $'\t' -v n="$issue_num" '$1 == n { print; exit }')

if [ -z "$selected" ]; then
  echo "Issue #$issue_num is not in Sandbox or Queue."
  exit 1
fi

title=$(echo "$selected" | cut -f3)

slug=$(echo "$title" \
  | tr "[:upper:]" "[:lower:]" \
  | sed "s/[^a-z0-9]/-/g" \
  | sed "s/-\{2,\}/-/g" \
  | sed "s/^-//; s/-$//" \
  | cut -c1-60)

branch_name="issue-${issue_num}-${slug}"

echo "Creating linked branch: $branch_name..."

gh issue develop "$issue_num" --checkout --name "$branch_name"

echo "Workspace ready! Happy coding."


