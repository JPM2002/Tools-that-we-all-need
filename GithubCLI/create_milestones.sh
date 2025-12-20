#!/usr/bin/env bash
set -e

# Load config
source ./config.env

if [ -z "$REPO" ]; then
  echo "REPO not set in config.env"
  exit 1
fi

echo "Creating milestones for repo: $REPO"
echo

for file in data/*.tsv; do
  echo "Processing $file"
  tail -n +2 "$file" | while IFS=$'\t' read -r title due_on description; do
    echo "  → $title"
    gh api -X POST "repos/$REPO/milestones" \
      -f title="$title" \
      -f due_on="$due_on" \
      -f description="$description"
  done
done

echo
echo "All milestones created."
