#!/bin/bash
# Ralph Wiggum Loop for Bolt
# Run this overnight to autonomously improve the codebase

cd "$(dirname "$0")"

echo "Starting Ralph Wiggum loop on bolt..."
echo "Press Ctrl+C to stop"
echo ""

iteration=0
while :; do
    iteration=$((iteration + 1))
    echo "=== Iteration $iteration - $(date) ==="

    # Feed the prompt to claude
    cat RALPH.md | claude

    echo ""
    echo "Sleeping 5 seconds before next iteration..."
    sleep 5
done
