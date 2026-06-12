# Git Workflow

## Committing and Pushing

- NEVER commit or push without user approval
- Always propose a meaningful commit title and message
- The message should summarize what changed (1-2 sentences), not list every file
- Use imperative mood ("Add", "Fix", "Update" — not "Added", "Fixed")
- Keep title under 72 characters
- Ask "commit?" before executing after proposing
- After commit succeeds, ask "push?" before pushing
- When pushing:
  - First check the current branch with `git branch -vv`
  - If on dev or main branch, refuse to push and ask user to create a new branch
  - Use `git push origin <branch_name>`

## Checking Quality Gates

All quality gates (lint, spell, type_check, tests, SonarCloud) must pass before merging.

**1. Check via GitHub CLI:**
```bash
gh pr view <MR_number> --json state,mergeable,statusCheckRollup
```

**2. Check SonarQube issues (required):**
```bash
curl -s "https://sonarcloud.io/api/issues/search?componentKeys=Safenai_dqm-ml-workspace&pullRequest=<MR_number>&statuses=OPEN,CONFIRMED" | jq '.total'
```

If `.total` > 0, there are issues to fix before merging.

## Request Guidelines

- **Rate limit**: Do not exceed 1 request/second to any website or API
- **What counts**:
  - Direct HTTP calls (`curl`, `requests`, `fetch`, etc.)
  - Web scraping or crawling
  - Web searches
  - Third-party API calls (e.g., SonarCloud API)
- **Safe to use without limits**:
  - Local file operations
  - Codebase searches (grep, glob, read tools)

**Example — checking MR quality gates:**
```bash
# GOOD — single request
curl -s "https://sonarcloud.io/api/issues/..." | jq '.total'

# BAD — multiple rapid requests (loop without delays)
for i in {1..10}; do curl ...; done
```

When checking MR status, run commands one at a time with natural pauses between them.
