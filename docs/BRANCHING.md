# Branching Strategy

This repository uses a Gitflow-style branching model.

## Long-lived branches

### `develop`

`develop` contains integrated work from feature branches. It is never deleted and is the source for new feature branches and release branches.

### `master`

`master` contains tested release code considered ready for production. Hotfix branches are created from `master`.

## Short-lived branches

### `feature/<work-item>-<description>`

Create feature branches from `develop`.

Example:

```text
feature/42-improve-team-section
```

Feature branches merge back into `develop` through a pull request after required quality checks pass. Delete the branch after a successful merge.

### `release/<version>`

Create release branches from `develop` when an integrated set of features is ready for release testing.

Example:

```text
release/1.0.0
```

When release testing is complete, merge the release branch into both `master` and `develop`, then delete the release branch.

### `bugfix/<work-item>-<description>`

Create bugfix branches from the relevant `release/*` branch for issues found during release testing. Merge the tested fix back into that release branch, then delete the bugfix branch.

### `hotfix/<work-item>-<description>`

Create hotfix branches from `master` for issues found in production or a higher environment. After successful testing, merge the fix into both `master` and `develop`, then delete the hotfix branch.

## Workflow

```text
              +---------------- feature/*
              |                    |
              |                    v
              +---------------> develop
                                  |
                                  v
                              release/*
                             /    |    \
                    bugfix/*      |     \
                                  v      \
                               master <---+
                                  |
                                  v
                               deploy
                                  |
                               hotfix/*
                                /     \
                               v       v
                            master   develop
```

## Pull requests

- Feature branches target `develop`.
- Bugfix branches target their parent release branch.
- Release branches merge into `master` and back into `develop`.
- Hotfix branches merge into `master` and `develop`.
- Rebase or update a short-lived branch with its base branch when necessary to reduce merge conflicts.
- Delete merged short-lived branches.

## Work-item traceability

When a task tracker is being used, include the work-item or issue number in the branch name and pull request so the code change can be traced back to the task that required it.
