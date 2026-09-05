# Workflow helpers

Deliberately outside `scripts/`: CI runs `ruff check src tests scripts experiments`,
and these are workflow plumbing rather than research code, so they should not sit
under the project's lint gate.

| File | Purpose |
| --- | --- |
| `commit-verified.py` | Commits files through GitHub's GraphQL `createCommitOnBranch`, which signs the commit server-side. Sends additions and, with `--all`, the deletions needed to make the target match the source tree. |
| `report-signature.py` | Reads the resulting commit back and fails the job if it did not actually come back verified. |

## Getting the Verified badge

A **Verified** badge means GitHub checked a signature made by a key belonging to the
author. Nothing can produce one on your behalf without holding a key you do not
control, which is why commits pushed over plain git read
`"verified": false, "reason": "unsigned"`. That is expected, not a misconfiguration.

Two routes, both usable from a phone:

**1. Merge a pull request on github.com.** GitHub builds the merge or squash commit
itself, so it is signed and authored by you. No setup. This is the right default for
everyday work.

**2. Run the *Publish verified commit* workflow.** Useful when you want a branch's
tree published without opening a PR. Authorship follows the token:

| Token | Author | Verified |
| --- | --- | --- |
| `GITHUB_TOKEN` (default) | `github-actions[bot]` | yes |
| `RESUME_PAT` secret | you | yes |

To author as yourself, create a fine-grained token with **Contents: Read and write**
on this repository and save it as the `RESUME_PAT` repository secret. A token scoped
to a different repository will not work here — repository access is part of the token.
