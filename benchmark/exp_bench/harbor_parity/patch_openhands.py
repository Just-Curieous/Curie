"""Apply EXP-Bench's two OpenHands source patches without hard-coded line numbers.

Upstream ExpDockerfile_default patches OpenHands at commit 02bc7de with

    sed -i '238i \\                "/var/run/docker.sock": {...},' .../docker/docker_runtime.py
    sed -i '36i\\    user_id = 123 if user_id == 0 else user_id' .../utils/runtime_init.py

Line numbers only fit that one commit. This script inserts the same lines next to the
code they belong to, so the OpenHands version can be changed (OPENHANDS_REF build arg).
At commit 02bc7de the result is byte-identical to the sed commands (checked when this
script was written; see harbor_parity/README.md). It fails loudly when an anchor is not
found instead of corrupting the source.

Usage: python patch_openhands.py /helper_openhands/OpenHands [--no-docker-sock]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOCKER_SOCK_LINE = '"/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},'
USER_ID_LINE = "user_id = 123 if user_id == 0 else user_id"


def patch_docker_runtime(path: Path) -> None:
    """Mount the host docker socket into the sandbox (upstream line 238 insert).

    Inserted as the first entry of the ``volumes = {`` dict that maps
    ``self.config.workspace_mount_path``.
    """
    lines = path.read_text().splitlines(keepends=True)
    if any(DOCKER_SOCK_LINE in line for line in lines):
        print(f"already patched: {path}")
        return
    for i in range(len(lines) - 1):
        if re.match(r"^\s*volumes = \{\s*$", lines[i]) and "self.config.workspace_mount_path: {" in lines[i + 1]:
            indent = re.match(r"^(\s*)", lines[i + 1]).group(1)
            lines.insert(i + 1, f"{indent}{DOCKER_SOCK_LINE}\n")
            path.write_text("".join(lines))
            print(f"patched (docker.sock mount at line {i + 2}): {path}")
            return
    sys.exit(
        f"ERROR: anchor 'volumes = {{' + 'self.config.workspace_mount_path: {{' not found in {path}. "
        "This OpenHands version needs a new docker.sock patch (or build with OPENHANDS_DOCKER_SOCK_PATCH=0)."
    )


def patch_runtime_init(path: Path) -> None:
    """Map sandbox uid 0 to 123 (upstream line 36 insert).

    Inserted as the first statement of ``init_user_and_working_directory`` (after its
    docstring), which is where line 36 falls at commit 02bc7de.
    """
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    if any(line.strip() == USER_ID_LINE for line in lines):
        print(f"already patched: {path}")
        return
    start = next(
        (i for i, line in enumerate(lines) if re.match(r"^def init_user_and_working_directory\(", line)),
        None,
    )
    if start is None:
        sys.exit(f"ERROR: init_user_and_working_directory not found in {path}")
    # Skip the signature (may span lines) up to the line ending with ':'.
    i = start
    while not lines[i].rstrip().endswith(":"):
        i += 1
    i += 1
    # Skip the docstring.
    if lines[i].strip().startswith(('"""', "'''")):
        quote = lines[i].strip()[:3]
        if lines[i].strip().count(quote) >= 2 and len(lines[i].strip()) > 3:
            i += 1
        else:
            i += 1
            while quote not in lines[i]:
                i += 1
            i += 1
    body_indent = re.match(r"^(\s*)", lines[i]).group(1) or "    "
    # At 02bc7de upstream's line 36 falls between the two leading comment lines
    # ("# if username is CURRENT_USER ..." / "# This is specific to the local runtime");
    # keep that exact position when the comment is there.
    for j in range(i, min(i + 4, len(lines))):
        if lines[j].strip() == "# This is specific to the local runtime":
            i = j
            break
    lines.insert(i, f"{body_indent}{USER_ID_LINE}\n")
    path.write_text("".join(lines))
    print(f"patched (uid 0 -> 123 at line {i + 1}): {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("openhands_dir", type=Path)
    parser.add_argument("--no-docker-sock", action="store_true", help="Skip the docker.sock mount patch.")
    args = parser.parse_args()
    root = args.openhands_dir
    if not args.no_docker_sock:
        patch_docker_runtime(root / "openhands/runtime/impl/docker/docker_runtime.py")
    patch_runtime_init(root / "openhands/runtime/utils/runtime_init.py")


if __name__ == "__main__":
    main()
