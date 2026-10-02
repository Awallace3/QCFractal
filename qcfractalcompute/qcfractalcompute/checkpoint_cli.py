"""
List and sweep the checkpoint directories of a compute manager's executors.

``list`` reads only the shared filesystem. ``sweep`` also asks the server for each record's status and reports
directories whose record is complete, invalid or deleted (or no longer exists); ``--delete`` removes them.
Directories of records that are still waiting, running, errored or cancelled are always kept, since a reset or
an uncancel resumes from them.
"""

from __future__ import annotations

import argparse
import datetime
import sys

import tabulate

from . import checkpointing
from .config import read_configuration

REMOVABLE_STATUSES = {"complete", "invalid", "deleted", "missing"}


def _checkpoint_executors(config, only: str | None):
    executors = {
        label: ex.checkpoint
        for label, ex in config.executors.items()
        if ex.checkpoint is not None and (only is None or label == only)
    }
    if not executors:
        raise SystemExit("No executor in this configuration has a checkpoint block" + (f" named {only}" if only else ""))
    return executors


def _record_rows(executors):
    rows = []
    for label, ckpt_config in executors.items():
        for record_id, record_dir in sorted(checkpointing.list_record_dirs(ckpt_config).items()):
            status = checkpointing.read_status(record_dir)
            done, running, attempts = checkpointing.describe_progress(status)
            updated = status.get("updated_at") if status else None
            updated_str = (
                datetime.datetime.fromtimestamp(updated).strftime("%Y-%m-%d %H:%M") if updated is not None else "-"
            )
            state = status.get("state", "-") if status else "-"
            rows.append([label, record_id, state, done, running, attempts, updated_str, ckpt_config])
    return rows


def _server_status(config, record_ids, username, password):
    from qcportal import PortalClient

    client = PortalClient(
        config.server.fractal_uri,
        username=username or config.server.username,
        password=password or config.server.password,
        verify=True if config.server.verify is None else config.server.verify,
        show_motd=False,
    )
    statuses = {}
    ids = sorted(record_ids)
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        for record_id, record in zip(chunk, client.get_records(chunk, missing_ok=True)):
            statuses[record_id] = "missing" if record is None else str(record.status.value)
    return statuses


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect and sweep QCFractal compute checkpoint directories")
    parser.add_argument("--config", required=True, help="Compute manager configuration file")
    parser.add_argument("--executor", default=None, help="Only this executor")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Show every checkpoint directory and its psi4 progress")
    sweep = sub.add_parser("sweep", help="Find checkpoint directories whose record no longer needs them")
    sweep.add_argument("--delete", action="store_true", help="Remove them (default: only report)")
    sweep.add_argument("--username", default=None, help="Server user with read access (default: the manager's)")
    sweep.add_argument("--password", default=None)
    clear = sub.add_parser("clear", help="Let records run again before checkpoint.fatal_hold expires")
    clear.add_argument("record_ids", nargs="+", type=int)
    args = parser.parse_args(argv)

    config = read_configuration([args.config])
    executors = _checkpoint_executors(config, args.executor)
    if args.command == "clear":
        for record_id in args.record_ids:
            for ckpt_config in executors.values():
                ledger = checkpointing.read_ledger(ckpt_config, record_id)
                if ledger.pop("fatal", None) is not None:
                    checkpointing.write_ledger(ckpt_config, record_id, ledger)
                    print(f"Released record {record_id}")
        return 0

    rows = _record_rows(executors)
    headers = ["executor", "record id", "psi4 state", "done", "next stage", "attempts", "updated"]

    if args.command == "list":
        print(tabulate.tabulate([row[:-1] for row in rows], headers=headers))
        return 0

    statuses = _server_status(config, {row[1] for row in rows}, args.username, args.password)
    removable = []
    for row in rows:
        server_status = statuses.get(row[1], "missing")
        row.insert(-1, server_status)
        if server_status in REMOVABLE_STATUSES:
            removable.append(row)

    print(tabulate.tabulate([row[:-1] for row in rows], headers=headers + ["server status"]))
    print(f"\n{len(removable)} of {len(rows)} checkpoint directories belong to complete, invalid or deleted records")

    if args.delete:
        for row in removable:
            checkpointing.remove_record_dir(row[-1], row[1])
            print(f"Removed {row[-1].record_dir(row[1])}")
    elif removable:
        print("Rerun with --delete to remove them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
