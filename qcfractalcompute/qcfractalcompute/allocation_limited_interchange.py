"""A Parsl HTEX interchange which limits dispatched tasks per provider block.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from collections import defaultdict
from typing import Any, Optional, Set

from parsl.executors.high_throughput.interchange import Interchange
from parsl.monitoring.radios.base import MonitoringRadioSender
from parsl.utils import setproctitle

logger = logging.getLogger("parsl.executors.high_throughput.interchange")


class AllocationLimitedInterchange(Interchange):
    """An interchange which drains each block after an exact dispatch quota."""

    def __init__(self, *, max_tasks_per_allocation: int, **interchange_config: Any) -> None:
        if max_tasks_per_allocation <= 0:
            raise ValueError("max_tasks_per_allocation must be a positive integer")

        super().__init__(**interchange_config)
        self.max_tasks_per_allocation = max_tasks_per_allocation
        self._tasks_dispatched_by_block: dict[str, int] = defaultdict(int)

    def _mark_block_draining(
        self,
        block_id: str,
        interesting_managers: Set[bytes],
        monitoring_radio: Optional[MonitoringRadioSender],
    ) -> None:
        """Stop dispatch to every manager belonging to one provider block."""

        for manager_id, manager in self._ready_managers.items():
            if manager["block_id"] != block_id:
                continue

            was_draining = manager["draining"]
            manager["draining"] = True

            if manager["tasks"]:
                # A result will make this manager interesting again, allowing
                # expire_drained_managers to finish it after its last task.
                interesting_managers.discard(manager_id)
            else:
                # Keep an idle peer interesting so expire_drained_managers can
                # send DRAINED_CODE on the next event-loop pass.
                interesting_managers.add(manager_id)

            if not was_draining:
                logger.info("Manager %r in block %s reached its allocation task limit", manager_id, block_id)
                self._send_monitoring_info(monitoring_radio, manager)

    def process_tasks_to_send(
        self,
        interesting_managers: Set[bytes],
        monitoring_radio: Optional[MonitoringRadioSender],
    ) -> None:
        """Dispatch tasks without allowing any block to exceed its quota."""

        if not interesting_managers:
            return

        # Stock Parsl avoids sorting managers when no tasks are pending. Retain
        # that fast path while still draining a manager which registers late for
        # an already-exhausted multi-manager block.
        if not self.pending_task_queue:
            for manager_id in list(interesting_managers):
                manager = self._ready_managers[manager_id]
                block_id = manager["block_id"]
                assert block_id is not None, "Registered HTEX managers must have a block ID"
                if self._tasks_dispatched_by_block[block_id] >= self.max_tasks_per_allocation:
                    self._mark_block_draining(block_id, interesting_managers, monitoring_radio)
            return

        shuffled_managers = self.manager_selector.sort_managers(self._ready_managers, interesting_managers)

        # Continue examining the selected managers if an earlier dispatch
        # empties the queue so exhausted peers are still marked draining.
        while shuffled_managers:
            manager_id = shuffled_managers.pop()
            manager = self._ready_managers[manager_id]
            block_id = manager["block_id"]
            assert block_id is not None, "Registered HTEX managers must have a block ID"

            tasks_dispatched = self._tasks_dispatched_by_block[block_id]
            quota_remaining = self.max_tasks_per_allocation - tasks_dispatched
            if quota_remaining <= 0:
                self._mark_block_draining(block_id, interesting_managers, monitoring_radio)
                continue

            if not self.pending_task_queue:
                continue

            tasks_inflight = len(manager["tasks"])
            normal_capacity = manager["max_capacity"] - tasks_inflight
            real_capacity = max(0, min(normal_capacity, quota_remaining))

            if not real_capacity or not manager["active"] or manager["draining"]:
                interesting_managers.discard(manager_id)
                continue

            tasks = self.get_tasks(real_capacity)
            block_marked_draining = False

            if tasks:
                self.manager_sock.send_multipart([manager_id, pickle.dumps(tasks)])
                task_count = len(tasks)
                self.count += task_count

                task_ids = [task["task_id"] for task in tasks]
                manager["tasks"].extend(task_ids)
                manager["idle_since"] = None
                self._tasks_dispatched_by_block[block_id] += task_count

                logger.debug("Sent tasks: %s to manager %r", task_ids, manager_id)

                if self._tasks_dispatched_by_block[block_id] >= self.max_tasks_per_allocation:
                    # Record the final task IDs as in flight before setting
                    # draining, so expire_drained_managers waits for results.
                    self._mark_block_draining(block_id, interesting_managers, monitoring_radio)
                    block_marked_draining = True
                else:
                    normal_capacity -= task_count
                    if normal_capacity > 0:
                        logger.debug("Manager %r has free capacity %s", manager_id, normal_capacity)
                    else:
                        logger.debug("Manager %r is now saturated", manager_id)
                        interesting_managers.discard(manager_id)

            # _mark_block_draining sends monitoring updates for every manager
            # in the block, including this manager, when the quota is reached.
            if not block_marked_draining:
                self._send_monitoring_info(monitoring_radio, manager)

        logger.debug(
            "Leaving ready-manager dispatch with %s managers still interesting",
            len(interesting_managers),
        )


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an allocation-limited Parsl HTEX interchange")
    parser.add_argument(
        "--max-tasks-per-allocation",
        type=_positive_integer,
        required=True,
        help="Exact number of tasks which may be dispatched to each provider block",
    )
    args = parser.parse_args()

    setproctitle("parsl: allocation-limited HTEX interchange")
    interchange_config = pickle.load(sys.stdin.buffer)
    interchange = AllocationLimitedInterchange(
        max_tasks_per_allocation=args.max_tasks_per_allocation,
        **interchange_config,
    )
    interchange.start()


if __name__ == "__main__":
    main()
