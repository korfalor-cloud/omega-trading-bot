"""Task Planning and Scheduling Module.

Handles:
- Task scheduling (periodic, one-time, conditional)
- Workflow orchestration
- Strategy lifecycle management
- Resource allocation
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = auto()
    SCHEDULED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class TaskPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ScheduledTask:
    """A scheduled task."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    func: Optional[Callable] = None
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    scheduled_at: datetime = field(default_factory=datetime.utcnow)
    interval_seconds: float = 0  # 0 = one-time
    max_retries: int = 0
    retry_count: int = 0
    timeout_seconds: float = 300
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None

    @property
    def is_due(self) -> bool:
        if self.status in (TaskStatus.CANCELLED, TaskStatus.COMPLETED):
            return False
        if self.next_run is None:
            return self.scheduled_at <= datetime.utcnow()
        return self.next_run <= datetime.utcnow()

    @property
    def is_recurring(self) -> bool:
        return self.interval_seconds > 0


class TaskScheduler:
    """Async task scheduler with priority queue and recurring tasks.

    Features:
    - One-time and recurring tasks
    - Priority-based execution
    - Automatic retries with backoff
    - Task dependencies
    - Workflow chains
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.max_concurrent = config.get("max_concurrent", 5)
        self._tasks: dict[str, ScheduledTask] = {}
        self._running: dict[str, asyncio.Task] = {}
        self._history: list[dict] = []
        self._running_flag = False

    def schedule(
        self,
        name: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        delay_seconds: float = 0,
        interval_seconds: float = 0,
        max_retries: int = 0,
        timeout_seconds: float = 300,
    ) -> ScheduledTask:
        """Schedule a task for execution."""
        task = ScheduledTask(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            scheduled_at=datetime.utcnow() + timedelta(seconds=delay_seconds),
            interval_seconds=interval_seconds,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            next_run=datetime.utcnow() + timedelta(seconds=delay_seconds),
        )
        self._tasks[task.id] = task
        logger.info(f"Scheduled task: {name} (id={task.id[:8]}, delay={delay_seconds}s)")
        return task

    def schedule_cron(
        self,
        name: str,
        func: Callable,
        hour: int,
        minute: int = 0,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> ScheduledTask:
        """Schedule a daily task at specific hour:minute."""
        now = datetime.utcnow()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        delay = (target - now).total_seconds()

        return self.schedule(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs or {},
            delay_seconds=delay,
            interval_seconds=86400,  # Daily
        )

    def cancel(self, task_id: str) -> bool:
        """Cancel a task."""
        task = self._tasks.get(task_id)
        if task and task.status in (TaskStatus.PENDING, TaskStatus.SCHEDULED):
            task.status = TaskStatus.CANCELLED
            return True
        return False

    def cancel_all(self, name_pattern: str = "") -> int:
        """Cancel matching tasks."""
        cancelled = 0
        for task in self._tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.SCHEDULED):
                if not name_pattern or name_pattern in task.name:
                    task.status = TaskStatus.CANCELLED
                    cancelled += 1
        return cancelled

    def get_due_tasks(self) -> list[ScheduledTask]:
        """Get tasks that are due for execution."""
        return sorted(
            [t for t in self._tasks.values() if t.is_due],
            key=lambda t: t.priority.value,
            reverse=True,
        )

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        return self._tasks.get(task_id)

    def get_tasks_by_status(self, status: TaskStatus) -> list[ScheduledTask]:
        return [t for t in self._tasks.values() if t.status == status]

    def get_status(self) -> dict:
        status_counts = {}
        for task in self._tasks.values():
            status_counts[task.status.name] = status_counts.get(task.status.name, 0) + 1

        return {
            "total_tasks": len(self._tasks),
            "status_counts": status_counts,
            "running": len(self._running),
            "history_size": len(self._history),
        }

    async def run_due_tasks(self) -> list[dict]:
        """Execute all due tasks (call this in your main loop)."""
        due = self.get_due_tasks()
        results = []

        for task in due[:self.max_concurrent - len(self._running)]:
            if task.id in self._running:
                continue

            task.status = TaskStatus.RUNNING
            task.last_run = datetime.utcnow()

            try:
                if asyncio.iscoroutinefunction(task.func):
                    result = await asyncio.wait_for(
                        task.func(*task.args, **task.kwargs),
                        timeout=task.timeout_seconds,
                    )
                else:
                    result = task.func(*task.args, **task.kwargs)

                task.result = result
                task.status = TaskStatus.COMPLETED

                # Schedule next run if recurring
                if task.is_recurring:
                    task.next_run = datetime.utcnow() + timedelta(seconds=task.interval_seconds)
                    task.status = TaskStatus.PENDING

                results.append({"task_id": task.id, "name": task.name, "result": result})

            except Exception as e:
                task.error = str(e)
                task.retry_count += 1

                if task.retry_count <= task.max_retries:
                    # Exponential backoff
                    backoff = min(300, 2 ** task.retry_count)
                    task.next_run = datetime.utcnow() + timedelta(seconds=backoff)
                    task.status = TaskStatus.PENDING
                    logger.warning(f"Task {task.name} failed, retry {task.retry_count} in {backoff}s")
                else:
                    task.status = TaskStatus.FAILED
                    logger.error(f"Task {task.name} failed after {task.max_retries} retries: {e}")

                results.append({"task_id": task.id, "name": task.name, "error": str(e)})

            self._history.append({
                "task_id": task.id,
                "name": task.name,
                "status": task.status.name,
                "timestamp": datetime.utcnow().isoformat(),
            })

        return results

    def cleanup_completed(self, max_age_hours: int = 24) -> int:
        """Remove old completed/failed tasks."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        to_remove = [
            tid for tid, t in self._tasks.items()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            and t.created_at < cutoff
        ]
        for tid in to_remove:
            del self._tasks[tid]
        return len(to_remove)


class WorkflowChain:
    """Chain tasks into a workflow pipeline."""

    def __init__(self, name: str):
        self.name = name
        self._steps: list[tuple[str, Callable]] = []
        self._results: dict[str, Any] = {}

    def add_step(self, name: str, func: Callable) -> "WorkflowChain":
        self._steps.append((name, func))
        return self

    async def execute(self, initial_input: Any = None) -> dict[str, Any]:
        """Execute all steps in sequence."""
        current_input = initial_input
        self._results = {}

        for step_name, step_func in self._steps:
            try:
                if asyncio.iscoroutinefunction(step_func):
                    result = await step_func(current_input)
                else:
                    result = step_func(current_input)

                self._results[step_name] = result
                current_input = result
                logger.info(f"Workflow '{self.name}' step '{step_name}' completed")

            except Exception as e:
                self._results[step_name] = {"error": str(e)}
                logger.error(f"Workflow '{self.name}' step '{step_name}' failed: {e}")
                break

        return self._results

    @property
    def results(self) -> dict[str, Any]:
        return self._results
