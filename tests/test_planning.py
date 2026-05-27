"""Tests for planning and scheduling module."""
from __future__ import annotations

import pytest
import asyncio

from tradingbot.planning.scheduler import (
    TaskScheduler,
    TaskStatus,
    TaskPriority,
    WorkflowChain,
)


class TestTaskScheduler:
    @pytest.fixture
    def scheduler(self):
        return TaskScheduler()

    def test_schedule_task(self, scheduler):
        task = scheduler.schedule("test", lambda: 42)
        assert task.status == TaskStatus.PENDING
        assert task.name == "test"

    def test_get_due_tasks(self, scheduler):
        scheduler.schedule("immediate", lambda: 1)
        due = scheduler.get_due_tasks()
        assert len(due) >= 1

    def test_cancel_task(self, scheduler):
        task = scheduler.schedule("cancel_me", lambda: 1)
        assert scheduler.cancel(task.id)
        assert task.status == TaskStatus.CANCELLED

    def test_cancel_all(self, scheduler):
        scheduler.schedule("a", lambda: 1)
        scheduler.schedule("b", lambda: 2)
        cancelled = scheduler.cancel_all()
        assert cancelled == 2

    @pytest.mark.asyncio
    async def test_run_due_tasks(self, scheduler):
        results = []
        scheduler.schedule("task1", lambda: results.append(1) or 42)
        task_results = await scheduler.run_due_tasks()
        assert len(task_results) == 1
        assert task_results[0]["result"] == 42

    @pytest.mark.asyncio
    async def test_async_task(self, scheduler):
        async def async_func():
            return 42

        scheduler.schedule("async_task", async_func)
        results = await scheduler.run_due_tasks()
        assert results[0]["result"] == 42

    @pytest.mark.asyncio
    async def test_recurring_task(self, scheduler):
        count = 0
        def increment():
            nonlocal count
            count += 1
            return count

        scheduler.schedule("recurring", increment, interval_seconds=0.01)
        await scheduler.run_due_tasks()
        assert count == 1

    @pytest.mark.asyncio
    async def test_task_retry(self, scheduler):
        attempts = 0
        def failing():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("not yet")

        scheduler.schedule("retry_task", failing, max_retries=3)
        await scheduler.run_due_tasks()
        assert attempts == 1

    def test_status(self, scheduler):
        scheduler.schedule("a", lambda: 1)
        status = scheduler.get_status()
        assert status["total_tasks"] == 1

    def test_cleanup(self, scheduler):
        task = scheduler.schedule("old", lambda: 1)
        task.status = TaskStatus.COMPLETED
        from datetime import datetime, timedelta
        task.created_at = datetime.utcnow() - timedelta(hours=48)
        removed = scheduler.cleanup_completed(max_age_hours=24)
        assert removed == 1

    def test_schedule_cron(self, scheduler):
        task = scheduler.schedule_cron("daily", lambda: 1, hour=0, minute=0)
        assert task.is_recurring


class TestWorkflowChain:
    @pytest.mark.asyncio
    async def test_workflow_execution(self):
        chain = WorkflowChain("test_pipeline")
        chain.add_step("double", lambda x: x * 2)
        chain.add_step("add_one", lambda x: x + 1)

        results = await chain.execute(5)
        assert results["double"] == 10
        assert results["add_one"] == 11

    @pytest.mark.asyncio
    async def test_async_workflow(self):
        async def double(x):
            return x * 2

        chain = WorkflowChain("async_pipeline")
        chain.add_step("double", double)
        results = await chain.execute(5)
        assert results["double"] == 10

    @pytest.mark.asyncio
    async def test_workflow_failure(self):
        def fail(x):
            raise ValueError("boom")

        chain = WorkflowChain("failing")
        chain.add_step("fail", fail)
        chain.add_step("after", lambda x: x + 1)
        results = await chain.execute(5)
        assert "error" in str(results["fail"])
        assert "after" not in results
