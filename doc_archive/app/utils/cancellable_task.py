"""
Provides utilities for handling cancellable async tasks.
"""
import asyncio
from typing import Any, Callable, Optional, TypeVar
from functools import wraps

T = TypeVar('T')

class CancellableTask:
    """
    Wrapper for asyncio tasks that can be safely cancelled.
    Provides timeout and cleanup functionality.
    """
    
    def __init__(self, timeout: Optional[float] = None):
        self.timeout = timeout
        self.task: Optional[asyncio.Task] = None
        self._cleanup_handlers = []
        
    def add_cleanup(self, handler: Callable[[], Any]):
        """Add a cleanup handler to run when task is cancelled."""
        self._cleanup_handlers.append(handler)
        
    async def run(self, coro) -> T:
        """Run coroutine with timeout and cleanup handling."""
        try:
            self.task = asyncio.create_task(coro)
            if self.timeout:
                return await asyncio.wait_for(self.task, self.timeout)
            return await self.task
            
        except asyncio.TimeoutError:
            if self.task:
                self.task.cancel()
            await self._run_cleanup()
            raise
            
        except asyncio.CancelledError:
            await self._run_cleanup()
            raise
            
        finally:
            self.task = None
            
    async def cancel(self):
        """Cancel the running task and run cleanup."""
        if self.task:
            self.task.cancel()
            await self._run_cleanup()
            
    async def _run_cleanup(self):
        """Run all cleanup handlers."""
        for handler in self._cleanup_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception:
                pass  # Cleanup errors shouldn't propagate

def cancellable(timeout: Optional[float] = None):
    """
    Decorator to make a coroutine cancellable with timeout.
    
    Usage:
    @cancellable(timeout=30)
    async def long_running_task():
        ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            task = CancellableTask(timeout)
            return await task.run(func(*args, **kwargs))
        return wrapper
    return decorator