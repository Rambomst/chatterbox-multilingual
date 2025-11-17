"""
Model pooling for concurrent TTS request processing.

This module provides a thread-safe model pool that allows multiple concurrent
requests to be processed without audio bleeding between requests.
Uses CUDA streams for better GPU scheduling when processing concurrent requests.
"""
import os
import threading
from contextlib import contextmanager
from queue import Queue
from typing import Generic, TypeVar, Callable, Optional
import weakref

import torch


T = TypeVar('T')


class CUDAStreamContext:
    """Context manager for CUDA stream operations."""

    def __init__(self, stream: Optional[torch.cuda.Stream], device: str):
        self.stream = stream
        self.device = device
        self.use_cuda = device not in ["cpu", "mps"] and torch.cuda.is_available()

    def __enter__(self):
        if self.use_cuda and self.stream is not None:
            self.stream.__enter__()
        return self

    def __exit__(self, *args):
        if self.use_cuda and self.stream is not None:
            self.stream.__exit__(*args)


class ModelPool(Generic[T]):
    """
    Thread-safe pool for managing multiple model instances with CUDA stream support.

    Features:
    - Queue-based allocation for fair request handling
    - CUDA streams for better GPU parallelization
    - Automatic cleanup on deletion
    - Context manager support for safe resource management
    - Configurable pool size via environment variable

    Args:
        factory: Callable that creates a new model instance
        pool_size: Number of models in the pool. Defaults to CHATTERBOX_MODEL_POOL_SIZE env var or 1.
        device: Device string (e.g., 'cuda', 'cpu') for CUDA stream creation
    """

    def __init__(self, factory: Callable[[], T], pool_size: int = None, device: str = 'cpu'):
        if pool_size is None:
            pool_size = int(os.environ.get('CHATTERBOX_MODEL_POOL_SIZE', '1'))

        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")

        self.factory = factory
        self.pool_size = pool_size
        self.device = device
        self._pool: Queue[tuple[T, Optional[torch.cuda.Stream]]] = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._initialized = False
        self._all_instances = []
        self._all_streams = []

        # Use weakref finalizer for cleanup
        self._finalizer = weakref.finalize(
            self, self._cleanup, self._all_instances, self._all_streams
        )

    def _initialize(self):
        """Lazy initialization of the pool with CUDA streams."""
        with self._lock:
            if self._initialized:
                return

            use_cuda = self.device not in ["cpu", "mps"] and torch.cuda.is_available()

            for i in range(self.pool_size):
                instance = self.factory()
                self._all_instances.append(instance)

                # Create a dedicated CUDA stream for this instance if using CUDA
                if use_cuda:
                    stream = torch.cuda.Stream()
                    self._all_streams.append(stream)
                else:
                    stream = None

                self._pool.put((instance, stream))

            self._initialized = True

    @contextmanager
    def acquire(self):
        """
        Acquire a model and its CUDA stream from the pool. Use as a context manager.

        Example:
            with pool.acquire() as (model, stream):
                with CUDAStreamContext(stream, device):
                    result = model.generate(...)

        Yields:
            Tuple of (model instance, CUDA stream or None)
        """
        if not self._initialized:
            self._initialize()

        # Block until a model is available
        model, stream = self._pool.get()
        try:
            yield model, stream
        finally:
            # Always return the model to the pool
            self._pool.put((model, stream))

    @staticmethod
    def _cleanup(instances, streams):
        """Clean up all model instances and CUDA streams."""
        # Synchronize all streams before cleanup
        for stream in streams:
            if stream is not None:
                stream.synchronize()

        for instance in instances:
            # Clear CUDA cache if using GPU
            if hasattr(instance, 'device'):
                device = getattr(instance, 'device', None)
                if device and isinstance(device, str) and 'cuda' in device:
                    torch.cuda.empty_cache()
                elif device and isinstance(device, torch.device) and device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Delete the instance
            del instance

        # Delete streams
        for stream in streams:
            del stream

        # Final cache clear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def shutdown(self):
        """Explicitly shutdown the pool and clean up resources."""
        if self._finalizer.alive:
            self._finalizer()

    @property
    def is_initialized(self) -> bool:
        """Check if the pool has been initialized."""
        return self._initialized

    @property
    def available_count(self) -> int:
        """Get the number of currently available models in the pool."""
        return self._pool.qsize()
