"""Bounded network overlap while all PDF/source work stays on its owner thread."""
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
from queue import Empty, Queue
from threading import Lock, get_ident


class SourceCalls:
    def __init__(self):
        self.owner = get_ident()
        self.queue = Queue()
        self.lock = Lock()
        self.closed = False

    def __call__(self, function):
        if get_ident() == self.owner:
            return function()
        future = Future()
        with self.lock:
            if self.closed:
                raise RuntimeError('Source dispatch is closed')
            self.queue.put((function, future))
        return future.result()

    def serve(self, timeout=0):
        try:
            function, future = self.queue.get(timeout=timeout)
        except Empty:
            return
        try:
            future.set_result(function())
        except BaseException as error:
            future.set_exception(error)

    def close(self):
        with self.lock:
            self.closed = True
            while not self.queue.empty():
                _, future = self.queue.get_nowait()
                future.set_exception(RuntimeError('Source dispatch is closed'))


def ordered_cloud_map(function, items, workers=1):
    """Yield in input order; hold at most workers prepared windows at a time.

    ``function(item, source_call)`` may use source_call for a context reflow.
    The producer and every source_call execute on the calling thread, even
    while independent HTTP requests execute concurrently.
    """
    if workers <= 1:
        for item in items:
            yield function(item, lambda action: action())
        return
    calls = SourceCalls()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='marker-cloud')
    pending = deque()
    iterator = iter(items)
    exhausted = False
    try:
        while pending or not exhausted:
            while len(pending) < workers and not exhausted:
                calls.serve()
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(pool.submit(function, item, calls))
            if not pending:
                break
            while not pending[0].done():
                calls.serve(timeout=0.05)
            yield pending.popleft().result()
    finally:
        calls.close()
        pool.shutdown(wait=True, cancel_futures=True)
