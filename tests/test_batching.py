import pytest
import asyncio
import time
import threading
from unittest.mock import MagicMock


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.generate.side_effect = lambda prompt: f"result:{prompt}"
    return engine


@pytest.fixture(autouse=True)
def fresh_queue(monkeypatch):
    import server.main as m
    q = asyncio.Queue()
    monkeypatch.setattr(m, "queue", q)
    return q


async def _run_loop_until(engine, *futures, timeout=2.0):
    import server.main as m
    task = asyncio.create_task(m.batch_loop(engine))
    try:
        return await asyncio.wait_for(asyncio.gather(*futures), timeout=timeout)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestBatchLoop:

    @pytest.mark.asyncio
    async def test_single_request_returns_result(self, fresh_queue, mock_engine):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        await fresh_queue.put(({"prompt": "hello world", "max_tokens": 128}, fut))

        (result,) = await _run_loop_until(mock_engine, fut)

        assert result == "result:hello world"
        mock_engine.generate.assert_called_once_with("hello world")

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests_all_resolved(self, fresh_queue, mock_engine):
        loop = asyncio.get_event_loop()
        prompts = ["alpha", "beta", "gamma"]
        futures = []
        for p in prompts:
            fut = loop.create_future()
            await fresh_queue.put(({"prompt": p, "max_tokens": 128}, fut))
            futures.append(fut)

        results = await _run_loop_until(mock_engine, *futures)

        assert results == [f"result:{p}" for p in prompts]
        assert mock_engine.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_pre_queued_requests_batched_in_one_executor_call(
        self, fresh_queue, mock_engine, monkeypatch
    ):
        """Requests already in the queue when batch_loop starts are collected into one batch."""
        import server.main as m

        executor_call_count = [0]
        loop = asyncio.get_event_loop()

        async def counting_executor(executor, func):
            executor_call_count[0] += 1
            return func()

        monkeypatch.setattr(loop, "run_in_executor", counting_executor)

        futures = []
        for i in range(3):
            fut = loop.create_future()
            await fresh_queue.put(({"prompt": f"p{i}", "max_tokens": 128}, fut))
            futures.append(fut)

        await _run_loop_until(mock_engine, *futures)

        assert executor_call_count[0] == 1

    @pytest.mark.asyncio
    async def test_max_batch_size_is_respected(self, fresh_queue, mock_engine, monkeypatch):
        """Requests exceeding MAX_BATCH_SIZE are split across multiple batches."""
        import server.main as m

        batch_sizes = []
        loop = asyncio.get_event_loop()

        async def recording_executor(executor, func):
            results = func()
            batch_sizes.append(len(results))
            return results

        monkeypatch.setattr(loop, "run_in_executor", recording_executor)

        n = m.MAX_BATCH_SIZE + 1
        futures = []
        for i in range(n):
            fut = loop.create_future()
            await fresh_queue.put(({"prompt": f"p{i}", "max_tokens": 128}, fut))
            futures.append(fut)

        await _run_loop_until(mock_engine, *futures)

        assert batch_sizes[0] == m.MAX_BATCH_SIZE
        assert sum(batch_sizes) == n

    @pytest.mark.asyncio
    async def test_engine_exception_propagates_to_future(self, fresh_queue):
        import server.main as m

        engine = MagicMock()
        engine.generate.side_effect = RuntimeError("inference failed")

        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        await fresh_queue.put(({"prompt": "bad prompt", "max_tokens": 128}, fut))

        with pytest.raises(RuntimeError, match="inference failed"):
            await _run_loop_until(engine, fut)

    @pytest.mark.asyncio
    async def test_batch_loop_exits_cleanly_on_cancellation(self, fresh_queue, mock_engine):
        import server.main as m

        task = asyncio.create_task(m.batch_loop(mock_engine))
        await asyncio.sleep(0.01)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        assert task.done()

    @pytest.mark.asyncio
    async def test_requests_arriving_after_timeout_go_into_next_batch(
        self, fresh_queue, mock_engine, monkeypatch
    ):
        """A request that arrives after the first batch completes forms a separate batch."""
        import server.main as m

        batch_sizes = []
        loop = asyncio.get_event_loop()

        async def recording_executor(executor, func):
            results = func()
            batch_sizes.append(len(results))
            return results

        monkeypatch.setattr(loop, "run_in_executor", recording_executor)

        task = asyncio.create_task(m.batch_loop(mock_engine))

        fut1 = loop.create_future()
        await fresh_queue.put(({"prompt": "early", "max_tokens": 128}, fut1))
        await asyncio.wait_for(fut1, timeout=2.0)

        fut2 = loop.create_future()
        await fresh_queue.put(({"prompt": "late", "max_tokens": 128}, fut2))
        await asyncio.wait_for(fut2, timeout=2.0)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(batch_sizes) == 2
        assert batch_sizes == [1, 1]


class TestHighVolume:

    @pytest.mark.asyncio
    async def test_100_concurrent_requests(self, fresh_queue, mock_engine):
        import server.main as m

        loop = asyncio.get_event_loop()
        n = 100
        futures = []
        for i in range(n):
            fut = loop.create_future()
            await fresh_queue.put(({"prompt": f"prompt_{i}", "max_tokens": 128}, fut))
            futures.append(fut)

        results = await _run_loop_until(mock_engine, *futures, timeout=10.0)

        assert len(results) == n
        assert results == [f"result:prompt_{i}" for i in range(n)]
        assert mock_engine.generate.call_count == n


class TestHeadOfLineBlocking:

    @pytest.mark.asyncio
    async def test_slow_request_blocks_fast_requests(self, fresh_queue):
        """
        Demonstrates head-of-line blocking: a slow request occupies the batch_loop's
        executor call, so fast requests queued while it runs cannot be dispatched
        until the slow one finishes — even though they themselves are instant.
        """
        import server.main as m

        SLOW_DURATION = 0.15  # 150ms

        slow_started = threading.Event()

        def generate(prompt):
            if prompt == "slow":
                slow_started.set()
                time.sleep(SLOW_DURATION)
            return f"result:{prompt}"

        engine = MagicMock()
        engine.generate.side_effect = generate

        loop = asyncio.get_event_loop()

        # Enqueue the slow request and start the batch_loop
        slow_fut = loop.create_future()
        await fresh_queue.put(({"prompt": "slow", "max_tokens": 128}, slow_fut))
        task = asyncio.create_task(m.batch_loop(engine))

        # Block until the slow request is actually executing in the thread pool
        await loop.run_in_executor(None, slow_started.wait)

        # Enqueue fast requests while slow is mid-execution
        fast_futs = []
        t0 = loop.time()
        for i in range(3):
            fut = loop.create_future()
            await fresh_queue.put(({"prompt": f"fast_{i}", "max_tokens": 128}, fut))
            fast_futs.append(fut)

        await asyncio.gather(*fast_futs)
        fast_wait = loop.time() - t0

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The fast requests were blocked for the remaining ~150ms of the slow request.
        # Under a scheduler with per-request prioritisation they would complete in <1ms.
        assert fast_wait >= SLOW_DURATION * 0.8, (
            f"Fast requests completed in {fast_wait * 1000:.0f}ms — "
            f"expected head-of-line blocking to delay them by ~{SLOW_DURATION * 1000:.0f}ms"
        )
        assert slow_fut.result() == "result:slow"
        assert [f.result() for f in fast_futs] == [f"result:fast_{i}" for i in range(3)]


    @pytest.mark.asyncio
    async def test_large_prompt_blocks_small_prompts(self, fresh_queue):
        """
        Demonstrates that prompt size drives head-of-line blocking: a large prompt
        with long inference time prevents small, fast prompts from being dispatched
        until the large one finishes — even though they would complete in <1ms on
        their own.
        """
        import server.main as m

        CHARS_PER_SECOND = 50_000  # simulated inference throughput

        LARGE_PROMPT = "x" * 10_000  # ~200ms at simulated throughput
        SMALL_PROMPT = "y" * 10      # ~0.2ms at simulated throughput

        large_started = threading.Event()

        def generate(prompt):
            duration = len(prompt) / CHARS_PER_SECOND
            if len(prompt) > 1_000:
                large_started.set()
            time.sleep(duration)
            return f"result:len={len(prompt)}"

        engine = MagicMock()
        engine.generate.side_effect = generate

        loop = asyncio.get_event_loop()

        large_fut = loop.create_future()
        await fresh_queue.put(({"prompt": LARGE_PROMPT, "max_tokens": 128}, large_fut))
        task = asyncio.create_task(m.batch_loop(engine))

        # Block until the large prompt is actually executing in the thread pool
        await loop.run_in_executor(None, large_started.wait)

        # Queue small prompts while the large one is mid-execution
        small_futs = []
        t0 = loop.time()
        for _ in range(3):
            fut = loop.create_future()
            await fresh_queue.put(({"prompt": SMALL_PROMPT, "max_tokens": 128}, fut))
            small_futs.append(fut)

        await asyncio.gather(*small_futs)
        small_wait = loop.time() - t0

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        expected_large_duration = len(LARGE_PROMPT) / CHARS_PER_SECOND

        # Small prompts were blocked for the remainder of the large prompt's runtime.
        # A scheduler aware of prompt size could have served them in <1ms.
        assert small_wait >= expected_large_duration * 0.5, (
            f"Small prompts completed in {small_wait * 1000:.0f}ms — "
            f"expected to be blocked ~{expected_large_duration * 1000:.0f}ms "
            "by the large prompt"
        )
        assert large_fut.result() == f"result:len={len(LARGE_PROMPT)}"
        assert all(f.result() == f"result:len={len(SMALL_PROMPT)}" for f in small_futs)


class TestHandleRequest:

    @pytest.mark.asyncio
    async def test_handle_request_returns_engine_result(self, fresh_queue, mock_engine):
        import server.main as m

        task = asyncio.create_task(m.batch_loop(mock_engine))
        result = await asyncio.wait_for(m.handle_request("my prompt", 64), timeout=2.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert result == "result:my prompt"

    @pytest.mark.asyncio
    async def test_handle_request_enqueues_correct_payload(self, fresh_queue):
        import server.main as m

        asyncio.create_task(m.handle_request("enqueue me", 256))
        await asyncio.sleep(0)

        assert fresh_queue.qsize() == 1
        payload, fut = await fresh_queue.get()
        assert payload == {"prompt": "enqueue me", "max_tokens": 256}
        assert isinstance(fut, asyncio.Future)
