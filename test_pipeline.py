"""
test_pipeline.py — Phase B test suite.

Covers: NLPPipeline, DynamicBatcher, JobQueue, and API endpoints.

Run:
    pytest test_pipeline.py -v
"""

import asyncio
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

# ═══════════════════════════════════════════════════════════════════════════════
# NLPPipeline tests
# ═══════════════════════════════════════════════════════════════════════════════
from pipeline import NLPPipeline, PipelineError

@pytest.fixture
def pipe():
    return NLPPipeline(max_length=100, max_batch=4)


class TestNLPPipeline:
    def test_string_input(self, pipe):
        result = pipe.process("Hello world")
        assert result == ["Hello world"]

    def test_list_input(self, pipe):
        result = pipe.process(["Hello", "World"])
        assert len(result) == 2

    def test_dict_input_text_key(self, pipe):
        result = pipe.process({"text": "Hello dict"})
        assert result == ["Hello dict"]

    def test_dict_input_content_key(self, pipe):
        result = pipe.process({"content": "Hello content"})
        assert result == ["Hello content"]

    def test_html_stripped(self, pipe):
        result = pipe.process("<b>Bold</b> text")
        assert "<b>" not in result[0]
        assert "Bold" in result[0]

    def test_html_entities_unescaped(self, pipe):
        result = pipe.process("AT&amp;T")
        assert result[0] == "AT&T"

    def test_whitespace_collapsed(self, pipe):
        result = pipe.process("hello    world")
        assert result[0] == "hello world"

    def test_truncation(self, pipe):
        long_text = "a" * 200
        result = pipe.process(long_text)
        assert len(result[0]) == 100

    def test_empty_string_raises(self, pipe):
        with pytest.raises(PipelineError):
            pipe.process("")

    def test_whitespace_only_raises(self, pipe):
        with pytest.raises(PipelineError):
            pipe.process("   ")

    def test_none_raises(self, pipe):
        with pytest.raises(PipelineError):
            pipe.process(None)

    def test_batch_too_large_raises(self, pipe):
        with pytest.raises(PipelineError):
            pipe.process(["a", "b", "c", "d", "e"])  # max_batch=4

    def test_empty_list_raises(self, pipe):
        with pytest.raises(PipelineError):
            pipe.process([])

    def test_list_with_non_string_raises(self, pipe):
        with pytest.raises(PipelineError):
            pipe.process(["hello", 123])

    def test_unsupported_type_raises(self, pipe):
        with pytest.raises(PipelineError):
            pipe.process(42)

    def test_unicode_normalised(self, pipe):
        # \ufb01 is the 'fi' ligature — NFKC should expand it
        result = pipe.process("\ufb01le")
        assert result[0] == "file"

    def test_max_length_param_override(self, pipe):
        result = pipe.process("a" * 200, max_length=50)
        assert len(result[0]) == 50


# ═══════════════════════════════════════════════════════════════════════════════
# DynamicBatcher tests
# ═══════════════════════════════════════════════════════════════════════════════
from batcher import BatchItem, DynamicBatcher


@pytest.mark.asyncio
class TestDynamicBatcher:
    async def test_flush_on_max_size(self):
        batches = []

        async def on_batch(batch):
            batches.append(batch)
            for item in batch:
                item.future.set_result("ok")

        batcher = DynamicBatcher(on_batch=on_batch, max_size=2, timeout_ms=5000)
        await batcher.start()

        loop = asyncio.get_event_loop()
        f1 = loop.create_future()
        f2 = loop.create_future()

        await batcher.add(BatchItem(job_id="j1", texts=["t1"], future=f1))
        await batcher.add(BatchItem(job_id="j2", texts=["t2"], future=f2))

        await asyncio.wait_for(asyncio.gather(f1, f2), timeout=2)
        assert len(batches) == 1
        assert len(batches[0]) == 2
        await batcher.stop()

    async def test_flush_on_timeout(self):
        batches = []

        async def on_batch(batch):
            batches.append(batch)
            for item in batch:
                item.future.set_result("ok")

        batcher = DynamicBatcher(on_batch=on_batch, max_size=100, timeout_ms=100)
        await batcher.start()

        loop = asyncio.get_event_loop()
        f = loop.create_future()
        await batcher.add(BatchItem(job_id="j1", texts=["t1"], future=f))

        await asyncio.wait_for(f, timeout=2)
        assert len(batches) == 1
        await batcher.stop()

    async def test_future_resolves(self):
        async def on_batch(batch):
            for item in batch:
                item.future.set_result("result_value")

        batcher = DynamicBatcher(on_batch=on_batch, max_size=1, timeout_ms=5000)
        await batcher.start()

        loop = asyncio.get_event_loop()
        f = loop.create_future()
        await batcher.add(BatchItem(job_id="j1", texts=["hello"], future=f))

        result = await asyncio.wait_for(f, timeout=2)
        assert result == "result_value"
        await batcher.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# JobQueue tests
# ═══════════════════════════════════════════════════════════════════════════════
from job_queue import JobQueue, QueueFullError
from schemas import Job, JobStatus


@pytest.mark.asyncio
class TestJobQueue:
    async def test_submit_and_get(self):
        q = JobQueue()
        job = Job(inputs="hello")
        await q.submit(job)
        stored = await q.get(job.job_id)
        assert stored is not None
        assert stored.job_id == job.job_id

    async def test_queue_depth(self):
        q = JobQueue()
        assert await q.depth() == 0
        await q.submit(Job(inputs="a"))
        assert await q.depth() == 1

    async def test_update_status(self):
        q = JobQueue()
        job = Job(inputs="hello")
        await q.submit(job)
        job.status = JobStatus.DONE
        job.outputs = {"result": 42}
        await q.update(job)
        stored = await q.get(job.job_id)
        assert stored.status == JobStatus.DONE

    async def test_queue_full_raises(self):
        import os
        os.environ["QUEUE_MAX_SIZE"] = "1"
        from importlib import reload
        import queue as qmod
        reload(qmod)
        q = qmod.JobQueue()
        await q.submit(Job(inputs="a"))
        with pytest.raises(QueueFullError):
            await q.submit(Job(inputs="b"))
        os.environ.pop("QUEUE_MAX_SIZE", None)

    async def test_get_missing_returns_none(self):
        q = JobQueue()
        result = await q.get("nonexistent-id")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# API endpoint tests
# ═══════════════════════════════════════════════════════════════════════════════
from app_phase_b import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestPhaseBEndpoints:
    def test_health_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_schema(self, client):
        body = client.get("/health").json()
        for key in ("status", "model_loaded", "device", "queue_depth", "uptime_s"):
            assert key in body

    def test_models_info_200(self, client):
        r = client.get("/models/info")
        assert r.status_code == 200

    def test_predict_sync_string(self, client):
        r = client.post("/predict", json={"inputs": "classify this"})
        assert r.status_code == 200
        body = r.json()
        assert "outputs" in body
        assert "latency_ms" in body
        assert "batch_size" in body

    def test_predict_sync_list(self, client):
        r = client.post("/predict", json={"inputs": ["text one", "text two"]})
        assert r.status_code == 200

    def test_predict_sync_empty_string(self, client):
        r = client.post("/predict", json={"inputs": ""})
        assert r.status_code == 422

    def test_predict_sync_null(self, client):
        r = client.post("/predict", json={"inputs": None})
        assert r.status_code == 422

    def test_predict_async_returns_202(self, client):
        r = client.post("/predict/async", json={"inputs": "async test"})
        assert r.status_code == 202

    def test_predict_async_returns_job_id(self, client):
        body = client.post("/predict/async", json={"inputs": "async test"}).json()
        assert "job_id" in body
        assert body["status"] == "pending"

    def test_job_polling(self, client):
        body = client.post("/predict/async", json={"inputs": "polling test"}).json()
        job_id = body["job_id"]
        # Poll up to 5 s
        import time
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            r = client.get(f"/jobs/{job_id}")
            assert r.status_code == 200
            status = r.json()["status"]
            if status in ("done", "failed"):
                break
            time.sleep(0.1)
        assert status == "done"

    def test_job_not_found(self, client):
        r = client.get("/jobs/does-not-exist")
        assert r.status_code == 404

    def test_html_input_cleaned(self, client):
        r = client.post("/predict", json={"inputs": "<b>hello</b> world"})
        assert r.status_code == 200

    def test_openapi_docs(self, client):
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200
