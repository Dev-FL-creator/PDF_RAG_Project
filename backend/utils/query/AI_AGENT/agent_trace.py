import json
import time
import asyncio
import threading
from typing import Dict, List
from fastapi.responses import StreamingResponse


_agent_traces: Dict[str, List[str]] = {}
_trace_lock = threading.Lock()


def get_trace_stream_response(trace_id: str) -> StreamingResponse:
    async def event_generator():
        last_index = 0
        max_wait_time = 120
        start_time = time.time()

        while True:
            if time.time() - start_time > max_wait_time:
                yield f"data: {json.dumps({'status': 'timeout'})}\n\n"
                break

            with _trace_lock:
                if trace_id in _agent_traces:
                    trace_lines = _agent_traces[trace_id]

                    while last_index < len(trace_lines):
                        line = trace_lines[last_index]
                        yield f"data: {json.dumps({'line': line})}\n\n"
                        last_index += 1

                        if "=== Agent Decision End ===" in line or "=== PDF Agent Decision End ===" in line:
                            yield f"data: {json.dumps({'status': 'done'})}\n\n"
                            del _agent_traces[trace_id]
                            return

            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")