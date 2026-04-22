import { useState, useEffect, useRef, useMemo } from "react";
import {
  Row, Col, Card, Form, Button, Spinner, Alert,
  OverlayTrigger, Popover, Modal, Dropdown
} from "react-bootstrap";
import { InfoCircle, X as XIcon } from "react-bootstrap-icons";
import { uploadPdfToIndex, queryPdfIndex } from "./lib/api";

/** Small reusable info popover */
function InfoIcon({ title, children, placement = "right", maxWidth = 520 }) {
  const overlay = (
    <Popover style={{ maxWidth }}>
      <Popover.Header as="h3">{title}</Popover.Header>
      <Popover.Body>{children}</Popover.Body>
    </Popover>
  );
  return (
    <OverlayTrigger trigger={["hover", "focus"]} placement={placement} overlay={overlay} rootClose>
      <Button variant="outline-secondary" size="sm" className="ms-1" aria-label={`${title} help`}>
        <InfoCircle />
      </Button>
    </OverlayTrigger>
  );
}

// Validate index name
function validateIndexName(name) {
  if (!name) return false;
  if (name.length < 2 || name.length > 128) return false;
  if (!/^[a-z0-9][a-z0-9-_]*$/.test(name)) return false;
  if (/--|__/.test(name)) return false;
  if (name.endsWith("-")) return false;
  return true;
}

// Create PDF index
async function createPdfIndex(apiBase, headers, body) {
  const res = await fetch(`${apiBase}/api/pdf_rag/create_index`, {
    method: "POST",
    headers: { ...headers, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const txt = await res.text();
  let j = {};
  try { j = JSON.parse(txt || "{}"); } catch {}
  if (!res.ok || j.ok === false) {
    throw new Error(j?.detail || j?.msg || txt || `HTTP ${res.status}`);
  }
  return j;
}

// Delete an existing index
async function deleteIndexApi(apiBase, headers, name) {
  const res = await fetch(`${apiBase}/api/indexes/${encodeURIComponent(name)}`, {
    method: "DELETE",
    headers,
  });
  const txt = await res.text();
  let j = {};
  try { j = JSON.parse(txt || "{}"); } catch {}
  if (!res.ok || j.ok === false) {
    throw new Error(j?.detail || j?.msg || txt || `HTTP ${res.status}`);
  }
  return j;
}

export default function RAGPage({ S, toast }) {
  const [indexName, setIndexName] = useState("");
  const [indexList, setIndexList] = useState([]);
  const [useCustomIdx, setUseCustomIdx] = useState(false);

  const [file, setFile] = useState(null);
  const fileInputRef = useRef(null);
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);

  const [retrieved, setRetrieved] = useState([]);
  const [answer, setAnswer] = useState("");
  const [chunks, setChunks] = useState(0);

  const [busyUpload, setBusyUpload] = useState(false);
  const [busyQuery, setBusyQuery] = useState(false);

  const [useAgent, setUseAgent] = useState(false);
  const [agentLiveTrace, setAgentLiveTrace] = useState("");
  const eventSourceRef = useRef(null);

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [pendingIndexName, setPendingIndexName] = useState("");
  const [creating, setCreating] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [indexToDelete, setIndexToDelete] = useState("");
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${S.apiBase}/api/indexes`, { headers: S.headers });
        if (!res.ok) return;
        const j = await res.json();
        if (j?.ok && Array.isArray(j.indexes)) {
          setIndexList(j.indexes);
          if (!useCustomIdx && !indexName && j.indexes.length > 0) {
            setIndexName(j.indexes[0]);
          }
        }
      } catch {}
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [S.apiBase, S.headers, useCustomIdx]);

  async function handleCreateIndex() {
    const name = (indexName || "").trim();
    if (!validateIndexName(name)) {
      toast("warning", "Invalid index name. Use lowercase letters/numbers/-/_");
      return;
    }
    
    try {
      const probe = await createPdfIndex(S.apiBase, S.headers, {
        index_name: name,
        vector_dimensions: 1536,
        recreate: false,
      });
      if (probe?.status === "exists") {
        setPendingIndexName(name);
        setConfirmOpen(true);
        return;
      }
      
      toast("success", `Index created: ${probe?.index_name || name}`);
      if (!indexList.includes(name)) setIndexList((lst) => [...lst, name]);
    } catch (e) {
      toast("danger", "Create index failed: " + (e?.message || e));
    }
  }

  function confirmContinueUpload() {
    setConfirmOpen(false);
    toast("info", `Index "${pendingIndexName}" exists. You can continue to upload to it.`);
    try {
      fileInputRef.current?.focus?.();
      fileInputRef.current?.click?.();
    } catch {}
  }

  async function confirmRecreate() {
    if (!pendingIndexName) return;
    setCreating(true);
    try {
      const j = await createPdfIndex(S.apiBase, S.headers, {
        index_name: pendingIndexName,
        vector_dimensions: 1536,
        recreate: true,
      });
      toast("success", `Recreated index: ${j?.index_name || pendingIndexName}`);
      setConfirmOpen(false);
      setPendingIndexName("");
      if (!indexList.includes(pendingIndexName)) {
        setIndexList((lst) => [...lst, pendingIndexName]);
      }
    } catch (e) {
      toast("danger", "Recreate failed: " + (e?.message || e));
    } finally {
      setCreating(false);
    }
  }

  function openDeleteModal(name, e) {
    e?.preventDefault?.();
    e?.stopPropagation?.();
    setIndexToDelete(name);
    setDeleteOpen(true);
  }

  async function confirmDelete() {
    if (!indexToDelete) return;
    setDeleting(true);
    try {
      await deleteIndexApi(S.apiBase, S.headers, indexToDelete);
      toast("success", `Deleted index: ${indexToDelete}`);
      setIndexList((lst) => {
        const rest = lst.filter((n) => n !== indexToDelete);
        setIndexName((cur) => (cur === indexToDelete ? (rest[0] || "") : cur));
        return rest;
      });
      setDeleteOpen(false);
      setIndexToDelete("");
    } catch (e) {
      toast("danger", `Delete failed: ${e?.message || e}`);
    } finally {
      setDeleting(false);
    }
  }

  async function handleUpload() {
    const idx = (indexName || "").trim();
    if (!file || !idx) {
      toast("warning", "Please select a PDF and enter/select an index name");
      return;
    }

    setBusyUpload(true);
    setChunks(0);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("index_name", idx);
      const j = await uploadPdfToIndex(S.apiBase, S.headers, fd);
      setChunks(j.chunks_uploaded);
      toast("success", `Indexed ${j.chunks_uploaded} chunks into ${j.index_name}`);
      if (!indexList.includes(idx)) setIndexList((lst) => [...lst, idx]);
    } catch (e) {
      toast("danger", "Upload failed: " + e.message);
    } finally {
      setBusyUpload(false);
    }
  }

  async function handleQuery() {
    const idx = (indexName || "").trim();
    if (!idx) {
      toast("warning", "Enter or select an index name first");
      return;
    }

    setBusyQuery(true);
    setRetrieved([]);
    setAnswer("");
    setAgentLiveTrace("");
    
    try {
      let traceId = null;
      
      if (useAgent) {
        traceId = `trace_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
        }
        setAgentLiveTrace("🤖 Agent Mode activated\nConnecting to real-time stream...\n\n");
        const eventSource = new EventSource(`${S.apiBase}/api/pdf_rag/agent_trace_stream/${traceId}`);
        eventSourceRef.current = eventSource;
        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.status === 'done' || data.status === 'timeout') {
              eventSource.close();
              eventSourceRef.current = null;
              if (data.status === 'timeout') {
                setAgentLiveTrace(prev => prev + "\n⏱️ Stream timeout\n");
              }
            } else if (data.line) {
              setAgentLiveTrace(prev => prev + data.line + '\n');
            }
          } catch (err) {
            console.error('Error parsing SSE data:', err);
          }
        };
        eventSource.onerror = (err) => {
          console.error('SSE connection error:', err);
          eventSource.close();
          eventSourceRef.current = null;
        };
      }
      
      const body = { 
        index_name: idx, 
        query, 
        top_k: Number(topK) || 5,
        use_agent: useAgent,
        trace_id: traceId
      };
      const j = await queryPdfIndex(S.apiBase, S.headers, body);

      let ans = (j.answer || "").toString().trim();
      const isNotFound = ans.toLowerCase().startsWith("not found");
      setRetrieved(isNotFound ? ["Not found in the document."] : (j.retrieved || []));
      setAnswer(ans || "(no answer)");
      if (useAgent && j.agent_trace) {
        setAgentLiveTrace(prev => (prev ? prev + j.agent_trace : j.agent_trace));
      }
    } catch (e) {
      toast("danger", "Query failed: " + e.message);
    } finally {
      setBusyQuery(false);
    }
  }

  function renderRetrieved(items) {
    if (!items || items.length === 0) {
      return (
        <div style={{ color: "gray", fontStyle: "italic" }}>
          (retrieved contexts will show here)
        </div>
      );
    }
    return items.map((s, i) => {
      const m = s.match(/^\[([^\]]+)\]\s*/);
      if (m) {
        const rest = s.slice(m[0].length);
        return (
          <div key={i} className="mb-3">
            <strong>[{m[1]}]</strong>{" "}
            <span style={{ whiteSpace: "pre-wrap" }}>{rest}</span>
          </div>
        );
      }
      return (
        <div key={i} className="mb-3" style={{ whiteSpace: "pre-wrap" }}>
          {s}
        </div>
      );
    });
  }

  const indexInfo = useMemo(() => (
    <Popover id="pdf-index-info" style={{ maxWidth: 520 }}>
      <Popover.Header as="h3">Manage PDF Index</Popover.Header>
      <Popover.Body>
        <p>Select an existing index or choose <b>Custom…</b> to type a new one.</p>
        <ul>
          <li><b>Create Index</b> – Type a new index name and click Create Index</li>
          <li><b>Continue & Upload</b> – Upload to existing index</li>
          <li><b>Delete & Recreate</b> – Overwrite existing index</li>
          <li><b>Delete</b> – Click X to remove index</li>
        </ul>
      </Popover.Body>
    </Popover>
  ), []);

  return (
    <Card className="shadow-sm" style={{ overflow: 'visible', minHeight: '85vh' }}>
      <Card.Body style={{ overflow: 'visible', padding: '1.5rem' }}>
        <Row className="g-4" style={{ height: '100%' }}>
          <Col md={4}>
            <Form.Group className="mb-3">
              <Form.Label>
                Index Name
                <InfoIcon title="Index Name">
                  Select an existing Azure AI Search index or create a new one.
                </InfoIcon>
              </Form.Label>

              {!useCustomIdx ? (
                <div style={{ maxWidth: 460, position: 'relative', zIndex: 10 }}>
                  <Dropdown>
                    <Dropdown.Toggle id="pdf-index-dropdown" className="w-100" variant="light">
                      <span className="text-truncate">{indexName || "(none)"}</span>
                    </Dropdown.Toggle>
                    <Dropdown.Menu className="w-100" style={{ maxHeight: '250px', overflowY: 'auto' }}>
                      {indexList.map((name) => (
                        <Dropdown.Item key={name} as="div" className="d-flex justify-content-between align-items-center" onClick={() => setIndexName(name)}>
                          <span className="text-truncate me-2">{name}</span>
                          <Button variant="link" size="sm" className="text-danger p-0" onClick={(e) => openDeleteModal(name, e)}>
                            <XIcon />
                          </Button>
                        </Dropdown.Item>
                      ))}
                      <Dropdown.Divider />
                      <Dropdown.Item onClick={() => { setUseCustomIdx(true); setIndexName(""); }}>
                        Custom…
                      </Dropdown.Item>
                    </Dropdown.Menu>
                  </Dropdown>
                </div>
              ) : (
                <div style={{ maxWidth: 460 }}>
                  <Form.Control placeholder="Type a custom index name" value={indexName} onChange={(e) => setIndexName(e.target.value)} />
                  <Button variant="link" size="sm" className="p-0 mt-1" onClick={() => { setUseCustomIdx(false); if (indexList[0]) setIndexName(indexList[0]); }}>
                    ← Back to list
                  </Button>
                </div>
              )}

              <div className="d-flex gap-2 mt-2">
                <Button variant="danger" onClick={handleCreateIndex} disabled={creating}>Create Index</Button>
                <OverlayTrigger trigger={["hover", "focus"]} placement="right" overlay={indexInfo} rootClose>
                  <Button variant="outline-secondary" size="sm"><InfoCircle /></Button>
                </OverlayTrigger>
              </div>
            </Form.Group>

            <Form.Group className="mb-3">
              <Form.Label>Upload PDF</Form.Label>
              <Form.Control ref={fileInputRef} type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files?.[0] || null)} />
            </Form.Group>

            <Button variant="primary" disabled={busyUpload} onClick={handleUpload}>
              {busyUpload ? <><Spinner size="sm" className="me-1" />Working...</> : "Upload & Index"}
            </Button>

            {chunks > 0 && <Alert className="mt-2" variant="success">✅ Indexed {chunks} chunks</Alert>}

            <hr />

            <Form.Group className="mb-3">
              <Form.Label>Question</Form.Label>
              <Form.Control as="textarea" rows={3} value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ask about the PDF" />
            </Form.Group>

            <Form.Group className="mb-3">
              <Form.Label>Top-K</Form.Label>
              <Form.Control type="number" min={1} max={20} value={topK} onChange={(e) => setTopK(Number(e.target.value) || 3)} />
            </Form.Group>

            <Form.Group className="mb-3">
              <Form.Label>🤖 Agent Mode</Form.Label>
              <div>
                <Button variant={useAgent ? "info" : "outline-secondary"} size="sm" onClick={() => setUseAgent(!useAgent)}>
                  {useAgent ? "✓ ON" : "OFF"}
                </Button>
              </div>
            </Form.Group>

            <Button variant="success" disabled={busyQuery} onClick={handleQuery}>
              {busyQuery ? <><Spinner size="sm" className="me-1" />Searching...</> : "Ask"}
            </Button>
          </Col>

          <Col md={8}>
            <Form.Group className="mb-3">
              <Form.Label>Retrieved Context</Form.Label>
              <div style={{ border: "1px solid #ddd", borderRadius: 4, padding: 12, minHeight: 200, maxHeight: 350, overflowY: "auto", backgroundColor: "#f8f9fa" }}>
                {renderRetrieved(retrieved)}
              </div>
            </Form.Group>

            <Form.Group className="mb-2">
              <Form.Label>Answer</Form.Label>
              <Form.Control as="textarea" rows={10} value={answer} readOnly style={{ fontSize: "1.05em" }} />
            </Form.Group>

            {useAgent && agentLiveTrace && (
              <Form.Group className="mb-2">
                <Form.Label>🤖 Agent Thinking Process</Form.Label>
                <div style={{ border: "1px solid #ddd", borderRadius: 4, padding: 8, minHeight: 150, maxHeight: 400, overflowY: "auto", backgroundColor: "#f8f9fa", fontFamily: "monospace", fontSize: "0.85em", whiteSpace: "pre-wrap" }}>
                  {agentLiveTrace}
                </div>
              </Form.Group>
            )}
          </Col>
        </Row>
      </Card.Body>

      <Modal show={confirmOpen} onHide={() => !creating && setConfirmOpen(false)} centered>
        <Modal.Header closeButton={!creating}>
          <Modal.Title>Index already exists</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          The index <code>{pendingIndexName}</code> already exists. Choose an action:
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setConfirmOpen(false)} disabled={creating}>Cancel</Button>
          <Button variant="primary" onClick={confirmContinueUpload} disabled={creating}>Continue & Upload</Button>
          <Button variant="danger" onClick={confirmRecreate} disabled={creating}>
            {creating ? "Recreating..." : "Delete & Recreate"}
          </Button>
        </Modal.Footer>
      </Modal>

      <Modal show={deleteOpen} onHide={() => !deleting && setDeleteOpen(false)} centered>
        <Modal.Header closeButton={!deleting}>
          <Modal.Title>Delete index</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          Are you sure you want to delete index <code>{indexToDelete}</code>? This cannot be undone.
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setDeleteOpen(false)} disabled={deleting}>Cancel</Button>
          <Button variant="danger" onClick={confirmDelete} disabled={deleting}>
            {deleting ? "Deleting..." : "Delete"}
          </Button>
        </Modal.Footer>
      </Modal>
    </Card>
  );
}
