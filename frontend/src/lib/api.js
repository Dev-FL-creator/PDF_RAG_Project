// API helper functions for PDF RAG

// Upload PDF to index
export async function uploadPdfToIndex(apiBase, headers, formData) {
  const res = await fetch(`${apiBase}/api/pdf_rag/upload`, {
    method: "POST",
    headers: { ...headers },
    body: formData
  });
  const txt = await res.text();
  let j = {};
  try { j = JSON.parse(txt || "{}"); } catch {}
  if (!res.ok || j.ok === false) {
    throw new Error(j?.detail || j?.msg || txt || `HTTP ${res.status}`);
  }
  return j;
}

// Query PDF index
export async function queryPdfIndex(apiBase, headers, body) {
  const res = await fetch(`${apiBase}/api/pdf_rag/query`, {
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
