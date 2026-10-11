import React, { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

// Backend base URL. Set VITE_API_URL in the host project's .env to override.
const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export default function InsightAIUI() {
  const [apiKey, setApiKey] = useState("");
  const [pdf, setPdf] = useState(null);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  // The key is typed in by the user and kept in memory only, never baked into the bundle.
  const request = async (path, options) => {
    setLoading(true);
    setStatus("");
    try {
      const res = await fetch(`${API_URL}${path}`, {
        ...options,
        headers: { ...options.headers, "X-API-Key": apiKey },
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail ?? `Request failed (${res.status})`);
      return data;
    } catch (err) {
      setStatus(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async () => {
    if (!pdf) return;
    const formData = new FormData();
    formData.append("file", pdf);
    const data = await request("/upload/", { method: "POST", body: formData, headers: {} });
    if (data) setStatus(`Indexed ${data.filename} (${data.chunks} chunks)`);
  };

  const handleQuery = async () => {
    if (!query) return;
    const data = await request("/query/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 3 }),
    });
    if (data) setResponse(data);
  };

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-4">
      <h1 className="text-2xl font-bold">InsightAI</h1>

      <Card>
        <CardContent className="space-y-2 py-4">
          <Input type="password" placeholder="API key" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
          <Input type="file" accept="application/pdf" onChange={(e) => setPdf(e.target.files[0])} />
          <Button onClick={handleUpload} disabled={loading}>
            {loading ? "Working..." : "Upload PDF"}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="space-y-2 py-4">
          <Textarea
            rows={3}
            placeholder="Ask a question about your uploaded PDFs"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <Button onClick={handleQuery} disabled={loading}>
            {loading ? "Working..." : "Submit Query"}
          </Button>
        </CardContent>
      </Card>

      {status && <p className="text-sm" role="status">{status}</p>}

      {response && (
        <Card>
          <CardContent className="py-4 space-y-2">
            <h2 className="font-semibold">Answer</h2>
            <p>{response.answer}</p>
            {response.results.map((r, i) => (
              <p key={i} className="text-sm text-muted-foreground">
                <strong>{r.source}</strong> ({r.score}): {r.text.slice(0, 200)}...
              </p>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
