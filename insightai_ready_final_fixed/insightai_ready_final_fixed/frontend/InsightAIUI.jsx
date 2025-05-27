import React, { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

export default function InsightAIUI() {
  const [pdf, setPdf] = useState(null);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setPdf(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!pdf) return;
    const formData = new FormData();
    formData.append("file", pdf);
    setLoading(true);
    const res = await fetch("http://localhost:8000/upload/", {
      method: "POST",
      body: formData,
    });
    setLoading(false);
    if (res.ok) alert("Upload successful");
  };

  const handleQuery = async () => {
    if (!query) return;
    setLoading(true);
    const res = await fetch("http://localhost:8000/query/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 1 }),
    });
    const data = await res.json();
    setResponse(data);
    setLoading(false);
  };

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-4">
      <h1 className="text-2xl font-bold">📄 InsightAI Interface</h1>

      <Card>
        <CardContent className="space-y-2 py-4">
          <Input type="file" accept="application/pdf" onChange={handleFileChange} />
          <Button onClick={handleUpload} disabled={loading}>
            {loading ? "Uploading..." : "Upload PDF"}
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
            {loading ? "Searching..." : "Submit Query"}
          </Button>
        </CardContent>
      </Card>

      {response && (
        <Card>
          <CardContent className="py-4 space-y-2">
            <h2 className="font-semibold">Response</h2>
            <p><strong>Answer:</strong> {response.answer}</p>
            <p className="text-sm text-muted-foreground">
              <strong>Matched Text:</strong> {response.results[0]?.text.slice(0, 200)}...
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}