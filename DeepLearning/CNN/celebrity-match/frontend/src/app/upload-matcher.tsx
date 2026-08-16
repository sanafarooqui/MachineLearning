// Needs useState + onChange, so this must be a Client Component — it can't
// be a (server-rendered) default Next.js component.
"use client";

import { useState } from "react";

type Match = {
  celebrity_name: string;
  similarity_score: number;
};

// NEXT_PUBLIC_ prefix is required for a Next.js env var to be readable in
// client-side code at all; falls back to the local FastAPI dev port.
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function UploadMatcher() {
  // Object URL for showing the chosen file immediately, before the API call
  // even starts.
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [matches, setMatches] = useState<Match[] | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setPreviewUrl(URL.createObjectURL(file));
    setMatches(null);
    setErrorMessage(null);
    setStatus("loading");

    // multipart/form-data body — matches what FastAPI's UploadFile expects
    // on the other end.
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/match`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        // Surface the API's error detail (e.g. "No face detected...")
        // directly instead of a generic failure message.
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail ?? `Request failed (${response.status})`);
      }

      const data: { matches: Match[] } = await response.json();
      setMatches(data.matches);
      setStatus("idle");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Something went wrong.");
      setStatus("error");
    }
  }

  return (
    <div className="flex w-full max-w-md flex-col items-center gap-6">
      <label
        htmlFor="selfie-upload"
        className="flex w-full cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed border-neutral-300 p-8 text-center transition hover:border-neutral-400 dark:border-neutral-700 dark:hover:border-neutral-500"
      >
        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt="Selfie preview"
            className="h-40 w-40 rounded-lg object-cover"
          />
        ) : (
          <span className="text-sm text-neutral-500">
            Click to upload a selfie (JPEG, PNG, or WebP)
          </span>
        )}
        <input
          id="selfie-upload"
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="hidden"
          onChange={handleFileChange}
        />
      </label>

      {status === "loading" && (
        <p className="text-sm text-neutral-500">Finding your match…</p>
      )}

      {status === "error" && errorMessage && (
        <p className="text-sm text-red-500">{errorMessage}</p>
      )}

      {matches && matches.length > 0 && (
        <ul className="flex w-full flex-col gap-3">
          {matches.map((match, index) => (
            <li
              key={match.celebrity_name}
              className="flex items-center justify-between rounded-lg border border-neutral-200 px-4 py-3 dark:border-neutral-800"
            >
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-neutral-400">
                  #{index + 1}
                </span>
                <span className="font-medium">{match.celebrity_name}</span>
              </div>
              <span className="text-sm text-neutral-500">
                {(match.similarity_score * 100).toFixed(1)}%
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
