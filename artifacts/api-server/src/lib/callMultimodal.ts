// ── callMultimodal ────────────────────────────────────────────────────────────
// Unified multimodal sender. Knows HOW to send to each provider format.
// Does NOT know which provider is active, which keys to use, or which to try.
// Caller builds MultimodalContext; this function executes it.

import { callVertexAiMultimodal, type VertexAiConfig } from "./vertexAi.js";

export type MultimodalFormat =
  | "whisper"        // OpenAI /v1/audio/transcriptions (FormData binary)
  | "gemini-inline"  // Gemini REST API (base64 inline_data)
  | "vertex"         // Vertex AI Multimodal (base64 via service account)
  | "openai-vision"  // OpenAI-compatible vision (base64 image_url)
  | "anthropic";     // Anthropic messages vision (base64 source)

export interface MultimodalContext {
  endpoint: string;       // full URL (unused for vertex — uses vertexConfig)
  apiKey: string;
  model: string;
  format: MultimodalFormat;
  whisperModel?: string;  // only for "whisper" format
  vertexConfig?: VertexAiConfig; // only for "vertex" format
}

/**
 * Send media to a provider and return the plain-text response.
 * Returns null on any failure — caller decides whether to try next context.
 *
 * @param context  - Provider endpoint, key, model, and format
 * @param buffer   - Raw media bytes (ArrayBuffer) — converted internally per format
 * @param mimeType - MIME type of the media (e.g. "audio/ogg", "image/jpeg")
 * @param prompt   - Instruction for the model
 * @param timeoutMs - Request timeout (default 25 s)
 */
export async function callMultimodal(
  context: MultimodalContext,
  buffer: ArrayBuffer,
  mimeType: string,
  prompt: string,
  timeoutMs = 25_000,
): Promise<string | null> {
  try {
    switch (context.format) {

      // ── Whisper: audio transcription via multipart/form-data ────────────────
      case "whisper": {
        const ext  = mimeType.split("/")[1] ?? "ogg";
        const blob = new Blob([buffer], { type: mimeType });
        const form = new FormData();
        form.append("file", blob, `audio.${ext}`);
        form.append("model", context.whisperModel ?? "whisper-1");

        const resp = await fetch(context.endpoint, {
          method: "POST",
          headers: { Authorization: `Bearer ${context.apiKey}` },
          body: form,
          signal: AbortSignal.timeout(timeoutMs),
        });
        if (!resp.ok) {
          console.warn(`[callMultimodal] whisper ${resp.status}: ${await resp.text().catch(() => "")}`);
          return null;
        }
        const data = await resp.json() as { text?: string };
        return data.text?.trim() || null;
      }

      // ── Gemini inline_data: audio / image / video via REST ──────────────────
      case "gemini-inline": {
        const base64 = Buffer.from(buffer).toString("base64");
        const resp = await fetch(context.endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{
              parts: [
                { text: prompt },
                { inline_data: { mime_type: mimeType, data: base64 } },
              ],
            }],
            generationConfig: { temperature: 0.1, maxOutputTokens: 256 },
          }),
          signal: AbortSignal.timeout(timeoutMs),
        });
        if (!resp.ok) {
          console.warn(`[callMultimodal] gemini-inline ${resp.status}: ${await resp.text().catch(() => "")}`);
          return null;
        }
        const data = await resp.json() as {
          candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
        };
        return (data.candidates?.[0]?.content?.parts?.[0]?.text ?? "").trim() || null;
      }

      // ── Vertex AI: multimodal via service account ───────────────────────────
      case "vertex": {
        if (!context.vertexConfig) {
          console.warn("[callMultimodal] vertex format requires vertexConfig");
          return null;
        }
        const base64 = Buffer.from(buffer).toString("base64");
        console.log(`[callMultimodal] vertex calling with mimeType=${mimeType} bufferSize=${buffer.byteLength} model=${context.vertexConfig.modelName}`);
        const text   = await callVertexAiMultimodal(
          context.vertexConfig, prompt, base64, mimeType, timeoutMs,
        );
        console.log(`[callMultimodal] vertex returned: "${String(text).substring(0, 80)}"`);
        return text?.trim() || null;
      }

      // ── OpenAI-compatible vision: image_url base64 ──────────────────────────
      case "openai-vision": {
        const base64 = Buffer.from(buffer).toString("base64");
        const resp = await fetch(context.endpoint, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${context.apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: context.model,
            max_tokens: 256,
            messages: [{
              role: "user",
              content: [
                { type: "text", text: prompt },
                { type: "image_url", image_url: { url: `data:${mimeType};base64,${base64}` } },
              ],
            }],
          }),
          signal: AbortSignal.timeout(timeoutMs),
        });
        if (!resp.ok) {
          console.warn(`[callMultimodal] openai-vision ${resp.status}`);
          return null;
        }
        const raw = await resp.text();
        if (raw.trim().startsWith("<")) return null; // HTML error page
        const data = JSON.parse(raw) as {
          choices?: Array<{ message?: { content?: string } }>;
          error?: unknown;
        };
        if (data.error) return null;
        return (data.choices?.[0]?.message?.content ?? "").trim() || null;
      }

      // ── Anthropic vision: base64 source block ───────────────────────────────
      case "anthropic": {
        const base64 = Buffer.from(buffer).toString("base64");
        const resp = await fetch(context.endpoint, {
          method: "POST",
          headers: {
            "x-api-key": context.apiKey,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
          },
          body: JSON.stringify({
            model: context.model,
            max_tokens: 256,
            messages: [{
              role: "user",
              content: [
                { type: "image", source: { type: "base64", media_type: mimeType, data: base64 } },
                { type: "text", text: prompt },
              ],
            }],
          }),
          signal: AbortSignal.timeout(timeoutMs),
        });
        if (!resp.ok) {
          console.warn(`[callMultimodal] anthropic ${resp.status}`);
          return null;
        }
        const data = await resp.json() as {
          content?: Array<{ text?: string }>;
          error?: unknown;
        };
        if (data.error) return null;
        return (data.content?.[0]?.text ?? "").trim() || null;
      }

      default:
        return null;
    }
  } catch (err) {
    console.warn(`[callMultimodal] ${context.format} threw:`, (err as Error).message);
    return null;
  }
}
