// ── AI Provider Engine ────────────────────────────────────────────────────────
// Manages all AI provider calls: load balancing, retry logic, failover.
// Supports OpenAI-compatible, Anthropic-compatible, Vertex AI, and raw endpoints.
// Extracted from ai.ts for clarity; re-exported via ai.ts for backward compat.

import {
  db,
  aiProvidersTable,
  providerUsageLogTable,
} from "@workspace/db";
import { eq, and, asc, sql } from "drizzle-orm";
import { decrypt } from "./encryption.js";
import { detectApiFormat, callWithFormat, resolveProviderType } from "./apiTransformer.js";
import { callVertexAi, parseVertexConfig } from "./vertexAi.js";

export type Message = { role: "user" | "assistant"; content: string };

// ── callAIWithMetadata: load-balanced call + returns provider/model info ──────
export async function callAIWithMetadata(
  messages: Message[],
  systemPrompt: string
): Promise<{ text: string; providerName: string; modelName: string }> {
  const enabledProviders = await db
    .select()
    .from(aiProvidersTable)
    .where(and(eq(aiProvidersTable.isEnabled, 1)))
    .orderBy(
      asc(aiProvidersTable.priority),
      asc(sql`COALESCE(${aiProvidersTable.lastUsedAt}, '1970-01-01T00:00:00.000Z')`)
    );

  const errors: string[] = [];

  for (const provider of enabledProviders) {
    const start = Date.now();
    let lastErrMsg = "";

    // ── Retry loop: up to 2 attempts, retry once on 429 with 2s delay ────────
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        if (attempt > 0) {
          console.warn(`[ai] Retrying provider "${provider.name}" after 429 (attempt ${attempt + 1})…`);
          await new Promise<void>(r => setTimeout(r, 2000));
        }
        const result = await callSingleProvider(provider, messages, systemPrompt);
        const latency = Date.now() - start;
        await db.update(aiProvidersTable)
          .set({ failCount: 0, lastUsedAt: new Date().toISOString() })
          .where(eq(aiProvidersTable.id, provider.id));
        await db.insert(providerUsageLogTable).values({
          providerId: provider.id, success: 1, latencyMs: latency, error: null,
          createdAt: new Date().toISOString(),
        });
        return { text: result, providerName: provider.name, modelName: provider.modelName };
      } catch (err: unknown) {
        lastErrMsg = err instanceof Error ? err.message : String(err);
        const lowered = lastErrMsg.toLowerCase();
        const isRateLimit = lowered.includes("429")
          || lowered.includes("resource_exhausted")
          || lowered.includes("resource has been exhausted")
          || lowered.includes("quota exceeded")
          || lowered.includes("rate limit")
          || lowered.includes("too many requests");
        if (!isRateLimit || attempt >= 1) break;
      }
    }

    // Provider exhausted (both attempts failed)
    const latency = Date.now() - start;
    errors.push(`[${provider.name}] ${lastErrMsg}`);
    await db.update(aiProvidersTable)
      .set({ failCount: sql`${aiProvidersTable.failCount} + 1`, lastUsedAt: new Date().toISOString() })
      .where(eq(aiProvidersTable.id, provider.id));
    await db.insert(providerUsageLogTable).values({
      providerId: provider.id, success: 0, latencyMs: latency,
      error: lastErrMsg.substring(0, 500), createdAt: new Date().toISOString(),
    });
  }

  // Fallback: single active provider
  if (enabledProviders.length === 0) {
    const [activeProvider] = await db
      .select().from(aiProvidersTable)
      .where(eq(aiProvidersTable.isActive, 1)).limit(1);
    if (!activeProvider) throw new Error("No active AI provider configured");
    const text = await callSingleProvider(activeProvider, messages, systemPrompt);
    return { text, providerName: activeProvider.name, modelName: activeProvider.modelName };
  }

  throw new Error(`All ${enabledProviders.length} providers failed: ${errors.join(" | ")}`);
}

// ── callAI: simple call using the single active provider (no load balancing) ──
export async function callAI(
  messages: Message[],
  systemPrompt: string
): Promise<string> {
  const [activeProvider] = await db
    .select()
    .from(aiProvidersTable)
    .where(eq(aiProvidersTable.isActive, 1))
    .limit(1);

  if (!activeProvider) {
    throw new Error("No active AI provider configured");
  }

  const apiKey = decrypt(activeProvider.apiKey);
  if (!apiKey) {
    throw new Error("AI provider API key is not configured");
  }

  try {
    const rawType    = activeProvider.providerType.toLowerCase();
    const rawTypeKey = rawType.replace(/\s+/g, "");
    const url        = (activeProvider.baseUrl ?? "").toLowerCase();
    const apiFormat  = detectApiFormat(rawTypeKey);

    if (apiFormat === "raw_single" || apiFormat === "raw_messages") {
      const endpointUrl = activeProvider.baseUrl ?? "";
      if (!endpointUrl) throw new Error("Raw API provider requires a full endpoint URL in Base URL field");
      const result = await callWithFormat(apiFormat, {
        apiKey,
        baseUrl: endpointUrl,
        model: activeProvider.modelName,
        systemPrompt,
        messages,
      });
      return result.text;
    }

    if (rawTypeKey === "vertexai") {
      const config     = parseVertexConfig(apiKey, activeProvider.baseUrl, activeProvider.modelName);
      const vertexMsgs = messages.map((m) => ({ role: m.role, content: m.content }));
      return await callVertexAi(config, vertexMsgs, systemPrompt);
    }

    const provType = resolveProviderType(rawType, url);

    if (provType === "anthropic" || provType === "orbit" || provType === "agentrouter") {
      const customBase = provType !== "anthropic" ? activeProvider.baseUrl : null;
      return await callAnthropicCompatible(
        apiKey,
        activeProvider.modelName,
        systemPrompt,
        messages,
        customBase,
      );
    }

    return await callOpenAICompatible(
      apiKey,
      activeProvider.baseUrl ?? "https://api.openai.com",
      activeProvider.modelName,
      systemPrompt,
      messages,
      provType,
    );
  } catch (err: any) {
    console.error(`❌ callAI error [${activeProvider.providerType}/${activeProvider.modelName}]:`, err.message);
    throw err;
  }
}

// ── callSingleProvider: call one provider by its record ───────────────────────
async function callSingleProvider(
  provider: typeof aiProvidersTable.$inferSelect,
  messages: Message[],
  systemPrompt: string,
): Promise<string> {
  const apiKey = decrypt(provider.apiKey);
  if (!apiKey) {
    throw new Error("AI provider API key is not configured");
  }

  const rawType    = provider.providerType.toLowerCase();
  const rawTypeKey = rawType.replace(/\s+/g, "");
  const url        = (provider.baseUrl ?? "").toLowerCase();
  const apiFormat  = detectApiFormat(rawTypeKey);

  if (apiFormat === "raw_single" || apiFormat === "raw_messages") {
    const endpointUrl = provider.baseUrl ?? "";
    if (!endpointUrl) throw new Error("Raw API provider requires a full endpoint URL in Base URL field");
    const result = await callWithFormat(apiFormat, {
      apiKey,
      baseUrl: endpointUrl,
      model: provider.modelName,
      systemPrompt,
      messages,
    });
    return result.text;
  }

  if (rawTypeKey === "vertexai") {
    const config       = parseVertexConfig(apiKey, provider.baseUrl, provider.modelName);
    const vertexMsgs   = messages.map((m) => ({ role: m.role, content: m.content }));
    return await callVertexAi(config, vertexMsgs, systemPrompt);
  }

  const provType = resolveProviderType(rawType, url);

  if (provType === "anthropic" || provType === "orbit" || provType === "agentrouter") {
    const customBase = provType !== "anthropic" ? provider.baseUrl : null;
    return await callAnthropicCompatible(apiKey, provider.modelName, systemPrompt, messages, customBase);
  }

  return await callOpenAICompatible(
    apiKey,
    provider.baseUrl ?? "https://api.openai.com",
    provider.modelName,
    systemPrompt,
    messages,
    provType,
  );
}

// ── callAIWithLoadBalancing: load-balanced call, returns text only ─────────────
export async function callAIWithLoadBalancing(
  messages: Message[],
  systemPrompt: string
): Promise<string> {
  const enabledProviders = await db
    .select()
    .from(aiProvidersTable)
    .where(and(eq(aiProvidersTable.isEnabled, 1)))
    .orderBy(
      asc(aiProvidersTable.priority),
      asc(sql`COALESCE(${aiProvidersTable.lastUsedAt}, '1970-01-01T00:00:00.000Z')`)
    );

  if (enabledProviders.length === 0) {
    return callAI(messages, systemPrompt);
  }

  const errors: string[] = [];

  for (const provider of enabledProviders) {
    const start = Date.now();
    let lastErrMsg = "";

    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        if (attempt > 0) {
          console.warn(`[ai] Retrying provider "${provider.name}" after 429 (attempt ${attempt + 1})…`);
          await new Promise<void>(r => setTimeout(r, 2000));
        }
        const result = await callSingleProvider(provider, messages, systemPrompt);
        const latency = Date.now() - start;

        void db.update(aiProvidersTable)
          .set({ failCount: 0, lastUsedAt: new Date().toISOString() })
          .where(eq(aiProvidersTable.id, provider.id));

        void db.insert(providerUsageLogTable).values({
          providerId: provider.id, success: 1, latencyMs: latency, error: null,
          createdAt: new Date().toISOString(),
        });

        return result;
      } catch (err: unknown) {
        lastErrMsg = err instanceof Error ? err.message : String(err);
        const lowered = lastErrMsg.toLowerCase();
        const isRateLimit = lowered.includes("429")
          || lowered.includes("resource_exhausted")
          || lowered.includes("resource has been exhausted")
          || lowered.includes("quota exceeded")
          || lowered.includes("rate limit")
          || lowered.includes("too many requests");
        if (!isRateLimit || attempt >= 1) break;
      }
    }

    const latency = Date.now() - start;
    errors.push(`[${provider.name}] ${lastErrMsg}`);
    console.error(`⚠️ Provider "${provider.name}" failed (${latency}ms):`, lastErrMsg);

    void db.update(aiProvidersTable)
      .set({ failCount: sql`${aiProvidersTable.failCount} + 1`, lastUsedAt: new Date().toISOString() })
      .where(eq(aiProvidersTable.id, provider.id));

    void db.insert(providerUsageLogTable).values({
      providerId: provider.id, success: 0, latencyMs: latency,
      error: lastErrMsg.substring(0, 500), createdAt: new Date().toISOString(),
    });
  }

  throw new Error(`All ${enabledProviders.length} providers failed: ${errors.join(" | ")}`);
}

// ── callAnthropicCompatible: Anthropic / Orbit / AgentRouter format ───────────
async function callAnthropicCompatible(
  apiKey: string,
  model: string,
  systemPrompt: string,
  messages: Message[],
  customBaseUrl?: string | null,
): Promise<string> {
  const baseUrl = customBaseUrl ? customBaseUrl.replace(/\/$/, "") : "https://api.anthropic.com";
  const fullUrl = `${baseUrl}/v1/messages`;
  const response = await fetch(fullUrl, {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      max_tokens: 700,
      system: systemPrompt,
      messages,
    }),
  });

  const rawText = await response.text();
  if (rawText.trim().startsWith("<")) {
    throw new Error(`API returned HTML instead of JSON (${fullUrl}). Check API key and base URL.`);
  }
  const data = JSON.parse(rawText) as {
    content?: Array<{ type: string; text: string }>;
    error?: { message: string };
  };
  if (data.error) {
    throw new Error(`Anthropic API error: ${data.error.message}`);
  }
  if (!response.ok) {
    throw new Error(`Anthropic API error: ${response.status}`);
  }
  return data.content?.[0]?.text ?? "";
}

// ── callOpenAICompatible: OpenAI / DeepSeek / Gemini / Groq / OpenRouter ──────
async function callOpenAICompatible(
  apiKey: string,
  baseUrl: string,
  model: string,
  systemPrompt: string,
  messages: Message[],
  providerType: string,
): Promise<string> {
  const cleanBase = baseUrl.replace(/\/$/, "");
  const skipV1 = providerType === "deepseek" || providerType === "gemini";
  const endpoint = skipV1 ? "/chat/completions" : "/v1/chat/completions";

  const extraHeaders: Record<string, string> = {};
  if (providerType === "openrouter") {
    extraHeaders["HTTP-Referer"] = process.env["APP_URL"]?.replace(/\/$/, "")
      ?? (process.env["REPLIT_DEV_DOMAIN"] ? `https://${process.env["REPLIT_DEV_DOMAIN"]}` : "");
    extraHeaders["X-Title"] = "Facebook AI Agent";
  }

  const fullUrl = `${cleanBase}${endpoint}`;
  const response = await fetch(fullUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      ...extraHeaders,
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "system", content: systemPrompt }, ...messages],
      max_tokens: 700,
    }),
  });

  const rawText = await response.text();
  if (rawText.trim().startsWith("<")) {
    throw new Error(`${providerType} API returned HTML instead of JSON (${fullUrl}). Check API key and base URL.`);
  }

  let data: {
    choices?: Array<{ message?: { content?: string } }>;
    error?: { message?: string; type?: string } | string;
  };
  try {
    data = JSON.parse(rawText);
  } catch {
    throw new Error(`${providerType} returned invalid JSON: ${rawText.substring(0, 200)}`);
  }

  if (data.error) {
    const errObj = data.error;
    const errMsg = typeof errObj === "string" ? errObj : errObj.message ?? JSON.stringify(errObj);
    const statusCode = response.status;
    throw new Error(`${providerType} API error ${statusCode}: ${errMsg}`);
  }
  if (!response.ok) {
    throw new Error(`${providerType} API error ${response.status}: ${rawText.substring(0, 300)}`);
  }
  return data.choices?.[0]?.message?.content ?? "";
}
