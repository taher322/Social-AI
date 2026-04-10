// ── AI Multimodal, Classifiers & Product Utilities ───────────────────────────
// Handles image/audio/video analysis, shopping intent classification,
// product summarization, and appointment slot fetching.
// Extracted from ai.ts for clarity; re-exported via ai.ts for backward compat.

import {
  db,
  aiProvidersTable,
  productsTable,
  availableSlotsTable,
} from "@workspace/db";
import { eq, and, desc, asc } from "drizzle-orm";
import { cache } from "./cache.js";
import { decrypt } from "./encryption.js";
import { detectApiFormat, callWithFormat, resolveProviderType } from "./apiTransformer.js";
import { callVertexAi, callVertexAiMultimodal, parseVertexConfig } from "./vertexAi.js";
import { callAIWithMetadata } from "./aiEngine.js";
import { getProviderCapabilities } from "./providerCapabilities.js";
import { callMultimodal, type MultimodalContext } from "./callMultimodal.js";

type Product = typeof productsTable.$inferSelect;

// ── ShoppingContext: multi-step shopping intent state ─────────────────────────
export interface ShoppingContext {
  step: "show_categories" | "show_filter_options" | "show_price_tiers" | "show_products" | "answer_question";
  activeCategory: string | null;
  filterType: "by_type" | "by_price" | null;
  priceTier: "budget" | "mid" | "premium" | null;
  keywords: string[];
  contextAction: "KEEP" | "UPDATE" | "DROP";
}

// ── CategoryClassification: legacy single-step classifier ────────────────────
export interface CategoryClassification {
  category: string | null;
  keyword: string | null;
  changed: boolean;
}

// ── classifyShoppingIntent: multi-step AI-powered shopping state machine ──────
export async function classifyShoppingIntent(
  messageText: string,
  currentContext: ShoppingContext | null,
  availableCategories: string[],
  availableBrandsOrTypes: string[],
  priceTiersDescription: string,
  recentMessages: string
): Promise<ShoppingContext> {
  if (availableCategories.length === 0) {
    return { step: "answer_question", activeCategory: null, filterType: null, priceTier: null, keywords: [], contextAction: "DROP" };
  }

  const contextDesc = currentContext
    ? `Current state: step="${currentContext.step}", category="${currentContext.activeCategory ?? "none"}", filterType="${currentContext.filterType ?? "none"}", priceTier="${currentContext.priceTier ?? "none"}", keywords="${currentContext.keywords.length > 0 ? currentContext.keywords.join(", ") : "none"}"`
    : "Current state: No previous context (first message in session).";

  const brandsDesc = availableBrandsOrTypes.length > 0
    ? `Available brands/types in current category: ${availableBrandsOrTypes.join(", ")}`
    : "";

  const recentBlock = recentMessages
    ? `\nRecent conversation (for context):\n${recentMessages}`
    : "";

  const classifySystemPrompt = `You are a shopping assistant state machine for a store chatbot.
Available product categories: ${availableCategories.join(", ")}
${brandsDesc}
${priceTiersDescription}
${contextDesc}
${recentBlock}

Based on the customer message, determine the next shopping step and respond ONLY with valid JSON (no markdown, no extra text):
{"step":"<value>","activeCategory":"<value or null>","filterType":"<value or null>","priceTier":"<value or null>","keywords":["<keyword1>","<keyword2>"],"contextAction":"<KEEP|UPDATE|DROP>"}

Step rules:
- "show_categories": customer asks general questions ("what do you have?" / "ماهي منتجاتكم" / "واش عندكم" / "عرضلي كل شي")
- "show_filter_options": customer just selected a category and no filter type chosen yet — ask by_type or by_price
- "show_price_tiers": customer chose by_price filter — show budget/mid/premium tiers
- "show_products": customer picked a specific brand/type/price tier OR mentioned a specific product keyword — show product cards
- "answer_question": customer asks a question not related to browsing (greeting, order status, complaint, etc.)

contextAction rules (decide what to do with the stored shopping context):
- "KEEP": message is still related to the current context (e.g. "كم الضمان؟" while browsing Samsung → KEEP Samsung context)
- "UPDATE": customer explicitly switched category or product (e.g. was browsing phones, now asks about clothes → UPDATE to new category)
- "DROP": customer completely changed topic unrelated to any shopping (greeting after long browsing, complaint, general question about store hours, sending an image of a different product class) → clears activeCategory so next message starts fresh

Mid-flow change rules:
- If customer was browsing "هواتف" but says "أريد ملابس" → step=show_filter_options, activeCategory="ملابس", contextAction=UPDATE
- If customer was browsing by_type but says "أريد حسب السعر" → filterType=by_price, step=show_price_tiers, keep activeCategory, contextAction=KEEP
- If customer was at any step but asks a general unrelated question → step=answer_question, contextAction=DROP
- If customer asks something about the current product (warranty, color, specs) → step=answer_question, contextAction=KEEP

keywords rules (CRITICAL — always apply for step=show_products):
- Extract 1 to 3 search terms that best describe what the customer is looking for
- Include the direct keyword AND synonyms/related terms (e.g. "مجفف شعر" → ["مجفف", "تجفيف", "تصفيف"])
- Include the functional description if the customer described use (e.g. "شيء يصفف الشعر" → ["تصفيف", "مجفف", "سشوار"])
- Correct spelling mistakes (e.g. "ايفن" → "ايفون")
- For Arabic, include both root forms when helpful (e.g. ["تصفيف", "مصفف", "سشوار"])
- Return [] (empty array) if step is not show_products or if no specific product was mentioned

filterType values: "by_type" | "by_price" | null
priceTier values: "budget" | "mid" | "premium" | null
contextAction values: "KEEP" | "UPDATE" | "DROP"
keywords: array of 1–3 strings (empty array if not applicable)`;

  try {
    const result = await callAIWithMetadata(
      [{ role: "user", content: messageText }],
      classifySystemPrompt
    );
    const raw = result.text.trim();
    const jsonMatch = raw.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return { step: "answer_question", activeCategory: currentContext?.activeCategory ?? null, filterType: currentContext?.filterType ?? null, priceTier: null, keywords: [], contextAction: "KEEP" };
    }
    const parsed = JSON.parse(jsonMatch[0]);
    const contextAction: ShoppingContext["contextAction"] =
      parsed.contextAction === "DROP" ? "DROP"
      : parsed.contextAction === "UPDATE" ? "UPDATE"
      : "KEEP";

    let parsedKeywords: string[] = [];
    if (Array.isArray(parsed.keywords)) {
      parsedKeywords = parsed.keywords.filter((k: unknown) => typeof k === "string" && k.trim().length > 0);
    } else if (typeof parsed.keyword === "string" && parsed.keyword.trim()) {
      parsedKeywords = [parsed.keyword.trim()];
    }

    return {
      step: parsed.step ?? "answer_question",
      activeCategory: parsed.activeCategory ?? null,
      filterType: parsed.filterType ?? null,
      priceTier: parsed.priceTier ?? null,
      keywords: parsedKeywords,
      contextAction,
    };
  } catch {
    return { step: "answer_question", activeCategory: currentContext?.activeCategory ?? null, filterType: currentContext?.filterType ?? null, priceTier: null, keywords: [], contextAction: "KEEP" };
  }
}

// ── classifyProductCategory: legacy single-step category classifier ───────────
export async function classifyProductCategory(
  messageText: string,
  availableCategories: string[],
  previousCategory: string | null
): Promise<CategoryClassification> {
  if (availableCategories.length === 0) {
    return { category: "all", keyword: null, changed: false };
  }

  const categoryList = availableCategories.join(", ");
  const prevCtx = previousCategory
    ? `Current active category: "${previousCategory}".`
    : "No previous category.";

  const classifySystemPrompt = `You are a product category classifier for a store chatbot.
Available product categories: ${categoryList}
${prevCtx}

Analyze the customer message and respond with ONLY a valid JSON object — no markdown, no extra text:
{"category": "<category name | all | none>", "keyword": "<specific product keyword or null>", "changed": <true|false>}

Rules:
- category: one of the available categories (correcting spelling/dialect), "all" (general catalog question like "what do you have?"), or "none" (not product-related)
- keyword: the specific product the customer mentioned with spelling corrected (e.g. "ايفون" for "ايفن"), or null
- changed: true if the new category differs from the previous active category`;

  try {
    const result = await callAIWithMetadata(
      [{ role: "user", content: messageText }],
      classifySystemPrompt
    );
    const raw = result.text.trim();
    const jsonMatch = raw.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return { category: "all", keyword: null, changed: true };
    const parsed = JSON.parse(jsonMatch[0]);
    return {
      category: parsed.category ?? "all",
      keyword: parsed.keyword ?? null,
      changed: Boolean(parsed.changed),
    };
  } catch {
    return { category: "all", keyword: null, changed: false };
  }
}

// ── getFreshAppointmentBlock: fresh DB query for today's available slots ──────
export async function getFreshAppointmentBlock(): Promise<string> {
  const today = new Date();
  const dayOfWeek = today.getDay();
  const todayStr = today.toISOString().split("T")[0];

  const todaySlots = await db
    .select()
    .from(availableSlotsTable)
    .where(and(eq(availableSlotsTable.dayOfWeek, dayOfWeek), eq(availableSlotsTable.isActive, 1)));

  if (todaySlots.length === 0) return "";

  return `\nAPPOINTMENT BOOKING:
Available time slots for today (${todayStr}): ${todaySlots.map((s) => s.timeSlot).join(", ")}
If the customer wants to book an appointment, respond ONLY with this exact JSON (no other text):
{"action":"create_appointment","service_name":"SERVICE_DESCRIPTION","appointment_date":"${todayStr}","time_slot":"HH:MM","note":"any note from customer"}
Always confirm the time slot is in the available list before creating an appointment.\n`;
}

// ── GeminiMultimodalAnalysis: result type from image/audio/video analysis ─────
export interface GeminiMultimodalAnalysis {
  normalizedText: string;
  detectedProductType: string | null;
  detectedCategory: string | null;
  detectedBrand: string | null;
  extractedKeywords: string[];
  userIntent: "product_inquiry" | "price_inquiry" | "question" | "general" | "unclear";
  confidence: number;
}

// ── getGeminiCredentials: find enabled Gemini provider credentials ─────────────
async function getGeminiCredentials(): Promise<{ key: string; model: string } | null> {
  const cacheKey = "gemini:creds";
  const cached = cache.get<{ key: string; model: string } | null>(cacheKey);
  if (cached !== undefined) return cached;

  const providers = await db
    .select()
    .from(aiProvidersTable)
    .where(eq(aiProvidersTable.isEnabled, 1));
  for (const p of providers) {
    const resolved = resolveProviderType(
      p.providerType.toLowerCase(),
      (p.baseUrl ?? "").toLowerCase()
    );
    if (resolved === "gemini") {
      const key = decrypt(p.apiKey);
      if (key) {
        const result = { key, model: p.modelName };
        cache.set(cacheKey, result, 60 * 1000);
        return result;
      }
    }
  }
  cache.set(cacheKey, null, 60 * 1000);
  return null;
}

// ── analyzeImageWithActiveProvider: fallback multimodal using active provider ──
async function analyzeImageWithActiveProvider(
  mediaBase64: string,
  mimeType: string,
  prompt: string,
): Promise<GeminiMultimodalAnalysis | null> {
  const [activeProvider] = await db
    .select()
    .from(aiProvidersTable)
    .where(eq(aiProvidersTable.isActive, 1))
    .limit(1);
  if (!activeProvider) return null;

  const apiKey = decrypt(activeProvider.apiKey);
  if (!apiKey) return null;

  const rawType    = activeProvider.providerType.toLowerCase();
  const rawTypeKey = rawType.replace(/\s+/g, "");
  const url        = (activeProvider.baseUrl ?? "").toLowerCase();
  const provType   = resolveProviderType(rawType, url);

  let responseText: string | null = null;
  try {
    if (rawTypeKey === "vertexai") {
      const config    = parseVertexConfig(apiKey, activeProvider.baseUrl, activeProvider.modelName);
      const timeoutMs = mimeType.startsWith("audio/") || mimeType.startsWith("video/") ? 25000 : 15000;
      try {
        responseText = await callVertexAiMultimodal(config, prompt, mediaBase64, mimeType, timeoutMs);
      } catch (vErr) {
        console.error("[multimodal] Vertex AI multimodal failed:", (vErr as Error).message);
        return null;
      }
    } else if (provType === "anthropic" || provType === "orbit" || provType === "agentrouter") {
      const base = (provType !== "anthropic" && activeProvider.baseUrl)
        ? activeProvider.baseUrl.replace(/\/$/, "")
        : "https://api.anthropic.com";
      const resp = await fetch(`${base}/v1/messages`, {
        method: "POST",
        headers: { "x-api-key": apiKey, "anthropic-version": "2023-06-01", "content-type": "application/json" },
        body: JSON.stringify({
          model: activeProvider.modelName, max_tokens: 512,
          messages: [{ role: "user", content: [
            { type: "image", source: { type: "base64", media_type: mimeType, data: mediaBase64 } },
            { type: "text", text: prompt },
          ]}],
        }),
        signal: AbortSignal.timeout(12000),
      });
      const data = await resp.json() as { content?: Array<{ text?: string }>; error?: { message: string } };
      if (data.error) return null;
      responseText = data.content?.[0]?.text ?? null;
    } else {
      const cleanBase = (activeProvider.baseUrl ?? "https://api.openai.com").replace(/\/$/, "");
      const skipV1 = provType === "deepseek" || provType === "gemini";
      const endpoint = skipV1 ? "/chat/completions" : "/v1/chat/completions";
      const resp = await fetch(`${cleanBase}${endpoint}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          model: activeProvider.modelName, max_tokens: 512,
          messages: [{ role: "user", content: [
            { type: "text", text: prompt },
            { type: "image_url", image_url: { url: `data:${mimeType};base64,${mediaBase64}` } },
          ]}],
        }),
        signal: AbortSignal.timeout(12000),
      });
      const rawResp = await resp.text();
      if (rawResp.trim().startsWith("<")) return null;
      const data = JSON.parse(rawResp) as { choices?: Array<{ message?: { content?: string } }>; error?: unknown };
      if (data.error) return null;
      responseText = data.choices?.[0]?.message?.content ?? null;
    }
  } catch { return null; }

  if (!responseText) return null;
  const jsonMatch = responseText.match(/\{[\s\S]*\}/);
  if (!jsonMatch) return null;
  try {
    const analysis = JSON.parse(jsonMatch[0]) as GeminiMultimodalAnalysis;
    analysis.confidence = Math.min(1, Math.max(0, Number(analysis.confidence) || 0));
    analysis.extractedKeywords = Array.isArray(analysis.extractedKeywords) ? analysis.extractedKeywords : [];
    analysis.normalizedText = String(analysis.normalizedText ?? "");
    console.log(`[multimodal] active-provider image analysis: intent=${analysis.userIntent} confidence=${analysis.confidence}`);
    return analysis;
  } catch { return null; }
}

// ── analyzeAttachmentWithGemini: analyze image/audio/video for product intent ─
export async function analyzeAttachmentWithGemini(
  attachmentUrl: string,
  attachmentType: "image" | "audio" | "video",
  userText?: string,
  pageAccessToken?: string
): Promise<GeminiMultimodalAnalysis | null> {
  // Step 1: Fetch media
  let mediaBase64: string;
  let mimeType: string;
  try {
    const fetchUrl = pageAccessToken
      ? `${attachmentUrl}${attachmentUrl.includes("?") ? "&" : "?"}access_token=${pageAccessToken}`
      : attachmentUrl;
    const resp = await fetch(fetchUrl, { signal: AbortSignal.timeout(15000) });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const buffer = await resp.arrayBuffer();
    if (buffer.byteLength > 15 * 1024 * 1024) {
      console.warn("[multimodal] Attachment too large (>15MB) — skipping");
      return null;
    }
    mediaBase64 = Buffer.from(buffer).toString("base64");
    const ct = resp.headers.get("content-type") ?? "";
    if (attachmentType === "image") {
      mimeType = ct.startsWith("image/") ? ct.split(";")[0]!.trim() : "image/jpeg";
    } else if (attachmentType === "video") {
      const urlL = attachmentUrl.toLowerCase();
      if (urlL.includes(".mp4")) mimeType = "video/mp4";
      else if (urlL.includes(".webm")) mimeType = "video/webm";
      else if (urlL.includes(".mov")) mimeType = "video/quicktime";
      else mimeType = ct.startsWith("video/") ? ct.split(";")[0]!.trim() : "video/mp4";
    } else {
      const urlL = attachmentUrl.toLowerCase();
      if (urlL.includes(".m4a")) mimeType = "audio/m4a";
      else if (urlL.includes(".mp3")) mimeType = "audio/mp3";
      else if (urlL.includes(".wav")) mimeType = "audio/wav";
      else mimeType = ct.startsWith("audio/") ? ct.split(";")[0]!.trim() : "audio/ogg";
    }
  } catch (err) {
    console.error("[multimodal] Failed to fetch attachment:", (err as Error).message);
    return null;
  }

  // Step 2: Build analysis prompt
  const jsonSchema = `{
  "normalizedText": "what the user likely wants (string)",
  "detectedProductType": "specific product type or null",
  "detectedCategory": "one of: phones, electronics, fashion, food, beauty, auto, auto_parts, courses, services, restaurant, general — or null",
  "detectedBrand": "brand name or null",
  "extractedKeywords": ["keyword1", "keyword2"],
  "userIntent": "product_inquiry | price_inquiry | question | general | unclear",
  "confidence": 0.0
}`;

  const prompt = attachmentType === "image"
    ? `You are a product recognition assistant for an e-commerce chatbot. Analyze this image and identify any product shown.
Return ONLY valid JSON with no markdown, no extra text:
${jsonSchema}${userText ? `\nThe user also typed: "${userText}"` : ""}`
    : attachmentType === "video"
    ? `You are a product recognition assistant for an e-commerce chatbot. Analyze this video and identify any product shown.
Return ONLY valid JSON with no markdown, no extra text:
${jsonSchema}`
    : `You are a customer service assistant. Transcribe this audio message and identify what product or service the customer wants.
Return ONLY valid JSON with no markdown, no extra text:
${jsonSchema}`;

  // Step 3: Try Gemini first; fallback to active provider
  const gemini = await getGeminiCredentials();
  if (!gemini) {
    console.warn(`[multimodal] No Gemini provider — trying active provider for ${attachmentType} analysis`);
    return analyzeImageWithActiveProvider(mediaBase64, mimeType, prompt);
  }

  const timeoutMs = attachmentType === "image" ? 15000 : 25000;
  const visionModel = attachmentType === "image"
    ? gemini.model
    : gemini.model.includes("lite") ? "gemini-2.0-flash" : gemini.model;

  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${visionModel}:generateContent?key=${gemini.key}`;

  try {
    const body = {
      contents: [
        {
          parts: [
            { text: prompt },
            { inline_data: { mime_type: mimeType, data: mediaBase64 } },
          ],
        },
      ],
      generationConfig: { temperature: 0.1, maxOutputTokens: 512 },
    };

    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });

    const raw = await resp.text();
    if (!resp.ok) {
      console.error("[multimodal] Gemini API error:", raw.substring(0, 300));
      console.warn(`[multimodal] Gemini failed for ${attachmentType} — trying active provider`);
      return analyzeImageWithActiveProvider(mediaBase64, mimeType, prompt);
    }

    const data = JSON.parse(raw) as {
      candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
    };

    const responseText = data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
    const jsonMatch = responseText.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      console.error("[multimodal] No JSON in Gemini response:", responseText.substring(0, 200));
      return null;
    }

    const analysis = JSON.parse(jsonMatch[0]) as GeminiMultimodalAnalysis;
    analysis.confidence = Math.min(1, Math.max(0, Number(analysis.confidence) || 0));
    analysis.extractedKeywords = Array.isArray(analysis.extractedKeywords)
      ? analysis.extractedKeywords
      : [];
    analysis.normalizedText = String(analysis.normalizedText ?? "");

    console.log(
      `[multimodal] ${attachmentType} analysis: intent=${analysis.userIntent} confidence=${analysis.confidence} category=${analysis.detectedCategory} brand=${analysis.detectedBrand}`
    );
    return analysis;
  } catch (err) {
    console.error("[multimodal] Gemini call failed:", (err as Error).message);
    return null;
  }
}

// ── matchProductsFromAnalysis: score products against multimodal analysis ──────
export function matchProductsFromAnalysis(
  analysis: GeminiMultimodalAnalysis,
  products: Product[]
): { matches: Product[]; tier: "strong" | "multiple" | "none" } {
  if (analysis.confidence < 0.3 || products.length === 0) {
    return { matches: [], tier: "none" };
  }

  const scored: Array<{ product: Product; score: number }> = [];

  for (const p of products) {
    if (p.status !== "available") continue;
    let score = 0;
    const nameL     = (p.name ?? "").toLowerCase();
    const descL     = (p.description ?? "").toLowerCase();
    const catL      = (p.category ?? "").toLowerCase();
    const brandL    = (p.brand ?? "").toLowerCase();
    const itemTypeL = (p.itemType ?? "").toLowerCase();

    if (analysis.detectedCategory) {
      const catDet = analysis.detectedCategory.toLowerCase();
      if (catL === catDet) score += 4;
      else if (catL.includes(catDet) || catDet.includes(catL)) score += 2;
    }
    if (analysis.detectedBrand) {
      const brandDet = analysis.detectedBrand.toLowerCase();
      if (brandL === brandDet) score += 4;
      else if (brandL.includes(brandDet) || brandDet.includes(brandL)) score += 2;
    }
    if (analysis.detectedProductType) {
      const typeL = analysis.detectedProductType.toLowerCase();
      if (itemTypeL.includes(typeL) || typeL.includes(itemTypeL)) score += 2;
      else if (nameL.includes(typeL)) score += 1;
    }
    for (const kw of analysis.extractedKeywords) {
      const kwL = kw.toLowerCase().trim();
      if (!kwL) continue;
      if (nameL.includes(kwL)) score += 2;
      else if (descL.includes(kwL) || catL.includes(kwL) || brandL.includes(kwL)) score += 1;
    }

    if (score > 0) scored.push({ product: p, score });
  }

  scored.sort((a, b) => b.score - a.score);
  const top4 = scored.slice(0, 4);
  if (top4.length === 0) return { matches: [], tier: "none" };

  const topScore = top4[0]!.score;
  if (topScore < 3) return { matches: [], tier: "none" };

  const isStrong =
    top4.length === 1 ||
    (top4.length >= 2 && topScore >= top4[1]!.score * 2 && topScore >= 5);

  if (isStrong) return { matches: [top4[0]!.product], tier: "strong" };
  return { matches: top4.map((t) => t.product), tier: "multiple" };
}

// ── buildPriorityList: ordered list of MultimodalContexts to try ──────────────
// Mirrors callAIWithMetadata: queries ALL enabled providers (isEnabled=1),
// orders active provider first (isActive=1 → DESC), then by priority ASC.
// Skips providers that don't support the requested attachment type.
async function buildPriorityList(
  attachmentType: "image" | "audio" | "video",
): Promise<MultimodalContext[]> {
  const contexts: MultimodalContext[] = [];

  // Same query style as callAIWithMetadata — isEnabled=1, active first
  const enabledProviders = await db
    .select().from(aiProvidersTable)
    .where(eq(aiProvidersTable.isEnabled, 1))
    .orderBy(desc(aiProvidersTable.isActive), asc(aiProvidersTable.priority));

  console.log(`[buildPriorityList] attachmentType=${attachmentType} enabledProviders=${enabledProviders.length}`);

  for (const provider of enabledProviders) {
    const apiKey = decrypt(provider.apiKey);
    if (!apiKey) continue;

    const rawTypeKey   = provider.providerType.toLowerCase().replace(/\s+/g, "");
    const pUrl         = (provider.baseUrl ?? "").toLowerCase();
    const resolvedType = resolveProviderType(provider.providerType.toLowerCase(), pUrl);
    const caps         = getProviderCapabilities(
      rawTypeKey === "vertexai" ? "vertexai" : resolvedType,
    );

    const needsAudio   = attachmentType === "audio";
    const supportsThis = needsAudio ? caps.supportsAudio : caps.supportsImage;
    const format       = needsAudio ? caps.audioFormat   : caps.imageFormat;

    console.log(`[buildPriorityList] ${provider.name} isActive=${provider.isActive} rawTypeKey=${rawTypeKey} supportsThis=${supportsThis} format=${format}`);

    if (!supportsThis || !format) continue;

    if (format === "vertex") {
      const vertexConfig = parseVertexConfig(apiKey, provider.baseUrl, provider.modelName);
      contexts.push({ endpoint: "", apiKey, model: provider.modelName, format: "vertex", vertexConfig });

    } else if (format === "gemini-inline") {
      const model = attachmentType === "image"
        ? provider.modelName
        : provider.modelName.includes("lite") ? "gemini-2.0-flash" : provider.modelName;
      const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
      contexts.push({ endpoint, apiKey, model, format: "gemini-inline" });

    } else if (format === "whisper") {
      const cleanBase = (provider.baseUrl ?? "https://api.openai.com").replace(/\/$/, "");
      contexts.push({
        endpoint: `${cleanBase}/v1/audio/transcriptions`,
        apiKey,
        model: provider.modelName,
        format: "whisper",
        whisperModel: caps.whisperModel ?? "whisper-1",
      });

    } else if (format === "anthropic") {
      const base = provider.baseUrl
        ? provider.baseUrl.replace(/\/$/, "")
        : "https://api.anthropic.com";
      contexts.push({ endpoint: `${base}/v1/messages`, apiKey, model: provider.modelName, format: "anthropic" });

    } else if (format === "openai-vision") {
      const cleanBase = (provider.baseUrl ?? "https://api.openai.com").replace(/\/$/, "");
      const skipV1    = resolvedType === "deepseek" || resolvedType === "gemini";
      const ep        = skipV1 ? "/chat/completions" : "/v1/chat/completions";
      contexts.push({ endpoint: `${cleanBase}${ep}`, apiKey, model: provider.modelName, format: "openai-vision" });
    }
  }

  console.log(`[buildPriorityList] built ${contexts.length} contexts: ${contexts.map((c) => c.format).join(", ")}`);
  return contexts;
}

// ── transcribeOrDescribeAttachment: plain-text description of media for AI ────
export async function transcribeOrDescribeAttachment(
  attachmentUrl: string,
  attachmentType: "image" | "audio" | "video",
  pageAccessToken?: string,
): Promise<string | null> {
  // Step 1: Fetch media → keep raw ArrayBuffer (conversion happens in callMultimodal)
  let buffer: ArrayBuffer;
  let mimeType: string;
  try {
    const fetchUrl = pageAccessToken
      ? `${attachmentUrl}${attachmentUrl.includes("?") ? "&" : "?"}access_token=${pageAccessToken}`
      : attachmentUrl;
    const resp = await fetch(fetchUrl, { signal: AbortSignal.timeout(15_000) });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    buffer = await resp.arrayBuffer();
    if (buffer.byteLength > 15 * 1024 * 1024) {
      console.warn("[transcribe] Attachment too large (>15MB) — skipping");
      return null;
    }
    const ct = resp.headers.get("content-type") ?? "";
    if (attachmentType === "image") {
      mimeType = ct.startsWith("image/") ? ct.split(";")[0]!.trim() : "image/jpeg";
    } else if (attachmentType === "video") {
      const urlL = attachmentUrl.toLowerCase();
      if (urlL.includes(".mp4"))       mimeType = "video/mp4";
      else if (urlL.includes(".webm")) mimeType = "video/webm";
      else if (urlL.includes(".mov"))  mimeType = "video/quicktime";
      else mimeType = ct.startsWith("video/") ? ct.split(";")[0]!.trim() : "video/mp4";
    } else {
      const urlL = attachmentUrl.toLowerCase();
      if (urlL.includes(".m4a"))       mimeType = "audio/m4a";
      else if (urlL.includes(".mp3"))  mimeType = "audio/mp3";
      else if (urlL.includes(".wav"))  mimeType = "audio/wav";
      else mimeType = ct.startsWith("audio/") ? ct.split(";")[0]!.trim() : "audio/ogg";
    }
  } catch (err) {
    console.error("[transcribe] Failed to fetch attachment:", (err as Error).message);
    return null;
  }

  // Step 2: Build prompt
  const prompt =
    attachmentType === "audio"
      ? "Transcribe this audio message exactly as spoken. The customer may be speaking Arabic or another language. Return ONLY the transcribed text, nothing else — no JSON, no labels, no explanation."
      : attachmentType === "image"
      ? "Describe this image in Arabic in one or two concise sentences, focusing on what the customer is likely showing or asking about. Return ONLY the description text, nothing else."
      : "Briefly describe what is shown in this video in Arabic. Return ONLY the description text, nothing else.";

  const timeoutMs = attachmentType === "image" ? 15_000 : 25_000;

  // Step 3: Try each context in priority order — stop on first success
  const contexts = await buildPriorityList(attachmentType);
  console.log(`[transcribe] contexts built: ${contexts.length} → ${contexts.map((c) => c.format).join(", ")}`);

  for (const ctx of contexts) {
    const text = await callMultimodal(ctx, buffer, mimeType, prompt, timeoutMs);
    if (text) {
      console.log(`[transcribe] ${ctx.format} ${attachmentType} → "${text.substring(0, 80)}"`);
      return text;
    }
    console.warn(`[transcribe] ${ctx.format} failed for ${attachmentType} — trying next`);
  }

  console.warn(`[transcribe] All providers failed for ${attachmentType}`);
  return null;
}

// ── summarizeProductForUser: AI-powered product description summarizer ─────────
export async function summarizeProductForUser(product: {
  name: string;
  description: string;
  category?: string | null;
  brand?: string | null;
  itemType?: string | null;
}): Promise<string | null> {
  if (!product.description || product.description.trim().length < 30) return null;

  const systemPrompt = [
    "أنت مساعد مبيعات محترف. مهمتك تقديم ملخص مقنع وواضح لوصف المنتج للعميل.",
    "القواعد الصارمة:",
    "- اكتب باللغة العربية فقط",
    "- لخّص الوصف في 2-4 جمل طبيعية ومقنعة",
    "- أبرز النقاط الأساسية والمميزات الرئيسية التي تهم العميل",
    "- استخدم أسلوباً ودياً ومناسباً للبيع دون مبالغة",
    "- لا تذكر السعر أو المخزون إطلاقاً (سيُضافان بشكل منفصل)",
    "- لا تستخدم JSON أو HTML أو أي تنسيق برمجي",
    "- لا تبدأ بـ 'بالتأكيد' أو 'إليك' أو 'يسعدني' أو ما شابه",
    "- أجب بالنص مباشرة",
  ].join("\n");

  const context = [
    `المنتج: ${product.name}`,
    product.category ? `الفئة: ${product.category}` : "",
    product.brand    ? `العلامة التجارية: ${product.brand}` : "",
    product.itemType ? `النوع: ${product.itemType}` : "",
    `الوصف: ${product.description}`,
  ].filter(Boolean).join("\n");

  try {
    const result = await callAIWithMetadata(
      [{ role: "user", content: `لخّص وصف هذا المنتج للعميل:\n\n${context}` }],
      systemPrompt
    );
    const text = result.text.trim();
    if (!text || text.length < 10) return null;
    if (text.startsWith("{") || text.startsWith("[")) return null;
    return text;
  } catch {
    return null;
  }
}
