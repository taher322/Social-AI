// ── Facebook Graph API Helpers ────────────────────────────────────────────────
// All Facebook Messenger/Graph API send functions and user-info helpers.
// Extracted from ai.ts for clarity; re-exported via ai.ts for backward compat.

import sharp from "sharp";
import { TTL } from "./cache.js";
import { rGet, rSet } from "./redisCache.js";

export async function sendFbMessage(
  pageAccessToken: string,
  recipientId: string,
  message: string,
  pageId?: string
): Promise<void> {
  const endpoint = pageId ? `${pageId}/messages` : "me/messages";
  const resp = await fetch(
    `https://graph.facebook.com/v25.0/${endpoint}?access_token=${pageAccessToken}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        recipient: { id: recipientId },
        messaging_type: "RESPONSE",
        message: { text: message },
      }),
    }
  );
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(`FB API error ${resp.status}: ${body}`);
  }
}

export async function sendFbImageMessage(
  pageAccessToken: string,
  recipientId: string,
  imageUrl: string,
  pageId?: string
): Promise<void> {
  const endpoint = pageId ? `${pageId}/messages` : "me/messages";
  const resp = await fetch(
    `https://graph.facebook.com/v25.0/${endpoint}?access_token=${pageAccessToken}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        recipient: { id: recipientId },
        messaging_type: "RESPONSE",
        message: {
          attachment: {
            type: "image",
            payload: { url: imageUrl, is_reusable: true },
          },
        },
      }),
    }
  );
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(`FB API error ${resp.status}: ${body}`);
  }
}

// ── Direct multipart image upload (no external URL needed) ───────────────────
// Sends a product image by uploading its binary data directly to Facebook,
// avoiding the dependency on Facebook fetching our server URL (which can timeout).
export async function sendFbImageFromDataUrl(
  pageAccessToken: string,
  recipientId: string,
  dataUrl: string,
  pageId?: string
): Promise<void> {
  const endpoint = pageId ? `${pageId}/messages` : "me/messages";

  if (dataUrl.startsWith("data:")) {
    // Base64 data URL → decode and upload via multipart
    const [meta, b64] = dataUrl.split(",") as [string, string];
    const mimeMatch = meta.match(/data:([^;]+)/);
    let mime = mimeMatch?.[1] ?? "image/jpeg";
    let buf  = Buffer.from(b64, "base64");

    // Facebook Messenger لا يدعم WebP/AVIF — نحوّله إلى JPEG قبل الإرسال
    if (mime === "image/webp" || mime === "image/avif") {
      buf  = await sharp(buf).jpeg({ quality: 85 }).toBuffer() as Buffer<ArrayBuffer>;
      mime = "image/jpeg";
    }

    const ext = mime.split("/")[1] ?? "jpg";

    const form = new FormData();
    form.append("recipient",       JSON.stringify({ id: recipientId }));
    form.append("messaging_type",  "RESPONSE");
    form.append("message",         JSON.stringify({
      attachment: { type: "image", payload: { is_reusable: true } },
    }));
    form.append("filedata", new Blob([buf], { type: mime }), `product.${ext}`);

    const resp = await fetch(
      `https://graph.facebook.com/v25.0/${endpoint}?access_token=${pageAccessToken}`,
      { method: "POST", body: form }
    );
    if (!resp.ok) {
      const body = await resp.text().catch(() => "");
      throw new Error(`FB API error ${resp.status}: ${body}`);
    }
  } else {
    // External URL — fall back to URL-based method
    await sendFbImageMessage(pageAccessToken, recipientId, dataUrl, pageId);
  }
}

// ── Upload data URL image to Facebook CDN ────────────────────────────────────
// Uploads an image to the page's photo album (unpublished) to get a CDN URL
// that Facebook can use in generic template image_url without auth issues.
// Requires pages_manage_posts or pages_read_user_content permission.
// Returns CDN URL string on success, null on failure.
export async function uploadDataUrlToFbCdn(
  pageAccessToken: string,
  pageId: string,
  dataUrl: string,
): Promise<string | null> {
  try {
    const [meta, b64] = dataUrl.split(",") as [string, string];
    const mime = meta.match(/data:([^;]+)/)?.[1] ?? "image/jpeg";
    let buf = Buffer.from(b64, "base64");

    if (mime !== "image/jpeg" && mime !== "image/jpg") {
      buf = await sharp(buf).jpeg({ quality: 85 }).toBuffer() as Buffer<ArrayBuffer>;
    }

    const form = new FormData();
    form.append("source", new Blob([buf], { type: "image/jpeg" }), "product.jpg");
    form.append("published", "false");

    const uploadRes = await fetch(
      `https://graph.facebook.com/v25.0/${pageId}/photos?access_token=${pageAccessToken}`,
      { method: "POST", body: form }
    );

    if (!uploadRes.ok) {
      const errText = await uploadRes.text().catch(() => "");
      console.warn(`[fb-cdn] Photo upload failed: ${errText}`);
      return null;
    }

    const uploadData = await uploadRes.json() as { id?: string };
    const photoId = uploadData.id;
    if (!photoId) return null;

    const photoRes = await fetch(
      `https://graph.facebook.com/v25.0/${photoId}?fields=images&access_token=${pageAccessToken}`
    );
    if (!photoRes.ok) return null;

    const photoData = await photoRes.json() as { images?: Array<{ source: string; width: number }> };
    const cdnUrl = photoData.images?.[0]?.source ?? null;
    console.log(`[fb-cdn] Uploaded image, photoId=${photoId}, url=${cdnUrl ? cdnUrl.substring(0, 60) + "..." : "null"}`);
    return cdnUrl;
  } catch (err) {
    console.warn("[fb-cdn] Error uploading to Facebook CDN:", err instanceof Error ? err.message : String(err));
    return null;
  }
}

export async function sendFbButtonMessage(
  pageAccessToken: string,
  recipientId: string,
  text: string,
  buttons: Array<{ title: string; payload: string }>,
  pageId?: string
): Promise<void> {
  const endpoint = pageId ? `${pageId}/messages` : "me/messages";
  const resp = await fetch(
    `https://graph.facebook.com/v25.0/${endpoint}?access_token=${pageAccessToken}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        recipient: { id: recipientId },
        messaging_type: "RESPONSE",
        message: {
          attachment: {
            type: "template",
            payload: {
              template_type: "button",
              text,
              buttons: buttons.map((b) => ({
                type: "postback",
                title: b.title,
                payload: b.payload,
              })),
            },
          },
        },
      }),
    }
  );
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(`FB API error ${resp.status}: ${body}`);
  }
}

export async function sendFbGenericTemplate(
  pageAccessToken: string,
  recipientId: string,
  elements: Array<{
    title: string;
    subtitle?: string;
    image_url?: string;
    buttons?: Array<
      | { type: "postback"; title: string; payload: string }
      | { type: "web_url"; title: string; url: string }
    >;
  }>,
  pageId?: string
): Promise<void> {
  const PLACEHOLDER_IMAGE = "https://placehold.co/400x400/f8fafc/94a3b8?text=No+Image";
  const safeElements = elements.slice(0, 10).map((el) => ({
    title: el.title.substring(0, 80),
    subtitle: el.subtitle ? el.subtitle.substring(0, 80) : undefined,
    image_url: el.image_url || PLACEHOLDER_IMAGE,
    buttons: (el.buttons ?? []).slice(0, 3),
  }));

  const endpoint = pageId ? `${pageId}/messages` : "me/messages";
  const resp = await fetch(
    `https://graph.facebook.com/v25.0/${endpoint}?access_token=${pageAccessToken}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        recipient: { id: recipientId },
        messaging_type: "RESPONSE",
        message: {
          attachment: {
            type: "template",
            payload: {
              template_type: "generic",
              elements: safeElements,
            },
          },
        },
      }),
    }
  );
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(`FB Generic Template error ${resp.status}: ${body}`);
  }
}

export async function getFbUserName(
  pageAccessToken: string,
  userId: string
): Promise<{ name: string; profileUrl: string }> {
  const cacheKey = `fbuser:${userId}`;
  const cached = await rGet<{ name: string; profileUrl: string }>(cacheKey);
  if (cached) return cached;

  try {
    const res = await fetch(
      `https://graph.facebook.com/v25.0/${userId}?fields=name&access_token=${pageAccessToken}`
    );
    const data = (await res.json()) as { name?: string };
    const result = {
      name: data.name ?? userId,
      profileUrl: `https://www.facebook.com/${userId}`,
    };
    await rSet(cacheKey, result, TTL.FB_USER);
    return result;
  } catch {
    return { name: userId, profileUrl: `https://www.facebook.com/${userId}` };
  }
}
