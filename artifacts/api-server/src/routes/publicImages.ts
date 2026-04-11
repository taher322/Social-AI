import { Router, type IRouter } from "express";
import sharp from "sharp";
import { db, productsTable, broadcastsTable } from "@workspace/db";
import { eq } from "drizzle-orm";

const router: IRouter = Router();

// ── GET /api/products/image/:id/:index ─────────────────────────────────────────
// Public — no auth. Facebook/Messenger fetches product card images from here.
router.get("/api/products/image/:id/:index", async (req, res): Promise<void> => {
  const id    = parseInt(req.params["id"]!,    10);
  const index = parseInt(req.params["index"] ?? "0", 10);
  if (Number.isNaN(id) || Number.isNaN(index)) { res.status(400).end(); return; }

  const [product] = await db
    .select({ images: productsTable.images })
    .from(productsTable)
    .where(eq(productsTable.id, id))
    .limit(1);

  if (!product?.images) { res.status(404).end(); return; }

  const imgs    = JSON.parse(product.images) as string[];
  const dataUrl = imgs[index] ?? imgs[0];
  if (!dataUrl)           { res.status(404).end(); return; }

  if (dataUrl.startsWith("data:")) {
    const [, b64] = dataUrl.split(",") as [string, string];
    const raw = Buffer.from(b64, "base64");

    // Always convert to JPEG — Facebook Messenger requires JPEG for card images.
    // This also handles images stored with a wrong mime prefix (e.g. declared as
    // image/jpeg but actually WebP binary), which would cause broken images.
    let buf: Buffer;
    try {
      buf = await sharp(raw).jpeg({ quality: 85 }).toBuffer();
    } catch {
      // sharp failed (unsupported format) — serve the raw buffer as-is
      buf = raw;
    }
    res.set("Content-Type", "image/jpeg");
    res.set("Cache-Control", "public, max-age=86400");
    res.send(buf);
  } else {
    res.redirect(302, dataUrl);
  }
});

// ── GET /api/broadcasts/image/:id ──────────────────────────────────────────────
// Public — no auth. Used for broadcast image previews.
router.get("/api/broadcasts/image/:id", async (req, res): Promise<void> => {
  const id = Number(req.params["id"]);
  const [broadcast] = await db
    .select({ imageUrl: broadcastsTable.imageUrl })
    .from(broadcastsTable)
    .where(eq(broadcastsTable.id, id))
    .limit(1);

  if (!broadcast?.imageUrl) { res.status(404).end(); return; }

  const dataUrl = broadcast.imageUrl;
  if (dataUrl.startsWith("data:")) {
    const [meta, b64] = dataUrl.split(",") as [string, string];
    const mimeMatch   = meta.match(/data:([^;]+)/);
    const mime        = mimeMatch?.[1] ?? "image/jpeg";
    const buf         = Buffer.from(b64, "base64");
    res.set("Content-Type", mime);
    res.set("Cache-Control", "public, max-age=86400");
    res.send(buf);
  } else {
    res.redirect(302, dataUrl);
  }
});

export default router;
