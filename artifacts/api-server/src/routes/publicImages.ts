import { Router, type IRouter } from "express";
import sharp from "sharp";
import { db, productsTable, broadcastsTable } from "@workspace/db";
import { eq } from "drizzle-orm";

const router: IRouter = Router();

// ── Product image handler ──────────────────────────────────────────────────────
// Registered on both /api/products/image/:id/:index and /products/image/:id/:index
// to handle proxies that may or may not strip the /api prefix.
async function serveProductImage(req: import("express").Request, res: import("express").Response): Promise<void> {
  const idParam    = Array.isArray(req.params["id"])    ? req.params["id"][0]!    : req.params["id"]!;
  const indexParam = Array.isArray(req.params["index"]) ? req.params["index"][0]! : req.params["index"] ?? "0";
  const id    = parseInt(idParam,    10);
  const index = parseInt(indexParam, 10);
  if (Number.isNaN(id) || Number.isNaN(index)) { res.status(400).end(); return; }

  const [product] = await db
    .select({ images: productsTable.images })
    .from(productsTable)
    .where(eq(productsTable.id, id))
    .limit(1);

  if (!product?.images) { res.status(404).end(); return; }

  const imgs    = JSON.parse(product.images) as string[];
  const dataUrl = imgs[index] ?? imgs[0];
  if (!dataUrl) { res.status(404).end(); return; }

  if (dataUrl.startsWith("data:")) {
    const [, b64] = dataUrl.split(",") as [string, string];
    const raw = Buffer.from(b64, "base64");
    let buf: Buffer;
    try {
      buf = await sharp(raw).jpeg({ quality: 85 }).toBuffer();
    } catch {
      buf = raw;
    }
    res.set("Content-Type", "image/jpeg");
    res.set("Cache-Control", "public, max-age=86400");
    res.send(buf);
  } else {
    res.redirect(302, dataUrl);
  }
}

router.get("/api/products/image/:id/:index", serveProductImage);
router.get("/products/image/:id/:index",     serveProductImage);

// ── Broadcast image handler ────────────────────────────────────────────────────
async function serveBroadcastImage(req: import("express").Request, res: import("express").Response): Promise<void> {
  const id = Number(req.params["id"]);
  const [broadcast] = await db
    .select({ imageUrl: broadcastsTable.imageUrl })
    .from(broadcastsTable)
    .where(eq(broadcastsTable.id, id))
    .limit(1);

  if (!broadcast?.imageUrl) { res.status(404).end(); return; }

  const dataUrl = broadcast.imageUrl;
  if (dataUrl.startsWith("data:")) {
    const [, b64] = dataUrl.split(",") as [string, string];
    const raw = Buffer.from(b64, "base64");
    let buf: Buffer;
    try {
      buf = await sharp(raw).jpeg({ quality: 85 }).toBuffer();
    } catch {
      buf = raw;
    }
    res.set("Content-Type", "image/jpeg");
    res.set("Cache-Control", "public, max-age=86400");
    res.send(buf);
  } else {
    res.redirect(302, dataUrl);
  }
}

router.get("/api/broadcasts/image/:id", serveBroadcastImage);
router.get("/broadcasts/image/:id",     serveBroadcastImage);

export default router;
