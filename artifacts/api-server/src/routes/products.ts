import { Router, type IRouter } from "express";
import multer from "multer";
import sharp from "sharp";
import { db, productsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { rDel } from "../lib/redisCache.js";
import { callAIWithMetadata } from "../lib/ai.js";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { files: 5, fileSize: 8 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype.startsWith("image/")) cb(null, true);
    else cb(new Error("Only image files are allowed"));
  },
});

// Compress image to max 800px wide, JPEG 78% quality → typically <100KB per image
async function compressImage(file: Express.Multer.File): Promise<string> {
  const compressed = await sharp(file.buffer)
    .rotate()
    .resize({ width: 800, height: 800, fit: "inside", withoutEnlargement: true })
    .jpeg({ quality: 78, progressive: true })
    .toBuffer();
  return `data:image/jpeg;base64,${compressed.toString("base64")}`;
}

const router: IRouter = Router();

const safeFloat = (s: string | undefined, fallback: number | null): number | null => {
  if (!s) return fallback;
  const n = parseFloat(s);
  return Number.isNaN(n) ? fallback : n;
};

const safeInt = (s: string | undefined, fallback: number): number => {
  if (!s) return fallback;
  const n = parseInt(s, 10);
  return Number.isNaN(n) ? fallback : n;
};

// Public endpoint — serves a product image by index from DB (no auth required)
// Used by Facebook webhook to send product images via Messenger
router.get("/products/image/:id/:index", async (req, res): Promise<void> => {
  const id = parseInt(req.params["id"]!, 10);
  const index = parseInt(req.params["index"] ?? "0", 10);
  if (Number.isNaN(id) || Number.isNaN(index)) { res.status(400).end(); return; }

  const [product] = await db
    .select({ images: productsTable.images })
    .from(productsTable)
    .where(eq(productsTable.id, id))
    .limit(1);

  if (!product?.images) {
    res.status(404).end();
    return;
  }

  const imgs = JSON.parse(product.images) as string[];
  const dataUrl = imgs[index] ?? imgs[0];

  if (!dataUrl) {
    res.status(404).end();
    return;
  }

  if (dataUrl.startsWith("data:")) {
    const [meta, b64] = dataUrl.split(",") as [string, string];
    const mimeMatch = meta.match(/data:([^;]+)/);
    const mime = mimeMatch?.[1] ?? "image/jpeg";
    let buf = Buffer.from(b64, "base64");

    // Facebook Messenger لا يدعم WebP — نحوله إلى JPEG
    if (mime === "image/webp" || mime === "image/avif") {
      buf = await sharp(buf).jpeg({ quality: 85 }).toBuffer() as Buffer<ArrayBuffer>;
      res.set("Content-Type", "image/jpeg");
    } else {
      res.set("Content-Type", mime);
    }

    res.set("Cache-Control", "public, max-age=86400");
    res.end(buf);
  } else {
    res.redirect(302, dataUrl);
  }
});

router.get("/products", async (_req, res): Promise<void> => {
  const rows = await db.select().from(productsTable).orderBy(productsTable.id);
  res.json(rows);
});

router.post(
  "/products",
  upload.array("images[]", 5),
  async (req, res): Promise<void> => {
    const files = (req.files as Express.Multer.File[]) ?? [];
    const imageUrls = await Promise.all(files.map(compressImage));

    const body = req.body as Record<string, string>;
    const [row] = await db
      .insert(productsTable)
      .values({
        name: body["name"] ?? "Unnamed Product",
        description: body["description"] ?? null,
        originalPrice: safeFloat(body["originalPrice"], null),
        discountPrice: safeFloat(body["discountPrice"], null),
        stockQuantity: safeInt(body["stockQuantity"], 0),
        lowStockThreshold: safeInt(body["lowStockThreshold"], 5),
        status: body["status"] ?? "available",
        images: imageUrls.length > 0 ? JSON.stringify(imageUrls) : null,
        mainImageIndex: safeInt(body["mainImageIndex"], 0),
        category: body["category"] || null,
        brand: body["brand"] || null,
        itemType: body["itemType"] || null,
        priceTier: body["priceTier"] || null,
        externalUrl: body["externalUrl"] || null,
        folderId: body["folderId"] ? parseInt(body["folderId"], 10) : null,
      })
      .returning();

    await rDel("products:available");
    res.status(201).json(row);
  }
);

router.put(
  "/products/:id",
  upload.array("images[]", 5),
  async (req, res): Promise<void> => {
    const raw = Array.isArray(req.params["id"]) ? req.params["id"][0] : req.params["id"];
    const id = parseInt(raw!, 10);
    if (Number.isNaN(id)) { res.status(400).json({ message: "Invalid ID" }); return; }

    const [existing] = await db
      .select()
      .from(productsTable)
      .where(eq(productsTable.id, id))
      .limit(1);

    if (!existing) {
      res.status(404).json({ message: "Product not found" });
      return;
    }

    const files = (req.files as Express.Multer.File[]) ?? [];
    const newImageUrls = await Promise.all(files.map(compressImage));

    const body = req.body as Record<string, string>;
    const keepImages: string[] = body["keepImages"] ? JSON.parse(body["keepImages"]) : [];
    const allImages = [...keepImages, ...newImageUrls];

    const [updated] = await db
      .update(productsTable)
      .set({
        name: body["name"] ?? existing.name,
        description: body["description"] ?? existing.description,
        originalPrice: body["originalPrice"] !== undefined ? safeFloat(body["originalPrice"], existing.originalPrice) : existing.originalPrice,
        discountPrice: body["discountPrice"] !== undefined ? safeFloat(body["discountPrice"], existing.discountPrice) : existing.discountPrice,
        stockQuantity: body["stockQuantity"] !== undefined ? safeInt(body["stockQuantity"], existing.stockQuantity) : existing.stockQuantity,
        lowStockThreshold: body["lowStockThreshold"] !== undefined ? safeInt(body["lowStockThreshold"], existing.lowStockThreshold) : existing.lowStockThreshold,
        status: body["status"] ?? existing.status,
        images: allImages.length > 0 ? JSON.stringify(allImages) : existing.images,
        mainImageIndex: body["mainImageIndex"] !== undefined ? safeInt(body["mainImageIndex"], existing.mainImageIndex) : existing.mainImageIndex,
        category: body["category"] !== undefined ? (body["category"] || null) : existing.category,
        brand: body["brand"] !== undefined ? (body["brand"] || null) : existing.brand,
        itemType: body["itemType"] !== undefined ? (body["itemType"] || null) : existing.itemType,
        priceTier: body["priceTier"] !== undefined ? (body["priceTier"] || null) : existing.priceTier,
        externalUrl: body["externalUrl"] !== undefined ? (body["externalUrl"] || null) : existing.externalUrl,
        folderId: body["folderId"] !== undefined
          ? (body["folderId"] ? parseInt(body["folderId"], 10) : null)
          : existing.folderId,
      })
      .where(eq(productsTable.id, id))
      .returning();

    await rDel("products:available");
    res.json(updated);
  }
);

router.delete("/products/:id", async (req, res): Promise<void> => {
  const raw = Array.isArray(req.params["id"]) ? req.params["id"][0] : req.params["id"];
  const id = parseInt(raw!, 10);
  if (Number.isNaN(id)) { res.status(400).json({ message: "Invalid ID" }); return; }
  await db.delete(productsTable).where(eq(productsTable.id, id));
  await rDel("products:available");
  res.json({ message: "Product deleted" });
});

router.patch("/products/:id/stock", async (req, res): Promise<void> => {
  const raw = Array.isArray(req.params["id"]) ? req.params["id"][0] : req.params["id"];
  const id = parseInt(raw!, 10);
  if (Number.isNaN(id)) { res.status(400).json({ message: "Invalid ID" }); return; }

  const { quantityChange } = req.body as { quantityChange: number };
  if (typeof quantityChange !== "number") {
    res.status(400).json({ message: "quantityChange must be a number" });
    return;
  }

  const [existing] = await db
    .select()
    .from(productsTable)
    .where(eq(productsTable.id, id))
    .limit(1);

  if (!existing) {
    res.status(404).json({ message: "Product not found" });
    return;
  }

  const newQty = Math.max(0, existing.stockQuantity + quantityChange);
  const newStatus =
    newQty === 0 ? "out_of_stock" : existing.status === "out_of_stock" ? "available" : existing.status;

  const [updated] = await db
    .update(productsTable)
    .set({ stockQuantity: newQty, status: newStatus })
    .where(eq(productsTable.id, id))
    .returning();

  await rDel("products:available");
  res.json(updated);
});

// ── POST /api/products/summarize-description ──────────────────────────────────
// Calls AI to extract key specs from a long product description.
router.post("/products/summarize-description", async (req, res): Promise<void> => {
  const { description } = req.body as { description?: string };
  if (!description || description.trim().length < 20) {
    res.status(400).json({ error: "الوصف قصير جداً أو فارغ" });
    return;
  }

  const systemPrompt = `أنت مساعد متخصص في استخراج مواصفات المنتجات التقنية من الأوصاف الطويلة.
مهمتك: استخرج المواصفات المهمة فقط من الوصف التالي وأعدها بشكل مختصر ومنظم لا يتجاوز 200 كلمة.
ركز على: الرامات، المعالج، المساحة الداخلية، البطارية، الشاشة، الكاميرا، نظام التشغيل، الميزات الرئيسية.
أعد النتيجة فقط دون مقدمة أو تعليق.`;

  try {
    const result = await callAIWithMetadata(
      [{ role: "user", content: description }],
      systemPrompt
    );
    res.json({ summary: result.text.trim() });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "فشل الاتصال بالذكاء الاصطناعي";
    res.status(500).json({ error: msg });
  }
});

export default router;
