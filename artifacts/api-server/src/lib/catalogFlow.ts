import { db, productsTable, deliveryPricesTable, aiConfigTable, productCategoriesTable } from "@workspace/db";
import { eq, and, or, isNull, sql } from "drizzle-orm";
import { sendFbMessage, sendFbGenericTemplate, uploadDataUrlToFbCdn, sendFbImageFromDataUrl, sendFbButtonMessage } from "./ai.js";
import { sendFbQuickReplies } from "./messengerUtils.js";
import { getCatEmoji, buildCatFullPath } from "./webhookUtils.js";
import { ALGERIA_WILAYAS } from "../routes/deliveryPrices.js";

// بناء جميع مقاطع المسار: "A/B/C" → ["A", "A/B", "A/B/C"]
function buildPathSegments(path: string): string[] {
  const parts = path.split("/");
  const segs: string[] = [];
  let cur = "";
  for (const part of parts) {
    cur = cur ? `${cur}/${part}` : part;
    segs.push(cur);
  }
  return segs;
}

// هل هذا المسار يملك منتجات؟ (بمطابقة أي من: مسار أب، مطابقة كاملة، مسار أعمق)
function pathHasProducts(fullPath: string, productCategories: string[]): boolean {
  return productCategories.some((pc) => {
    if (!pc) return false;
    return (
      pc === fullPath ||                     // مطابقة كاملة
      pc.startsWith(fullPath + "/") ||       // المنتج في تصنيف فرعي أعمق
      fullPath.startsWith(pc + "/") ||       // المنتج مخزن في تصنيف أب
      fullPath === pc                        // نفس المسار
    );
  });
}

// ── Catalog filter type ───────────────────────────────────────────────────────
export type CatalogFilters = {
  category?: string;
  categoryLeafName?: string;
  brand?: string;
  priceTier?: string;
  uncategorized?: boolean;
};

// ── Delivery options ──────────────────────────────────────────────────────────
export async function sendDeliveryOptions(
  pageAccessToken: string,
  senderId: string,
  wilayaName: string,
  pageId?: string
): Promise<void> {
  const wilayaRecord = ALGERIA_WILAYAS.find((w) => w.wilayaName === wilayaName);
  const [wp] = wilayaRecord
    ? await db.select().from(deliveryPricesTable)
        .where(eq(deliveryPricesTable.wilayaId, wilayaRecord.wilayaId)).limit(1)
    : await db.select().from(deliveryPricesTable)
        .where(eq(deliveryPricesTable.wilayaName, wilayaName)).limit(1);
  const [appConf] = await db.select().from(aiConfigTable).limit(1);
  const currency = appConf?.currency ?? "DZD";

  const homePrice   = wp?.homePrice  ?? 0;
  const officePrice = wp?.officePrice ?? 0;
  const fmtPrice = (p: number) => p === 0 ? "مجاني 🎁" : `${p} ${currency}`;

  const promptMsg = `🚚 اختر نوع التوصيل إلى ${wilayaName}:`;
  await sendFbQuickReplies(pageAccessToken, senderId, promptMsg, [
    { title: `🏠 للمنزل — ${fmtPrice(homePrice)}`,       payload: "DELIVERY_HOME" },
    { title: `🏢 مكتب البريد — ${fmtPrice(officePrice)}`, payload: "DELIVERY_OFFICE" },
  ], pageId);
}

// ── Catalog category menu ─────────────────────────────────────────────────────
export async function sendCatalogCategoryMenu(
  pageAccessToken: string,
  senderId: string,
  pageId?: string
): Promise<void> {
  const rootCats = await db.select()
    .from(productCategoriesTable)
    .where(isNull(productCategoriesTable.parentId))
    .orderBy(productCategoriesTable.id);

  // ── استخرج التصنيفات الجذرية التي فيها منتجات متاحة فعلاً ──────────────────
  const productCatRows = await db
    .selectDistinct({ category: productsTable.category })
    .from(productsTable)
    .where(eq(productsTable.status, "available"));

  const usedRoots = new Set(
    productCatRows
      .map((r) => r.category)
      .filter(Boolean)
      .map((c) => c!.split("/")[0]!)
  );

  const catsWithProducts = rootCats.filter((cat) => usedRoots.has(cat.name));

  // ── لا يوجد تصنيفات (أو كلها فارغة) → اعرض المنتجات مباشرة ────────────────
  if (catsWithProducts.length === 0) {
    const availableProducts = await db.select({ id: productsTable.id })
      .from(productsTable)
      .where(eq(productsTable.status, "available"))
      .limit(1);

    if (availableProducts.length === 0) {
      await sendFbMessage(pageAccessToken, senderId, "لا توجد منتجات متاحة حالياً. تواصل معنا لمزيد من المعلومات.", pageId);
      return;
    }

    await sendCatalogPage(pageAccessToken, senderId, {}, 1, pageId);
    return;
  }

  // ── يوجد تصنيفات بها منتجات → أضف زر للمنتجات غير المصنفة إن وُجدت ────────
  const uncatProducts = await db.select({ id: productsTable.id })
    .from(productsTable)
    .where(and(eq(productsTable.status, "available"), isNull(productsTable.category)))
    .limit(1);

  const quickReplies = catsWithProducts.slice(0, uncatProducts.length > 0 ? 9 : 10).map((cat) => ({
    title: `${getCatEmoji(cat.name)} ${cat.name}`.substring(0, 20),
    payload: `BROWSE_SUB:${cat.id}`,
  }));

  if (uncatProducts.length > 0) {
    quickReplies.push({ title: "📦 منتجات أخرى", payload: "BROWSE_UNCATEGORIZED" });
  }

  await sendFbQuickReplies(
    pageAccessToken, senderId,
    "🛍️ اختر الفئة التي تريد تصفحها:",
    quickReplies, pageId
  );
}

// ── Catalog page (paginated product carousel) ─────────────────────────────────
export async function sendCatalogPage(
  pageAccessToken: string,
  senderId: string,
  filters: CatalogFilters,
  page: number,
  pageId?: string
): Promise<void> {
  const PAGE_SIZE = 10;
  const offset = (page - 1) * PAGE_SIZE;

  const conditions = [eq(productsTable.status, "available")];
  if (filters.uncategorized) {
    conditions.push(isNull(productsTable.category));
  } else if (filters.category) {
    // ── بناء شروط التطابق الشاملة ──────────────────────────────────────────────
    // نطابق: كل مقاطع المسار الأب + المسار الكامل + المسارات الأعمق + اسم الورقة
    const segments = buildPathSegments(filters.category);
    const catConds = [
      // مطابقة كل مقطع أب (مثل "Realme" عند الفلترة بـ "Realme/47000 الي 59000")
      ...segments.map((seg) => eq(productsTable.category, seg)),
      // مطابقة تصنيفات أعمق (مثل "Realme/47000 الي 59000/subset")
      sql`${productsTable.category} LIKE ${filters.category + "/%"}`,
    ];
    // اسم الورقة كاختصار إن اختلف
    if (filters.categoryLeafName && !segments.includes(filters.categoryLeafName)) {
      catConds.push(eq(productsTable.category, filters.categoryLeafName));
    }
    conditions.push(or(...catConds)!);
  }
  if (filters.brand)     conditions.push(eq(productsTable.brand, filters.brand));
  if (filters.priceTier) conditions.push(eq(productsTable.priceTier, filters.priceTier));

  const allMatching = await db.select().from(productsTable)
    .where(and(...conditions))
    .orderBy(productsTable.id);

  const matching = allMatching.slice(offset, offset + PAGE_SIZE);

  if (matching.length === 0) {
    const filterDesc = [
      filters.category  ? `الفئة: ${filters.category}`  : null,
      filters.brand     ? `العلامة: ${filters.brand}`    : null,
      filters.priceTier ? `السعر: ${filters.priceTier}`  : null,
    ].filter(Boolean).join("، ");
    await sendFbMessage(
      pageAccessToken, senderId,
      `عذراً، لا توجد منتجات تطابق (${filterDesc || "الفلتر المحدد"}). هل تريد تصفح فئات أخرى؟`,
      pageId
    );
    await sendFbQuickReplies(
      pageAccessToken, senderId,
      "اختر:", [{ title: "🏠 كل الفئات", payload: "BROWSE_CATALOG" }],
      pageId
    );
    return;
  }

  const PLACEHOLDER = "https://placehold.co/400x400/f8fafc/94a3b8?text=No+Image";

  // ── Build product cards ──────────────────────────────────────────────────────
  // For each product, resolve the image URL:
  //   1. If fbImageUrl cached → use it (Facebook CDN, public)
  //   2. If data URL image → try to upload to Facebook CDN → cache → use
  //   3. If external URL image → use directly
  //   4. Fallback → placeholder (image will still show correctly from FB CDN)
  //
  // Products that fail CDN upload are sent as individual image+button messages.

  type CardProduct = typeof matching[number] & { resolvedImageUrl: string };
  const carouselProducts: CardProduct[] = [];
  const individualProducts: CardProduct[] = [];

  for (const p of matching) {
    let resolvedImageUrl = PLACEHOLDER;

    try {
      // Check cached CDN URL first
      if (p.fbImageUrl) {
        resolvedImageUrl = p.fbImageUrl;
      } else if (p.images) {
        const imgs = JSON.parse(p.images) as string[];
        const idx = p.mainImageIndex ?? 0;
        const img = imgs[idx] ?? imgs[0];

        if (img) {
          if (img.startsWith("data:")) {
            // Data URL: try upload to Facebook CDN
            const cdnUrl = pageId
              ? await uploadDataUrlToFbCdn(pageAccessToken, pageId, img).catch(() => null)
              : null;

            if (cdnUrl) {
              resolvedImageUrl = cdnUrl;
              // Cache for future use
              await db.update(productsTable)
                .set({ fbImageUrl: cdnUrl })
                .where(eq(productsTable.id, p.id))
                .catch(() => {/* non-critical */});
              console.log(`[catalogFlow] Cached Facebook CDN URL for product ${p.id}`);
            } else {
              // CDN upload failed → send this product individually
              const card: CardProduct = { ...p, resolvedImageUrl };
              individualProducts.push(card);
              continue;
            }
          } else {
            // External URL → use directly
            resolvedImageUrl = img;
          }
        }
      }
    } catch (e) {
      console.warn("[catalogFlow] Image resolution failed for product", p.id, (e as Error).message);
    }

    carouselProducts.push({ ...p, resolvedImageUrl });
  }

  // ── Send individual products (data URL, CDN failed) ──────────────────────────
  for (const p of individualProducts) {
    try {
      const imgs = JSON.parse(p.images ?? "[]") as string[];
      const idx = p.mainImageIndex ?? 0;
      const dataUrl = imgs[idx] ?? imgs[0];
      if (dataUrl) {
        await sendFbImageFromDataUrl(pageAccessToken, senderId, dataUrl, pageId);
      }
      const price    = p.discountPrice ?? p.originalPrice;
      const priceStr = price ? `${price} دج` : "اتصل للسعر";
      const isOutOfStock = (p.stockQuantity ?? 0) === 0;
      await sendFbButtonMessage(
        pageAccessToken, senderId,
        `${p.name.substring(0, 60)}\n${priceStr}`,
        [
          { title: "📋 التفاصيل", payload: `DETAILS:${p.id}` },
          isOutOfStock
            ? { title: "⏳ طلب مسبق", payload: `PREORDER_START:${p.id}` }
            : { title: "🛒 اطلب الآن", payload: `ORDER_NOW:${p.id}` },
        ],
        pageId
      );
    } catch (e) {
      console.warn("[catalogFlow] Failed to send individual product", p.id, (e as Error).message);
    }
  }

  // ── Send carousel for products with resolved image URLs ──────────────────────
  const elements = carouselProducts.map((p) => {
    const price    = p.discountPrice ?? p.originalPrice;
    const priceStr = price ? `${price} دج` : "اتصل للسعر";
    const tierLabel: Record<string, string> = { budget: "💚", mid_range: "💛", premium: "💎" };
    const tierIcon = p.priceTier ? (tierLabel[p.priceTier] ?? "") : "";

    const subtitleParts: string[] = [];
    if (p.brand)       subtitleParts.push(p.brand);
    if (p.itemType)    subtitleParts.push(p.itemType);
    subtitleParts.push(`${tierIcon} ${priceStr}`.trim());
    if (p.description) subtitleParts.push(p.description.substring(0, 30));

    const isOutOfStock = (p.stockQuantity ?? 0) === 0;
    const buttons: Array<
      | { type: "postback"; title: string; payload: string }
      | { type: "web_url"; title: string; url: string }
    > = [
      { type: "postback", title: "📋 التفاصيل", payload: `DETAILS:${p.id}` },
      isOutOfStock
        ? { type: "postback", title: "⏳ طلب مسبق", payload: `PREORDER_START:${p.id}` }
        : { type: "postback", title: "🛒 اطلب الآن", payload: `ORDER_NOW:${p.id}` },
    ];
    if (p.externalUrl) {
      buttons.push({ type: "web_url", title: "🔗 رابط المنتج", url: p.externalUrl });
    }

    return {
      title:     p.name.substring(0, 80),
      subtitle:  subtitleParts.join(" | ").substring(0, 80),
      image_url: p.resolvedImageUrl,
      buttons,
    };
  });

  const totalVisible = elements.length + individualProducts.length;
  if (totalVisible > 1) {
    await sendFbMessage(pageAccessToken, senderId, `وجدنا ${totalVisible} منتج 👇 اسحب ← لرؤية جميعها:`, pageId);
  }

  if (elements.length > 0) {
    await sendFbGenericTemplate(pageAccessToken, senderId, elements, pageId);
  }

  const hasMore = allMatching.length > offset + PAGE_SIZE;
  if (hasMore) {
    const filtersEncoded = Object.entries(filters)
      .filter(([, v]) => v)
      .map(([k, v]) => `${k}=${encodeURIComponent(v!)}`)
      .join("&");
    await sendFbQuickReplies(
      pageAccessToken, senderId,
      `عرض ${matching.length} من ${allMatching.length} — اضغط لرؤية المزيد:`,
      [
        { title: "⬅️ المزيد",   payload: `BROWSE_PAGE:${filtersEncoded}:${page + 1}` },
        { title: "🏠 الفئات",   payload: "BROWSE_CATALOG" },
      ],
      pageId
    );
  } else {
    await sendFbQuickReplies(
      pageAccessToken, senderId,
      allMatching.length === 1
        ? "هذا هو المنتج المتاح في هذه الفئة 👆"
        : `تم عرض كل المنتجات (${allMatching.length}) 👆`,
      [{ title: "🏠 كل الفئات", payload: "BROWSE_CATALOG" }],
      pageId
    );
  }
}

// ── Browse sub-category handler ───────────────────────────────────────────────
export async function handleBrowseSub(
  pageAccessToken: string,
  senderId: string,
  catId: number,
  pageId?: string
): Promise<boolean> {
  const [thisCat] = await db.select()
    .from(productCategoriesTable)
    .where(eq(productCategoriesTable.id, catId))
    .limit(1);
  if (!thisCat) return false;

  const allCats = await db.select().from(productCategoriesTable);

  const children = await db.select()
    .from(productCategoriesTable)
    .where(eq(productCategoriesTable.parentId, catId))
    .orderBy(productCategoriesTable.id);

  if (children.length > 0) {
    // ── تحقق من وجود منتجات في كل فئة فرعية ─────────────────────────────────
    const productCatRows = await db
      .selectDistinct({ category: productsTable.category })
      .from(productsTable)
      .where(eq(productsTable.status, "available"));
    const productCategories = productCatRows.map((r) => r.category ?? "");

    const childrenWithProducts = children.filter((child) => {
      const childPath = buildCatFullPath(child.id, allCats);
      return pathHasProducts(childPath, productCategories);
    });

    if (childrenWithProducts.length === 0) {
      // الفئات الفرعية كلها فارغة → اعرض مباشرة للكتالوج بالتصنيف الأب
      const fullPath = buildCatFullPath(catId, allCats);
      await sendCatalogPage(pageAccessToken, senderId, { category: fullPath, categoryLeafName: thisCat.name }, 1, pageId);
    } else {
      const quickReplies = childrenWithProducts.slice(0, 10).map((child) => ({
        title: `${getCatEmoji(child.name)} ${child.name}`.substring(0, 20),
        payload: `BROWSE_SUB:${child.id}`,
      }));
      await sendFbQuickReplies(
        pageAccessToken, senderId,
        `📂 ${thisCat.name} — اختر:`,
        quickReplies, pageId
      );
    }
  } else {
    const fullPath = buildCatFullPath(catId, allCats);
    await sendCatalogPage(
      pageAccessToken, senderId,
      { category: fullPath, categoryLeafName: thisCat.name },
      1, pageId
    );
  }
  return true;
}
