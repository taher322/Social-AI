import http from "http";
import sharp from "sharp";
import { pool, db, aiProvidersTable, aiConfigTable, productInquiriesTable, fbSettingsTable, conversationsTable, broadcastsTable, leadsTable, platformEventsTable, processedMessagesTable, productsTable } from "@workspace/db";
import { eq, and, sql, lte, lt } from "drizzle-orm";
import app from "./app.js";
import { runSeed } from "./lib/seed.js";
import { decrypt } from "./lib/encryption.js";
import { sendFbMessage, sendFbImageFromDataUrl } from "./lib/ai.js";

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error("PORT environment variable is required but was not provided.");
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

const encryptionKey = process.env["ENCRYPTION_KEY"];
if (!encryptionKey) {
  throw new Error(
    "[server] ENCRYPTION_KEY environment variable is required. " +
    "AI provider API keys are encrypted at rest using AES-256-CBC and cannot be stored or decrypted without this key. " +
    "Set ENCRYPTION_KEY to a random string of at least 32 characters in your environment configuration."
  );
}
if (encryptionKey.length < 32) {
  console.warn(
    `[server] WARNING: ENCRYPTION_KEY is only ${encryptionKey.length} character(s) — ` +
    "minimum 32 characters recommended for AES-256-CBC (shorter keys will be zero-padded, reducing security)"
  );
}

if (!process.env["JWT_SECRET"]) {
  console.warn("[server] WARNING: JWT_SECRET not set — dashboard sessions will invalidate on every server restart");
}

async function checkActiveProvider(): Promise<void> {
  try {
    const [active] = await db
      .select()
      .from(aiProvidersTable)
      .where(eq(aiProvidersTable.isActive, 1))
      .limit(1);

    if (!active) {
      console.warn("⚠️  No active AI provider configured.\n     Go to /providers and activate one with a valid API key.");
      return;
    }

    const key = decrypt(active.apiKey);
    if (!key) {
      console.warn(`⚠️  Active provider "${active.name}" has no API key.\n     Go to /providers and set a valid API key.`);
    }
  } catch {
    // Non-critical — just a startup hint
  }
}

async function runAbandonedCartReminder(): Promise<void> {
  try {
    const [config] = await db.select().from(aiConfigTable).limit(1);
    if (!config || !config.abandonedCartEnabled) return;

    const [settings] = await db.select().from(fbSettingsTable).limit(1);
    if (!settings?.pageAccessToken) return;

    const delayHours = config.abandonedCartDelayHours ?? 1;
    const delayMs = delayHours * 60 * 60 * 1000;
    const cutoff = new Date(Date.now() - delayMs).toISOString();

    const inquiries = await db.select().from(productInquiriesTable)
      .where(and(
        eq(productInquiriesTable.converted, 0),
        eq(productInquiriesTable.reminderSent, 0),
        sql`${productInquiriesTable.inquiredAt} < ${cutoff}`
      ));

    if (inquiries.length === 0) return;

    const template = config.abandonedCartMessage ?? "مرحباً! 👋 لاحظنا اهتمامك بـ {product_name}\nهل تريد إتمام طلبك؟ نحن هنا لمساعدتك 😊";
    const pageName = config.pageName ?? "";

    for (const inq of inquiries) {
      const msg = template
        .replace(/\{product_name\}/g, inq.productName)
        .replace(/\{page_name\}/g, pageName);

      try {
        await sendFbMessage(settings.pageAccessToken, inq.fbUserId, msg, settings.pageId ?? undefined);
        await db.insert(conversationsTable).values({
          fbUserId: inq.fbUserId,
          fbUserName: inq.fbUserName,
          message: msg,
          sender: "bot",
          timestamp: new Date(),
        });
        await db.update(productInquiriesTable)
          .set({ reminderSent: 1 })
          .where(eq(productInquiriesTable.id, inq.id));
        console.log(`[abandoned-cart] Sent reminder to ${inq.fbUserId} for "${inq.productName}"`);
      } catch (err: unknown) {
        const errMsg = err instanceof Error ? err.message : String(err);
        console.error(`[abandoned-cart] Failed to send reminder to ${inq.fbUserId}:`, errMsg);
      }
    }
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    console.error("[abandoned-cart] Job error:", errMsg);
  }
}

async function runScheduledBroadcasts(): Promise<void> {
  try {
    const now = new Date().toISOString();

    const pending = await db
      .select()
      .from(broadcastsTable)
      .where(
        and(
          eq(broadcastsTable.status, "draft"),
          lte(broadcastsTable.scheduledAt, now)
        )
      );

    if (pending.length === 0) return;

    const [fbRow] = await db
      .select({ pageAccessToken: fbSettingsTable.pageAccessToken, pageId: fbSettingsTable.pageId })
      .from(fbSettingsTable)
      .limit(1);
    const token = fbRow?.pageAccessToken ?? null;
    const fbPageId = fbRow?.pageId ?? undefined;

    if (!token) return;

    for (const broadcast of pending) {
      // Claim the record immediately to prevent double-sending
      await db
        .update(broadcastsTable)
        .set({ status: "sending" })
        .where(and(eq(broadcastsTable.id, broadcast.id), eq(broadcastsTable.status, "draft")));

      try {
        const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
        let activeUserIds: string[] = [];

        if (broadcast.targetFilter === "label" && broadcast.targetLabel) {
          const labelLeads = await db
            .select({ fbUserId: leadsTable.fbUserId })
            .from(leadsTable)
            .where(eq(leadsTable.label, broadcast.targetLabel));
          const labelUserIds = labelLeads.map((l) => l.fbUserId);
          if (labelUserIds.length > 0) {
            const recentRows = await db.execute(sql`
              SELECT fb_user_id FROM conversations
              WHERE fb_user_id = ANY(${labelUserIds})
              GROUP BY fb_user_id HAVING MAX(timestamp) > ${twentyFourHoursAgo}
            `);
            activeUserIds = (recentRows.rows as { fb_user_id: string }[]).map((r) => r.fb_user_id);
          }
        } else if (broadcast.targetFilter === "appointments") {
          const apptRows = await db.execute(sql`
            SELECT DISTINCT a.fb_user_id FROM appointments a
            WHERE a.status IN ('pending', 'confirmed')
              AND EXISTS (
                SELECT 1 FROM (
                  SELECT fb_user_id, MAX(timestamp) AS last_ts FROM conversations GROUP BY fb_user_id
                ) sub WHERE sub.fb_user_id = a.fb_user_id AND sub.last_ts > ${twentyFourHoursAgo}
              )
          `);
          activeUserIds = (apptRows.rows as { fb_user_id: string }[]).map((r) => r.fb_user_id);
        } else {
          const recentRows = await db.execute(sql`
            SELECT fb_user_id FROM conversations
            GROUP BY fb_user_id HAVING MAX(timestamp) > ${twentyFourHoursAgo}
          `);
          activeUserIds = (recentRows.rows as { fb_user_id: string }[]).map((r) => r.fb_user_id);
        }

        const totalRecipients = activeUserIds.length;
        let sentCount = 0;

        for (const userId of activeUserIds) {
          let textSent = false;
          if (broadcast.imageUrl) {
            try {
              await sendFbImageFromDataUrl(token, userId, broadcast.imageUrl, fbPageId);
            } catch (e) {
              console.warn(`[broadcast] Image send failed for user ${userId}:`, e instanceof Error ? e.message : String(e));
            }
          }
          try {
            await sendFbMessage(token, userId, broadcast.messageText, fbPageId);
            textSent = true;
          } catch (e) {
            console.warn(`[broadcast] Text send failed for user ${userId}:`, e instanceof Error ? e.message : String(e));
          }
          if (textSent) sentCount++;
        }

        await db
          .update(broadcastsTable)
          .set({ status: "sent", sentCount, totalRecipients, sentAt: new Date().toISOString() })
          .where(eq(broadcastsTable.id, broadcast.id));

        console.log(`[scheduled-broadcast] Sent broadcast #${broadcast.id} "${broadcast.title}" to ${sentCount}/${totalRecipients} users`);
      } catch (err: unknown) {
        const errMsg = err instanceof Error ? err.message : String(err);
        console.error(`[scheduled-broadcast] Failed to send broadcast #${broadcast.id}:`, errMsg);
        await db
          .update(broadcastsTable)
          .set({ status: "failed" })
          .where(eq(broadcastsTable.id, broadcast.id));
      }
    }
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    console.error("[scheduled-broadcast] Job error:", errMsg);
  }
}

async function runProcessedMessagesCleanup(): Promise<void> {
  try {
    const cutoff = new Date(Date.now() - 2 * 60 * 60 * 1000);
    const result = await db
      .delete(processedMessagesTable)
      .where(lt(processedMessagesTable.processedAt, cutoff));
    const deleted = result.rowCount ?? 0;
    if (deleted > 0) {
      console.log(`[idempotency-cleanup] Deleted ${deleted} processed message IDs older than 2h`);
    }
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    console.error("[idempotency-cleanup] Job error:", errMsg);
  }
}

async function runPlatformEventsCleanup(): Promise<void> {
  try {
    const cutoff = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
    const result = await db
      .delete(platformEventsTable)
      .where(lt(platformEventsTable.createdAt, cutoff));
    const deleted = result.rowCount ?? 0;
    if (deleted > 0) {
      console.log(`[platform-events-cleanup] Deleted ${deleted} events older than 30 days`);
    }
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    console.error("[platform-events-cleanup] Job error:", errMsg);
  }
}

async function start() {
  try {
    await pool.query("SELECT 1");
    console.log("[server] Database connected");
    await runSeed();
    await checkActiveProvider();
  } catch (err) {
    console.error("[server] Database connection failed:", err);
  }

  setInterval(runAbandonedCartReminder, 30 * 60 * 1000);
  console.log("[server] Abandoned cart reminder job scheduled (every 30 min)");

  setInterval(runScheduledBroadcasts, 60 * 1000);
  console.log("[server] Scheduled broadcasts job running (every 60 sec)");

  void runProcessedMessagesCleanup();
  setInterval(runProcessedMessagesCleanup, 60 * 60 * 1000);
  console.log("[server] Idempotency cleanup job scheduled (every 1h, removes mids older than 2h)");

  void runPlatformEventsCleanup();
  setInterval(runPlatformEventsCleanup, 24 * 60 * 60 * 1000);
  console.log("[server] Platform events cleanup job scheduled (every 24h, keeps last 30 days)");

  const server = http.createServer(async (req, res) => {
    const rawUrl  = req.url ?? "/";
    const urlPath = rawUrl.split("?")[0]!;

    const productMatch = urlPath.match(/^\/api\/products\/image\/(\d+)\/(\d+)$/);
    if (productMatch && req.method === "GET") {
      try {
        const id    = parseInt(productMatch[1]!, 10);
        const index = parseInt(productMatch[2]!, 10);
        const [product] = await db
          .select({ images: productsTable.images })
          .from(productsTable)
          .where(eq(productsTable.id, id))
          .limit(1);
        if (!product?.images) { res.writeHead(404); res.end(); return; }
        const imgs    = JSON.parse(product.images) as string[];
        const dataUrl = imgs[index] ?? imgs[0];
        if (!dataUrl) { res.writeHead(404); res.end(); return; }
        if (dataUrl.startsWith("data:")) {
          const [, b64] = dataUrl.split(",") as [string, string];
          const raw = Buffer.from(b64, "base64");
          let buf: Buffer;
          try { buf = await sharp(raw).jpeg({ quality: 85 }).toBuffer(); }
          catch { buf = raw; }
          res.writeHead(200, {
            "Content-Type":  "image/jpeg",
            "Cache-Control": "public, max-age=86400",
          });
          res.end(buf);
        } else {
          res.writeHead(302, { Location: dataUrl });
          res.end();
        }
      } catch (err) {
        console.error("[image-server] product image error:", err);
        res.writeHead(500); res.end();
      }
      return;
    }

    const broadcastMatch = urlPath.match(/^\/api\/broadcasts\/image\/(\d+)$/);
    if (broadcastMatch && req.method === "GET") {
      try {
        const id = parseInt(broadcastMatch[1]!, 10);
        const [broadcast] = await db
          .select({ imageUrl: broadcastsTable.imageUrl })
          .from(broadcastsTable)
          .where(eq(broadcastsTable.id, id))
          .limit(1);
        if (!broadcast?.imageUrl) { res.writeHead(404); res.end(); return; }
        const dataUrl = broadcast.imageUrl;
        if (dataUrl.startsWith("data:")) {
          const [, b64] = dataUrl.split(",") as [string, string];
          const raw = Buffer.from(b64, "base64");
          let buf: Buffer;
          try { buf = await sharp(raw).jpeg({ quality: 85 }).toBuffer(); }
          catch { buf = raw; }
          res.writeHead(200, {
            "Content-Type":  "image/jpeg",
            "Cache-Control": "public, max-age=86400",
          });
          res.end(buf);
        } else {
          res.writeHead(302, { Location: dataUrl });
          res.end();
        }
      } catch (err) {
        console.error("[image-server] broadcast image error:", err);
        res.writeHead(500); res.end();
      }
      return;
    }

    app(req, res);
  });

  server.listen(port, () => {
    console.log(`[server] Listening on port ${port}`);
  });
}

start();
