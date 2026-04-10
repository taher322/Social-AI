import { Router, type IRouter } from "express";
import {
  db, productsTable, conversationsTable, ordersTable,
  leadsTable, orderSessionsTable, productInquiriesTable,
  userProductContextTable, userCountersTable, faqsTable,
  processedMessagesTable,
} from "@workspace/db";
import { eq, desc, and, sql, count } from "drizzle-orm";
import { cache, TTL } from "../lib/cache.js";
import { rGet, rSet } from "../lib/redisCache.js";
import { broadcastNotification } from "./notifications.js";
import { ALGERIA_WILAYAS } from "./deliveryPrices.js";

import {
  buildSystemPrompt, buildCommentSystemPrompt,
  detectJailbreak, detectSalesTrigger, detectBookingIntent,
  getFreshAppointmentBlock, classifyShoppingIntent,
  type ShoppingContext, type SalesTriggerType,
  parseOrderAction, parseStartOrderAction, parseConfirmOrderAction,
  sendFbMessage,
  getFbUserName, isWithinBusinessHours,
  analyzeAttachmentWithGemini, matchProductsFromAnalysis,
} from "../lib/ai.js";

import {
  checkTextRateLimit, RESCUE_KEYWORDS,
  logPlatformEvent, verifyWebhookSignature, analyzeSentiment,
  extractPhone, isValidPhoneNumber, extractEmail,
  resolveWilaya, buildProductImageUrl,
  parseSaveLeadAction, parseCheckOrderStatusAction,
  checkWebhookRequestRate, isStaleWebhookEvent,
} from "../lib/webhookUtils.js";

import { getSettings, getConfig, isUserPaused, saveConversation } from "../lib/dbHelpers.js";

import {
  handlePreOrderSession,
  handleDeliverySession,
  handleConfirmSession,
  handleOrderMidFlow,
  type MsgCtx,
} from "../lib/orderInterceptors.js";

import { handlePageComment } from "../lib/commentHandler.js";

import {
  sendFbQuickReplies, BUFFER_SKIP, bufferMessage, getOrCreateSession,
} from "../lib/messengerUtils.js";

import {
  sendDeliveryOptions, sendCatalogCategoryMenu,
} from "../lib/catalogFlow.js";

import { handleProductPayload } from "../lib/orderFlow.js";

import {
  handleCheckOrderStatus, handleBrowseCatalog, handleSendImage,
  handleAppointment, handleStartOrder, handleConfirmOrder, handleCreateOrder,
  type ActionCtx,
} from "../lib/webhookActions.js";

import { handleAttachment } from "../lib/webhookAttachment.js";
import { handleAiCall, type AiCallParams } from "../lib/webhookAiCall.js";
import { isGreeting } from "../lib/greetings.js";
import { verifyReplyPrices } from "../lib/priceVerification.js";
import {
  buildCacheKey, isCacheable, isResponseStorable,
  getCachedReply, storeCachedReply,
} from "../lib/exactMatchCache.js";

const router: IRouter = Router();

// ── GET /webhook — Facebook verification ──────────────────────────────────────
router.get("/webhook", async (req, res): Promise<void> => {
  const settings = await getSettings();
  const mode      = req.query["hub.mode"];
  const token     = req.query["hub.verify_token"];
  const challenge = req.query["hub.challenge"];

  if (mode === "subscribe" && token === (settings?.verifyToken ?? "")) {
    res.status(200).send(challenge);
  } else {
    res.status(403).json({ message: "Forbidden" });
  }
});

// ── POST /webhook — main event handler ────────────────────────────────────────
router.post("/webhook", async (req, res): Promise<void> => {
  // ── Layer 1: IP rate limiting (120 req/min per IP) ──────────────────────────
  const clientIp =
    (req.headers["x-forwarded-for"] as string | undefined)?.split(",")[0]?.trim()
    ?? req.socket.remoteAddress
    ?? "unknown";
  if (!await checkWebhookRequestRate(clientIp)) {
    res.status(429).json({ message: "Too Many Requests" });
    return;
  }

  // ── Layer 2: Signature verification ─────────────────────────────────────────
  const settings = await getSettings();

  if (settings?.appSecret) {
    const sig     = req.headers["x-hub-signature-256"] as string | undefined;
    const rawBody = req.rawBody;
    if (!verifyWebhookSignature(rawBody, sig, settings.appSecret)) {
      res.status(403).json({ message: "Invalid signature" });
      return;
    }
  }

  res.json({ message: "EVENT_RECEIVED" });

  const body = req.body as {
    object?: string;
    entry?: Array<{
      id?: string;
      messaging?: Array<{
        sender?: { id: string };
        message?: {
          text?: string;
          mid?: string;
          is_echo?: boolean;
          quick_reply?: { payload?: string };
          attachments?: Array<{ type: string; payload: { url?: string } }>;
        };
        postback?: { payload?: string; title?: string };
        timestamp?: number;
      }>;
      changes?: Array<{
        field?: string;
        value?: {
          item?: string;
          verb?: string;
          comment_id?: string;
          post_id?: string;
          parent_id?: string;
          from?: { id: string; name?: string };
          message?: string;
          sender_id?: string;
        };
      }>;
    }>;
  };

  if (body.object !== "page") return;
  if (!settings?.pageAccessToken) return;

  const config = await getConfig();
  if (!config) return;

  for (const entry of body.entry ?? []) {
    for (const event of entry.messaging ?? []) {
      if (event.message?.is_echo) continue;

      // ── Layer 3: Replay attack protection — skip events older than 10 min ───
      if (isStaleWebhookEvent(event.timestamp)) {
        console.warn(`[webhook] Stale event (${Math.round((Date.now() - (event.timestamp ?? 0)) / 1000)}s old) from ${event.sender?.id ?? "?"} — skipped`);
        continue;
      }

      try {
      let fromAttachment = false;

      // ── Postback event handling ──────────────────────────────────────────────
      if (event.postback && event.sender?.id) {
        const pbSenderId = event.sender.id;
        const { name: pbUserName } = await getFbUserName(settings.pageAccessToken, pbSenderId);

        if (await isUserPaused(pbSenderId)) {
          const handoffMsg = "🤝 يتولى أحد ممثلينا محادثتك حالياً. سيرد عليك قريباً.";
          await sendFbMessage(settings.pageAccessToken, pbSenderId, handoffMsg, settings.pageId ?? undefined);
          continue;
        }
        if (!config.botEnabled) {
          const disabledMsg = config.botDisabledMessage ?? "عذراً، المساعد الذكي غير متاح حالياً. يرجى التواصل معنا لاحقاً.";
          await sendFbMessage(settings.pageAccessToken, pbSenderId, disabledMsg, settings.pageId ?? undefined);
          continue;
        }
        await handleProductPayload(event.postback.payload ?? "", pbSenderId, pbUserName, settings.pageAccessToken, settings.pageId ?? undefined);
        continue;
      }

      // ── Quick reply handling ─────────────────────────────────────────────────
      if (event.message?.quick_reply?.payload && event.sender?.id) {
        const qrSenderId = event.sender.id;
        const { name: qrUserName } = await getFbUserName(settings.pageAccessToken, qrSenderId);

        if (await isUserPaused(qrSenderId)) {
          const handoffMsg = "🤝 يتولى أحد ممثلينا محادثتك حالياً. سيرد عليك قريباً.";
          await sendFbMessage(settings.pageAccessToken, qrSenderId, handoffMsg, settings.pageId ?? undefined);
          continue;
        }
        if (!config.botEnabled) {
          const disabledMsg = config.botDisabledMessage ?? "عذراً، المساعد الذكي غير متاح حالياً. يرجى التواصل معنا لاحقاً.";
          await sendFbMessage(settings.pageAccessToken, qrSenderId, disabledMsg, settings.pageId ?? undefined);
          continue;
        }
        await handleProductPayload(event.message.quick_reply.payload, qrSenderId, qrUserName, settings.pageAccessToken, settings.pageId ?? undefined);
        continue;
      }

      // ── Phase 7B: Attachment-only messages (extracted to webhookAttachment.ts) ─
      if (!event.message?.text && event.sender?.id) {
        const attResult = await handleAttachment(
          event.message,
          event.sender.id,
          settings as Parameters<typeof handleAttachment>[2],
          config,
        );
        if (attResult.handled === "skip") continue;
        if (attResult.handled === true && event.message) {
          event.message.text = attResult.effectiveText;
          fromAttachment = true;
        }
      }

      if (!event.message?.text || !event.sender?.id) continue;
      const senderId       = event.sender.id;
      const rawMessageText = event.message.text;

      // ── Idempotency guard ─────────────────────────────────────────────────
      // Facebook occasionally re-delivers the same message (network retry).
      // We claim the mid by inserting it — if the insert fails (unique conflict)
      // the message was already processed, so we skip silently.
      const mid = event.message.mid;
      if (mid) {
        try {
          await db.insert(processedMessagesTable).values({ mid, senderId });
        } catch {
          console.log(`[idempotency] Duplicate mid=${mid.substring(0, 40)} senderId=${senderId} — skipped`);
          continue;
        }
      }

      // ── Text-message rate limit — 30 msg / 60 s per sender ─────────────────
      if (!await checkTextRateLimit(senderId)) {
        void logPlatformEvent("text_rate_limited", senderId, "30 msg/min exceeded");
        console.warn(`[rate-limit] Sender ${senderId} exceeded 30 msg/min — message dropped silently`);
        continue;
      }

      const _imageAttachment = (event.message?.attachments ?? []).find((a) => a.type === "image");
      const _audioAttachment = (event.message?.attachments ?? []).find((a) => a.type === "audio");

      const { name: userName, profileUrl } = await getFbUserName(settings.pageAccessToken, senderId);

      // ── Kill switch ──────────────────────────────────────────────────────────
      if (!config.botEnabled) {
        const disabledMsg = config.botDisabledMessage ?? "عذراً، المساعد الذكي غير متاح حالياً. يرجى التواصل معنا لاحقاً.";
        await sendFbMessage(settings.pageAccessToken, senderId, disabledMsg, settings.pageId ?? undefined);
        await db.insert(conversationsTable).values({
          fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
          message: rawMessageText, sender: "user", timestamp: new Date(),
        });
        await db.insert(conversationsTable).values({
          fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
          message: disabledMsg, sender: "bot", timestamp: new Date(),
        });
        void logPlatformEvent("kill_switch_blocked", senderId, rawMessageText.substring(0, 120));
        continue;
      }

      const paused = await isUserPaused(senderId);

      if (!fromAttachment) {
        broadcastNotification({
          type: "new_message",
          title: `رسالة جديدة من ${userName}`,
          body: rawMessageText.length > 80 ? rawMessageText.substring(0, 80) + "…" : rawMessageText,
          route: "/conversations",
        });
      }

      // Always update known lead on any message
      const [existingLeadForTracking] = await db.select()
        .from(leadsTable).where(eq(leadsTable.fbUserId, senderId)).limit(1);
      if (existingLeadForTracking) {
        await db.update(leadsTable).set({
          lastInteractionAt: new Date().toISOString(),
          totalMessages: (existingLeadForTracking.totalMessages ?? 0) + 1,
          updatedAt: new Date(),
        }).where(eq(leadsTable.fbUserId, senderId));
      }

      // ── Lead capture ─────────────────────────────────────────────────────────
      if (config.leadCaptureEnabled) {
        const _lcFields    = config.leadCaptureFields ?? "phone";
        const detectedPhone = _lcFields.includes("phone") ? extractPhone(rawMessageText) : null;
        const detectedEmail = _lcFields.includes("email") ? extractEmail(rawMessageText) : null;
        if (detectedPhone || detectedEmail) {
          if (existingLeadForTracking) {
            await db.update(leadsTable).set({
              phone: detectedPhone ?? existingLeadForTracking.phone,
              email: detectedEmail ?? existingLeadForTracking.email,
              updatedAt: new Date(),
            }).where(eq(leadsTable.fbUserId, senderId));
          } else {
            await db.insert(leadsTable).values({
              fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
              phone: detectedPhone ?? null, email: detectedEmail ?? null,
              label: "new", source: "messenger",
              lastInteractionAt: new Date().toISOString(), totalMessages: 1,
            }).onConflictDoNothing();
          }
        }
      }

      // ── Handoff keyword ──────────────────────────────────────────────────────
      if (config.handoffKeyword && rawMessageText.trim().toLowerCase() === config.handoffKeyword.toLowerCase()) {
        await db.update(conversationsTable).set({ isPaused: 1 }).where(eq(conversationsTable.fbUserId, senderId));
        const handoffMsg = config.handoffMessage ?? "تم تحويلك إلى فريق الدعم البشري. سيتواصل معك أحد ممثلينا قريباً.";
        await sendFbMessage(settings.pageAccessToken, senderId, handoffMsg, settings.pageId ?? undefined);
        await db.insert(conversationsTable).values({
          fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
          message: handoffMsg, sender: "bot", isPaused: 1, timestamp: new Date(),
        });
        continue;
      }

      if (paused) {
        // Save user message even during handoff so admin can see it in the dashboard
        await db.insert(conversationsTable).values({
          fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
          message: rawMessageText, sender: "user", isPaused: 1, timestamp: new Date(),
        });
        continue;
      }

      // ── Text shortcuts for confirm/cancel ────────────────────────────────────
      const msgLower = rawMessageText.trim().toLowerCase();
      if (msgLower === "تأكيد" || msgLower === "confirm" || msgLower === "نعم" || msgLower === "اكيد") {
        const [pendingSession] = await db.select().from(orderSessionsTable)
          .where(and(eq(orderSessionsTable.fbUserId, senderId), eq(orderSessionsTable.step, "awaiting_confirm")))
          .limit(1);
        if (pendingSession) {
          await handleProductPayload("CONFIRM_ORDER", senderId, userName, settings.pageAccessToken, settings.pageId ?? undefined);
          continue;
        }
      }
      if (msgLower === "إلغاء" || msgLower === "الغاء" || msgLower === "cancel" || msgLower === "لا") {
        const [pendingSession] = await db.select().from(orderSessionsTable)
          .where(and(eq(orderSessionsTable.fbUserId, senderId), eq(orderSessionsTable.step, "awaiting_confirm")))
          .limit(1);
        if (pendingSession) {
          await handleProductPayload("CANCEL_ORDER", senderId, userName, settings.pageAccessToken, settings.pageId ?? undefined);
          continue;
        }
      }

      // ── Order session interceptors ────────────────────────────────────────────
      {
        const _msgCtx: MsgCtx = {
          senderId, userName, profileUrl,
          rawMessageText,
          pageAccessToken: settings.pageAccessToken,
          pageId: settings.pageId ?? undefined,
          config,
        };
        if (await handlePreOrderSession(_msgCtx))  continue;
        if (await handleDeliverySession(_msgCtx))  continue;
        if (await handleConfirmSession(_msgCtx))   continue;
        if (await handleOrderMidFlow(_msgCtx))     continue;
      }

      // ── Working hours check ──────────────────────────────────────────────────
      if (config.workingHoursEnabled !== 0 && !isWithinBusinessHours(config.businessHoursStart, config.businessHoursEnd, config.timezone ?? "Africa/Algiers")) {
        const outsideMsg = config.outsideHoursMessage ?? "مرحباً! نحن حالياً خارج ساعات العمل. يرجى التواصل معنا خلال ساعات العمل.";
        await sendFbMessage(settings.pageAccessToken, senderId, outsideMsg, settings.pageId ?? undefined);
        await db.insert(conversationsTable).values({
          fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
          message: outsideMsg, sender: "bot", timestamp: new Date(),
        });
        continue;
      }

      // ── Catalog browser — text intent detection (pre-AI) ─────────────────────
      {
        const CATALOG_INTENT_PATTERNS = [
          /\b(catalog|catalogue|كتالوج|كتالوغ)\b/i,
          /\b(products|منتجات|المنتجات|عروض)\b/i,
          /^(أرني|ارني|show me|voir)\s*(المنتجات|كل شيء|everything|tout)/i,
          /\b(phones?|هواتف|telephone|تليفون)\b/i,
          /\b(courses?|كورسات?|دورات?|تدريب)\b/i,
          /\b(fashion|أزياء|ملابس|موضة)\b/i,
          /\b(electronics|إلكترونيات|الكترونيات)\b/i,
          /\b(تصفح|browse|parcourir)\b/i,
          /^(أرني|ارني|show me|voir les?)\s+\w+/i,
        ];
        const isCatalogIntent = CATALOG_INTENT_PATTERNS.some((p) => p.test(rawMessageText));
        if (isCatalogIntent) {
          const [activeOrderSession] = await db.select().from(orderSessionsTable)
            .where(eq(orderSessionsTable.fbUserId, senderId)).limit(1);
          if (!activeOrderSession) {
            await db.insert(conversationsTable).values({
              fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
              message: rawMessageText, sender: "user", timestamp: new Date(),
            });
            await sendCatalogCategoryMenu(settings.pageAccessToken, senderId, settings.pageId ?? undefined);
            cache.set(`catalog_shown:${senderId}`, true, 90 * 1000);
            await db.insert(conversationsTable).values({
              fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
              message: "🛍️ اختر الفئة التي تريد تصفحها:", sender: "bot", timestamp: new Date(),
            });
            void logPlatformEvent("catalog_browse_started", senderId, rawMessageText.substring(0, 80));
            continue;
          }
        }
      }

      // ── Message buffer (debounce rapid messages) ──────────────────────────────
      // Voice/image transcriptions skip the buffer — Gemini already took ~3s so
      // the debounce window is meaningless and would cause BUFFER_SKIP drops.
      const messageText = fromAttachment ? rawMessageText : await bufferMessage(senderId, rawMessageText);
      if (messageText === BUFFER_SKIP) continue;

      const sentiment    = analyzeSentiment(messageText);
      const salesTrigger: SalesTriggerType = detectSalesTrigger(messageText);
      if (salesTrigger) console.log(`[sales-trigger] Detected "${salesTrigger}" for ${senderId}`);

      const [userMsgRow] = await db.insert(conversationsTable).values({
        fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
        message: messageText, sender: "user",
        isPaused: paused ? 1 : 0, sentiment, salesTriggerType: salesTrigger, timestamp: new Date(),
      }).returning({ id: conversationsTable.id });
      const lastUserMsgId = userMsgRow?.id ?? null;

      // ── Blocked keywords ─────────────────────────────────────────────────────
      if (config.blockedKeywords) {
        const keywords = config.blockedKeywords.split(",").map((k) => k.trim().toLowerCase()).filter(Boolean);
        const msgLowerForBlock = messageText.toLowerCase();
        const matchedKeyword   = keywords.find((kw) => msgLowerForBlock.includes(kw));
        if (matchedKeyword) {
          const blockReply = config.offTopicResponse ?? "عذراً، لا يمكنني الإجابة على هذا الموضوع.";
          await sendFbMessage(settings.pageAccessToken, senderId, blockReply, settings.pageId ?? undefined);
          await db.insert(conversationsTable).values({
            fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
            message: blockReply, sender: "bot", timestamp: new Date(),
          });
          void logPlatformEvent("blocked_keyword", senderId, `keyword="${matchedKeyword}" msg="${messageText.substring(0, 80)}"`);
          console.log(`[blocked-keyword] Blocked message from ${senderId}: matched "${matchedKeyword}"`);
          continue;
        }
      }

      // ── Conversation Rescue ──────────────────────────────────────────────────
      {
        const msgLowerForRescue    = messageText.toLowerCase();
        const hasFrustrationKeyword = RESCUE_KEYWORDS.some((kw) => msgLowerForRescue.includes(kw.toLowerCase()));
        let recentNegativeCount     = 0;
        if (!hasFrustrationKeyword) {
          const recentMsgs = await db.select({ sentiment: conversationsTable.sentiment })
            .from(conversationsTable)
            .where(and(eq(conversationsTable.fbUserId, senderId), eq(conversationsTable.sender, "bot")))
            .orderBy(desc(conversationsTable.timestamp)).limit(5);
          recentNegativeCount = recentMsgs.filter((m) => m.sentiment === "negative").length;
        }
        if (hasFrustrationKeyword || recentNegativeCount >= 2) {
          const alreadyPaused = await isUserPaused(senderId);
          if (!alreadyPaused) {
            await db.update(conversationsTable).set({ isPaused: 1 }).where(eq(conversationsTable.fbUserId, senderId));
            const handoffMsg = config.handoffMessage ?? "تم تحويلك إلى فريق الدعم البشري. سيتواصل معك أحد ممثلينا قريباً.";
            await sendFbMessage(settings.pageAccessToken, senderId, handoffMsg, settings.pageId ?? undefined);
            await db.insert(conversationsTable).values({
              fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
              message: handoffMsg, sender: "bot", isPaused: 1, rescueTriggered: 1, timestamp: new Date(),
            });
            const rescueReason = hasFrustrationKeyword ? "frustration keyword detected" : `${recentNegativeCount} negative sentiments in last 5 replies`;
            void logPlatformEvent("rescue_triggered", senderId, rescueReason);
            void logPlatformEvent("lost_risk_prevented", senderId, `reason=rescue ${rescueReason}`);
            console.log(`[rescue] Triggered for ${senderId}: ${rescueReason}`);
            continue;
          }
        }
      }

      // ── Smart Escalation (hesitation) ────────────────────────────────────────
      if (salesTrigger === "hesitation" && config.smartEscalationEnabled && !paused) {
        const alreadyPaused = await isUserPaused(senderId);
        if (!alreadyPaused) {
          await db.update(conversationsTable).set({ isPaused: 1 }).where(eq(conversationsTable.fbUserId, senderId));
          const handoffMsg = config.handoffMessage ?? "تم تحويلك إلى فريق الدعم البشري. سيتواصل معك أحد ممثلينا قريباً.";
          await sendFbMessage(settings.pageAccessToken, senderId, handoffMsg, settings.pageId ?? undefined);
          await db.insert(conversationsTable).values({
            fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
            message: handoffMsg, sender: "bot", isPaused: 1, timestamp: new Date(),
          });
          void logPlatformEvent("lost_risk_prevented", senderId, "reason=hesitation_smart_escalation");
          console.log(`[smart-escalation] Hesitation detected for ${senderId} — transferred to human`);
          continue;
        }
      }

      // ── Load history + products + FAQs ───────────────────────────────────────
      const history = await db.select().from(conversationsTable)
        .where(eq(conversationsTable.fbUserId, senderId))
        .orderBy(desc(conversationsTable.timestamp)).limit(10);

      const isFirstMessage = history.filter((h) => h.sender === "bot").length === 0;
      const messages = history.reverse().map((m) => ({
        role: m.sender === "user" ? "user" as const : "assistant" as const,
        content: m.message,
      }));

      const { isNew: isNewSession } = await getOrCreateSession(senderId);

      // ── Safe Mode — jailbreak detection ──────────────────────────────────────
      if (config.safeModeEnabled) {
        if (detectJailbreak(messageText)) {
          const safeReply = "عذراً، لا يمكنني الاستجابة لهذا النوع من الطلبات. يمكنني مساعدتك في أسئلة تتعلق بمنتجاتنا وخدماتنا فقط.";
          await sendFbMessage(settings.pageAccessToken, senderId, safeReply, settings.pageId ?? undefined);
          await db.insert(conversationsTable).values({
            fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
            message: safeReply, sender: "bot", safeModeBlocked: 1, sourceType: "safe_mode_blocked", timestamp: new Date(),
          });
          void logPlatformEvent("safe_mode_blocked", senderId, messageText.substring(0, 120));
          console.log(`[safe-mode] Jailbreak blocked for ${senderId}`);
          continue;
        }
      }

      // ── Greeting shortcut — رد ثابت على التحيات بدون استدعاء AI ────────────
      // يُفعَّل إذا: (1) greetingMessage مُعرَّف في الإعدادات + (2) الرسالة تحية نقية
      // التأثير: يتجاوز تحميل المنتجات/FAQs وبناء الـ prompt واستدعاء AI كاملاً
      if (config.greetingMessage && isGreeting(messageText)) {
        const greetReply = config.greetingMessage;
        await sendFbMessage(settings.pageAccessToken, senderId, greetReply, settings.pageId ?? undefined);
        await db.insert(conversationsTable).values({
          fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
          message: greetReply, sender: "bot", sourceType: "greeting_shortcut", timestamp: new Date(),
        });
        void logPlatformEvent("greeting_shortcut", senderId, messageText.substring(0, 80));
        console.log(`[greeting] Shortcut reply to "${messageText.substring(0, 30)}" for ${senderId}`);
        continue;
      }

      // ── Fetch products + FAQs (cached) ───────────────────────────────────────
      const [allProducts, preFetchedFaqs] = await Promise.all([
        rGet<typeof productsTable.$inferSelect[]>("products:available").then((cached) =>
          cached
            ? cached
            : db.select().from(productsTable).where(eq(productsTable.status, "available")).then(async (rows) => {
                await rSet("products:available", rows, TTL.PRODUCTS);
                return rows;
              })
        ),
        rGet<typeof faqsTable.$inferSelect[]>("faqs:active").then((cached) =>
          cached
            ? cached
            : db.select().from(faqsTable).where(eq(faqsTable.isActive, 1)).then(async (rows) => {
                await rSet("faqs:active", rows, TTL.FAQS);
                return rows;
              })
        ),
      ]);

      // ── Price Lock ───────────────────────────────────────────────────────────
      if (config.priceLockEnabled) {
        const priceTriggers    = ["سعر", "بشحال", "بكم", "ثمن", "كم سعر", "price", "prix", "cost", "tarif", "combien"];
        const msgLowerForPrice = messageText.toLowerCase();
        if (priceTriggers.some((kw) => msgLowerForPrice.includes(kw))) {
          const matchedProduct = allProducts.find((p) => msgLowerForPrice.includes(p.name.toLowerCase()));
          let priceReply: string;
          if (matchedProduct) {
            const price = matchedProduct.discountPrice ?? matchedProduct.originalPrice;
            priceReply = price
              ? `💰 سعر ${matchedProduct.name}: **${price} ${config.currency ?? "DZD"}**\n\nهل تريد إتمام الطلب الآن؟ 🛒`
              : `✅ سعر ${matchedProduct.name} متاح عند التواصل. هل تريد إتمام طلب؟`;
          } else {
            priceReply = "يسعدنا إعلامك بالسعر! هل يمكنك تحديد المنتج الذي تسأل عنه؟ 😊";
          }
          await sendFbMessage(settings.pageAccessToken, senderId, priceReply, settings.pageId ?? undefined);
          await db.insert(conversationsTable).values({
            fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
            message: priceReply, sender: "bot", sourceType: "price_lock",
            salesTriggerType: salesTrigger, timestamp: new Date(),
          });
          void logPlatformEvent("price_lock_triggered", senderId, `product=${matchedProduct?.name ?? "unknown"}`);
          console.log(`[price-lock] Intercepted price query for ${senderId}`);
          continue;
        }
      }

      // ── Active product context ────────────────────────────────────────────────
      let activeProduct: typeof productsTable.$inferSelect | undefined;
      {
        const TTL_MS = 30 * 60 * 1000;
        const [ctx] = await db.select().from(userProductContextTable)
          .where(eq(userProductContextTable.fbUserId, senderId)).limit(1);
        if (ctx) {
          const ageMs = Date.now() - new Date(ctx.updatedAt).getTime();
          if (ageMs <= TTL_MS) {
            const [ap] = await db.select().from(productsTable)
              .where(eq(productsTable.id, ctx.productId)).limit(1);
            if (ap && ap.status === "available") activeProduct = ap;
          }
        }
      }

      // ── Text + Image enrichment ───────────────────────────────────────────────
      if (_imageAttachment?.payload.url && !activeProduct) {
        try {
          const imgAnalysis = await analyzeAttachmentWithGemini(
            _imageAttachment.payload.url, "image", rawMessageText, settings.pageAccessToken
          );
          if (imgAnalysis && imgAnalysis.confidence >= 0.5) {
            const { matches, tier } = matchProductsFromAnalysis(imgAnalysis, allProducts);
            if ((tier === "strong" || tier === "multiple") && matches[0]) {
              activeProduct = matches[0];
              await db.insert(userProductContextTable)
                .values({ fbUserId: senderId, productId: matches[0].id, updatedAt: new Date() })
                .onConflictDoUpdate({
                  target: userProductContextTable.fbUserId,
                  set: { productId: matches[0].id, updatedAt: new Date() },
                });
              void logPlatformEvent("multimodal_text_image_enrich", senderId, `product=${matches[0].name} confidence=${imgAnalysis.confidence}`);
            }
          }
        } catch (enrichErr) {
          console.error("[multimodal] Text+image enrichment failed:", (enrichErr as Error).message);
        }
      }

      // ── Multi-Step Shopping Flow ──────────────────────────────────────────────
      const availableInStock = allProducts.filter((p) => p.status === "available" && p.stockQuantity > 0);
      const availableCategories = [...new Set(availableInStock.map((p: any) => p.category as string | null).filter((c): c is string => Boolean(c)))] as string[];
      // Extract clean brand/top-level names (e.g. "Vivo/12000 الي 24000" → "Vivo")
      const availableCategoryLabels = [...new Set(availableCategories.map((c) => {
        const slash = c.indexOf("/");
        return (slash > 0 ? c.substring(0, slash) : c).trim();
      }))].filter(Boolean);

      // Helper: match a product's category against a target using path-prefix logic
      // so "Vivo" matches "Vivo/12000 الي 24000" etc.
      const matchesCategoryTarget = (productCat: string | null | undefined, target: string): boolean => {
        if (!productCat) return false;
        const cat = productCat.toLowerCase();
        const t   = target.toLowerCase().trim();
        return cat === t || cat.startsWith(t + "/");
      };
      const matchesCatOrBrand = (p: typeof availableInStock[0], target: string): boolean => {
        const t = target.toLowerCase().trim();
        return (
          matchesCategoryTarget(p.category, target) ||
          (p.brand ?? "").toLowerCase().trim() === t ||
          (p.name ?? "").toLowerCase().includes(t)
        );
      };

      let filteredProducts = availableInStock.slice(0, 30);
      let shoppingInstruction = "";

      // ── Exact Match Cache — Lookup ──────────────────────────────────────────
      // يعمل قبل تصنيف Shopping Flow والـ AI لتوفير الوقت والتكلفة.
      // يتحقق من السياق السابق للمستخدم دون استدعاء AI إضافي.
      const _prevShopCtx   = await rGet<ShoppingContext>(`shopctx:${senderId}`);
      const _exactCacheKey = buildCacheKey(messageText, availableInStock);
      const _cacheable     = isCacheable(messageText, !!activeProduct, !!_prevShopCtx?.activeCategory);

      if (_cacheable) {
        const _cachedReply = await getCachedReply(_exactCacheKey);
        if (_cachedReply) {
          void logPlatformEvent("cache_hit", senderId, `"${messageText.substring(0, 60)}"`);
          console.log(`[exact-cache] HIT for ${senderId} — skipping AI call`);
          await sendFbMessage(settings.pageAccessToken, senderId, _cachedReply, settings.pageId ?? undefined);
          await db.insert(conversationsTable).values({
            fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
            message: _cachedReply, sender: "bot", sourceType: "exact_cache_hit", timestamp: new Date(),
          });
          continue;
        }
      }
      // ── End Exact Match Cache Lookup ────────────────────────────────────────

      if (availableCategories.length > 0) {
        const shopCacheKey    = `shopctx:${senderId}`;
        const currentContext  = (await rGet<ShoppingContext>(shopCacheKey)) ?? null;
        const currentCatProducts = currentContext?.activeCategory
          ? availableInStock.filter((p) => matchesCatOrBrand(p, currentContext.activeCategory!))
          : [];
        const availableBrandsOrTypes = [...new Set([
          ...currentCatProducts.map((p) => p.brand).filter(Boolean),
          ...currentCatProducts.map((p) => p.itemType).filter(Boolean),
        ])] as string[];

        const catPrices = currentCatProducts.map((p) => p.discountPrice ?? p.originalPrice ?? 0).filter((v) => v > 0).sort((a, b) => a - b);
        let priceTiersDescription = "No price tier information available for current category.";
        let p33 = 0, p66 = 0;
        if (catPrices.length >= 3) {
          p33 = catPrices[Math.floor(catPrices.length * 0.33)] ?? 0;
          p66 = catPrices[Math.floor(catPrices.length * 0.66)] ?? 0;
          const budgetCount  = currentCatProducts.filter((p) => (p.discountPrice ?? p.originalPrice ?? 0) <= p33).length;
          const midCount     = currentCatProducts.filter((p) => { const pr = p.discountPrice ?? p.originalPrice ?? 0; return pr > p33 && pr <= p66; }).length;
          const premiumCount = currentCatProducts.filter((p) => (p.discountPrice ?? p.originalPrice ?? 0) > p66).length;
          const cur = config.currency ?? "";
          priceTiersDescription = `Price tiers for "${currentContext?.activeCategory ?? "current category"}": Budget (≤${p33} ${cur}): ${budgetCount} products | Mid (${p33}–${p66} ${cur}): ${midCount} products | Premium (>${p66} ${cur}): ${premiumCount} products`;
        }

        const recentMsgLines = history
          .slice(-4)
          .map((h) => `${h.sender === "user" ? "customer" : "bot"}: ${h.message.substring(0, 150)}`)
          .join("\n");

        const shopCtx = await classifyShoppingIntent(messageText, currentContext, availableCategoryLabels, availableBrandsOrTypes, priceTiersDescription, recentMsgLines);

        const contextToStore: ShoppingContext = shopCtx.contextAction === "DROP"
          ? { ...shopCtx, activeCategory: null, filterType: null, priceTier: null, keywords: [] }
          : shopCtx;

        const shopCtxTTL = (shopCtx.step !== "answer_question" && shopCtx.contextAction !== "DROP")
          ? 20 * 60 * 1000
          : 5 * 60 * 1000;

        await rSet(shopCacheKey, contextToStore, shopCtxTTL);

        const catProducts = shopCtx.activeCategory
          ? availableInStock.filter((p) => matchesCatOrBrand(p, shopCtx.activeCategory!))
          : availableInStock;

        switch (shopCtx.step) {
          case "show_categories": {
            // Anti-loop: if we already showed categories last turn, break out and list products directly
            if (currentContext?.step === "show_categories") {
              filteredProducts = availableInStock.slice(0, 20);
              shoppingInstruction = `\n\nSHOPPING FLOW — SHOW ALL PRODUCTS:\nCustomer has already seen the category menu. Present all available products now as friendly cards with key specs and prices. Guide them to pick one or ask a specific question.\n`;
              void logPlatformEvent("shopping_flow", senderId, `step=show_categories_loop_break cat=- filter=- tier=- kw=- sent=${filteredProducts.length}`);
              break;
            }
            await sendCatalogCategoryMenu(settings.pageAccessToken, senderId, settings.pageId ?? undefined);
            await saveConversation({ fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl, message: "🛍️ اختر الفئة التي تريد تصفحها:", sender: "bot" });
            void logPlatformEvent("shopping_flow", senderId, `step=show_categories cat=- filter=- tier=- kw=- direct`);
            continue;
          }
          case "show_filter_options": {
            filteredProducts = [];
            const hasBrands = availableBrandsOrTypes.length > 0;
            const opts = hasBrands
              ? `1️⃣ By type / brand (بحسب النوع أو الماركة)\n2️⃣ By price range (بحسب نطاق السعر)`
              : `1️⃣ By price range (بحسب نطاق السعر)`;
            shoppingInstruction = `\n\nSHOPPING FLOW — SHOW FILTER OPTIONS:\nCustomer chose category "${shopCtx.activeCategory}". Ask how they'd like to filter:\n${opts}\nBe friendly and concise. Wait for their choice.\n`;
            break;
          }
          case "show_price_tiers": {
            filteredProducts = [];
            shoppingInstruction = `\n\nSHOPPING FLOW — SHOW PRICE TIERS:\nCustomer wants to filter by price in category "${shopCtx.activeCategory}".\n${priceTiersDescription}\nPresent the three tiers clearly with counts. Invite them to choose one. Be friendly.\n`;
            break;
          }
          case "show_products": {
            if (shopCtx.filterType === "by_price" && shopCtx.priceTier && catPrices.length >= 3) {
              const allCatPrices = catProducts.map((p) => p.discountPrice ?? p.originalPrice ?? 0).filter((v) => v > 0).sort((a, b) => a - b);
              const lp33 = allCatPrices[Math.floor(allCatPrices.length * 0.33)] ?? 0;
              const lp66 = allCatPrices[Math.floor(allCatPrices.length * 0.66)] ?? 0;
              if (shopCtx.priceTier === "budget")     filteredProducts = catProducts.filter((p) => (p.discountPrice ?? p.originalPrice ?? 0) <= lp33).slice(0, 20);
              else if (shopCtx.priceTier === "mid")   filteredProducts = catProducts.filter((p) => { const pr = p.discountPrice ?? p.originalPrice ?? 0; return pr > lp33 && pr <= lp66; }).slice(0, 20);
              else                                     filteredProducts = catProducts.filter((p) => (p.discountPrice ?? p.originalPrice ?? 0) > lp66).slice(0, 20);
            } else if (shopCtx.keywords.length > 0) {
              const kws = shopCtx.keywords.map((k) => k.toLowerCase());
              const matchScore = (p: typeof catProducts[0]): number => {
                let score = 0;
                const pName  = p.name.toLowerCase();
                const pDesc  = (p.description ?? "").toLowerCase();
                const pBrand = (p.brand ?? "").toLowerCase();
                const pType  = (p.itemType ?? "").toLowerCase();
                for (const kw of kws) {
                  if (pName.includes(kw))  score += 3;   // اسم مباشر — أعلى أولوية
                  if (pBrand.includes(kw)) score += 2;   // ماركة
                  if (pType.includes(kw))  score += 2;   // نوع
                  if (pDesc.includes(kw))  score += 1;   // وصف — أقل أولوية
                }
                return score;
              };
              const scored  = catProducts.map((p) => ({ p, score: matchScore(p) })).filter((x) => x.score > 0);
              const matched = scored.sort((a, b) => b.score - a.score).map((x) => x.p);
              filteredProducts = matched.length > 0 ? matched.slice(0, 15) : catProducts.slice(0, 20);
            } else {
              filteredProducts = catProducts.slice(0, 20);
            }
            if (filteredProducts.length === 0) filteredProducts = availableInStock.slice(0, 20);
            shoppingInstruction = `\n\nSHOPPING FLOW — SHOW PRODUCTS:\nPresent the available products as cards. Be friendly, highlight key specs and pricing.\n`;
            break;
          }
          case "answer_question":
          default: {
            const useCategory = shopCtx.contextAction !== "DROP" && shopCtx.activeCategory;
            filteredProducts = useCategory ? catProducts.slice(0, 20) : availableInStock.slice(0, 30);
            break;
          }
        }

        void logPlatformEvent("shopping_flow", senderId,
          `step=${shopCtx.step} action=${shopCtx.contextAction} cat=${shopCtx.activeCategory ?? "-"} filter=${shopCtx.filterType ?? "-"} tier=${shopCtx.priceTier ?? "-"} kws=[${shopCtx.keywords.join(",")}] sent=${filteredProducts.length}`);
      } else {
        filteredProducts = availableInStock.slice(0, 30);
      }

      // ── Build system prompt ───────────────────────────────────────────────────
      // When shopping flow intentionally empties filteredProducts (show_categories, show_filter_options,
      // show_price_tiers) but products DO exist, pass availableInStock so the AI knows products are
      // available instead of reading "No products currently available." and hallucinating an empty store.
      const promptProducts = (filteredProducts.length === 0 && shoppingInstruction && availableInStock.length > 0)
        ? availableInStock.slice(0, 15)
        : filteredProducts;
      let systemPrompt = await buildSystemPrompt(config, promptProducts, { fbUserId: senderId, salesTrigger, activeProduct, preFetchedFaqs });
      if (shoppingInstruction) systemPrompt += shoppingInstruction;

      const appointmentsEnabled = Boolean(config.appointmentsEnabled);
      if (detectBookingIntent(messageText)) {
        if (appointmentsEnabled) {
          const freshBlock = await getFreshAppointmentBlock();
          if (freshBlock) systemPrompt += freshBlock;
        } else {
          systemPrompt += "\n\nAPPOINTMENTS DISABLED: If the customer asks about booking an appointment, respond politely that you do not currently accept appointment bookings. In Arabic say: 'عذراً، لا نقبل حجز المواعيد حالياً.'\n";
        }
      }

      // ── Call AI + parse response (extracted to webhookAiCall.ts) ─────────────
      const _aiCallParams: AiCallParams = {
        messages, systemPrompt,
        senderId, userName, profileUrl,
        settings: settings as AiCallParams["settings"],
        config, salesTrigger,
      };
      const aiCallResult = await handleAiCall(_aiCallParams);
      if (aiCallResult.outcome === "handled") continue;

      let { replyText } = aiCallResult;
      const { aiSentiment, aiConfidenceScore,
              replyProviderName, replyModelName, replySourceType } = aiCallResult;

      // ── Price Verification Layer — طبقة الحماية الثانية للأسعار ──────────────
      // تعمل على كل رد يذكر سعراً بجانب اسم منتج — لا تُفعَّل على الردود بدون أسعار
      const _priceCheck = verifyReplyPrices(replyText, availableInStock, activeProduct);
      if (!_priceCheck.safe) {
        console.warn(`[price-verify] ⚠️  ${_priceCheck.reason} — correcting reply for ${senderId}`);
        void logPlatformEvent("price_mismatch_blocked", senderId, _priceCheck.reason.substring(0, 120));
        replyText = _priceCheck.corrected;
      }

      // ── Exact Match Cache — Store ───────────────────────────────────────────
      // يُخزَّن بعد Price Verification لضمان حفظ الرد المصحَّح فقط.
      // الشروط: الرسالة مؤهلة + الرد لا يحتوي JSON أوامر + طول مناسب.
      if (_cacheable && isResponseStorable(replyText)) {
        void storeCachedReply(_exactCacheKey, replyText);
        console.log(`[exact-cache] STORED key=${_exactCacheKey.substring(0, 22)} len=${replyText.length}`);
      }
      // ── End Exact Match Cache Store ─────────────────────────────────────────

      // ── Off-topic counter (persisted to DB — survives server restarts) ──────────
      if (config.strictTopicMode && config.offTopicResponse) {
        const offTopicRef     = config.offTopicResponse.trim();
        const isOffTopicReply = replyText.trim() === offTopicRef || replyText.trim().startsWith(offTopicRef);
        if (isOffTopicReply) {
          const [uc] = await db.select({ offTopicCount: userCountersTable.offTopicCount })
            .from(userCountersTable).where(eq(userCountersTable.fbUserId, senderId)).limit(1);
          const newCount   = (uc?.offTopicCount ?? 0) + 1;
          const maxAllowed = config.maxOffTopicMessages ?? 3;
          await db.insert(userCountersTable)
            .values({ fbUserId: senderId, offTopicCount: newCount })
            .onConflictDoUpdate({ target: userCountersTable.fbUserId, set: { offTopicCount: newCount, updatedAt: new Date() } });
          if (newCount >= maxAllowed) {
            await db.insert(userCountersTable)
              .values({ fbUserId: senderId, offTopicCount: 0 })
              .onConflictDoUpdate({ target: userCountersTable.fbUserId, set: { offTopicCount: 0, updatedAt: new Date() } });
            await db.update(conversationsTable).set({ isPaused: 1 }).where(eq(conversationsTable.fbUserId, senderId));
            const handoffMsg = config.handoffMessage ?? "تم تحويلك إلى فريق الدعم البشري. سيتواصل معك أحد ممثلينا قريباً.";
            await sendFbMessage(settings.pageAccessToken, senderId, handoffMsg, settings.pageId ?? undefined);
            await db.insert(conversationsTable).values({
              fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
              message: handoffMsg, sender: "bot", isPaused: 1, timestamp: new Date(),
            });
            void logPlatformEvent("off_topic_escalation", senderId, `count=${newCount} max=${maxAllowed}`);
            void logPlatformEvent("handoff", senderId, `reason=off_topic_escalation`);
            console.log(`[off-topic] User ${senderId} exceeded maxOffTopicMessages (${maxAllowed}), triggering handoff`);
            continue;
          }
          console.log(`[off-topic] User ${senderId} off-topic count: ${newCount}/${maxAllowed}`);
        } else {
          // Reset counter when user is back on topic
          await db.insert(userCountersTable)
            .values({ fbUserId: senderId, offTopicCount: 0 })
            .onConflictDoUpdate({ target: userCountersTable.fbUserId, set: { offTopicCount: 0, updatedAt: new Date() } });
        }
      }

      // ── save_lead action ──────────────────────────────────────────────────────
      const saveLeadAction = parseSaveLeadAction(replyText);
      if (saveLeadAction) {
        const [existingLead] = await db.select().from(leadsTable).where(eq(leadsTable.fbUserId, senderId)).limit(1);
        if (existingLead) {
          await db.update(leadsTable).set({
            phone: saveLeadAction.phone ?? existingLead.phone,
            email: saveLeadAction.email ?? existingLead.email,
            notes: saveLeadAction.notes ?? existingLead.notes,
            lastInteractionAt: new Date().toISOString(),
            updatedAt: new Date(),
          }).where(eq(leadsTable.fbUserId, senderId));
        } else {
          await db.insert(leadsTable).values({
            fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
            phone: saveLeadAction.phone ?? null, email: saveLeadAction.email ?? null,
            notes: saveLeadAction.notes ?? null,
            label: "new", source: "messenger",
            lastInteractionAt: new Date().toISOString(), totalMessages: 1,
          }).onConflictDoNothing();
        }
        replyText = replyText.replace(/\{[\s\S]*?"action"\s*:\s*"save_lead"[\s\S]*?\}/, "").trim();
      }

      // ── AI action dispatch (extracted to webhookActions.ts) ───────────────────
      // settings.pageAccessToken is guaranteed non-null here (checked at entry)
      const actionCtx: ActionCtx = {
        senderId, userName, profileUrl, replyText,
        settings: settings as ActionCtx["settings"],
        config, allProducts, salesTrigger,
        replyProviderName, replyModelName, lastUserMsgId,
      };
      if (await handleCheckOrderStatus(actionCtx)) continue;
      if (await handleBrowseCatalog(actionCtx))    continue;
      if (await handleSendImage(actionCtx))        continue;
      if (await handleAppointment(actionCtx))      continue;
      if (await handleStartOrder(actionCtx))       continue;
      if (await handleConfirmOrder(actionCtx))     continue;
      if (await handleCreateOrder(actionCtx))      continue;

      // ── Human Guarantee + final reply ────────────────────────────────────────
      {
        if (config.humanGuaranteeEnabled) {
          replyText = replyText + "\n\n💬 إذا أردت التحدث مع شخص حقيقي، اكتب: \"بشري\"";
        }

        const replyTextLower  = replyText.toLowerCase();
        const mentionedProduct = config.useQuickReplies
          ? allProducts.find((p) => replyTextLower.includes(p.name.toLowerCase()))
          : undefined;

        if (isFirstMessage && config.useQuickReplies) {
          const DEFAULT_QR_BUTTONS = [
            { title: "📦 استفسار منتجات", payload: "PRODUCTS" },
            { title: "📅 حجز موعد",        payload: "APPOINTMENT" },
            { title: "🚚 خدمة التوصيل",    payload: "DELIVERY" },
          ];
          let qrButtons = DEFAULT_QR_BUTTONS;
          if (config.quickReplyButtons) {
            try {
              const parsed = JSON.parse(config.quickReplyButtons) as { title: string; payload: string }[];
              if (Array.isArray(parsed) && parsed.length > 0) qrButtons = parsed;
            } catch {}
          }
          try {
            await sendFbQuickReplies(settings.pageAccessToken, senderId, replyText, qrButtons.slice(0, 13), settings.pageId ?? undefined);
          } catch (e) {
            console.warn("[webhook] sendFbQuickReplies failed, falling back to plain message:", e instanceof Error ? e.message : String(e));
            await sendFbMessage(settings.pageAccessToken, senderId, replyText, settings.pageId ?? undefined);
          }
        } else if (mentionedProduct) {
          await sendFbMessage(settings.pageAccessToken, senderId, replyText, settings.pageId ?? undefined);
          try {
            await sendFbQuickReplies(
              settings.pageAccessToken, senderId,
              `🔷 ${mentionedProduct.name}`,
              [
                { title: "🛒 اطلب الآن",   payload: `ORDER_NOW:${mentionedProduct.id}` },
                { title: "💰 السعر",        payload: `PRICE_INFO:${mentionedProduct.id}` },
                { title: "📸 صورة المنتج", payload: `PRODUCT_IMAGE:${mentionedProduct.id}` },
              ],
              settings.pageId ?? undefined
            );
          } catch (e) {
            console.warn("[webhook] sendFbQuickReplies (product) failed:", e instanceof Error ? e.message : String(e));
          }
        } else {
          await sendFbMessage(settings.pageAccessToken, senderId, replyText, settings.pageId ?? undefined);
        }

        await db.insert(conversationsTable).values({
          fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
          message: replyText, sender: "bot", sentiment: aiSentiment,
          confidenceScore: aiConfidenceScore,
          providerName: replyProviderName || null, modelName: replyModelName || null,
          sourceType: replySourceType, salesTriggerType: salesTrigger, timestamp: new Date(),
        });

        // Abandoned cart tracking
        if (config.abandonedCartEnabled) {
          const replyLower  = replyText.toLowerCase();
          const userMsgLower = messageText.toLowerCase();
          const mentionedProd = allProducts.find((p) => {
            const nameLower = p.name.toLowerCase();
            return replyLower.includes(nameLower) || userMsgLower.includes(nameLower);
          });
          const hasOrderAction = !!parseStartOrderAction(replyText) || !!parseConfirmOrderAction(replyText) || !!parseOrderAction(replyText);
          if (mentionedProd && !hasOrderAction) {
            const now = new Date().toISOString();
            const [existing] = await db.select().from(productInquiriesTable)
              .where(and(eq(productInquiriesTable.fbUserId, senderId), eq(productInquiriesTable.productName, mentionedProd.name), eq(productInquiriesTable.converted, 0)))
              .limit(1);
            if (existing) {
              await db.update(productInquiriesTable)
                .set({ inquiredAt: now, reminderSent: 0 })
                .where(eq(productInquiriesTable.id, existing.id));
            } else {
              await db.insert(productInquiriesTable).values({
                fbUserId: senderId, fbUserName: userName,
                productName: mentionedProd.name, productId: mentionedProd.id,
                inquiredAt: now, createdAt: now,
              });
            }
          }
        }

        // Lead capture message (2nd interaction)
        if (config.leadCaptureEnabled && !isFirstMessage) {
          const [existingLead] = await db.select().from(leadsTable)
            .where(eq(leadsTable.fbUserId, senderId)).limit(1);
          const needsCapture = !existingLead || (!existingLead.phone && !existingLead.email);
          const msgCount     = history.filter((h) => h.sender === "user").length;
          if (needsCapture && msgCount === 2) {
            const captureMsg = config.leadCaptureMessage ?? "يسعدنا خدمتك! هل يمكنك مشاركتنا رقم هاتفك للتواصل؟";
            await sendFbMessage(settings.pageAccessToken, senderId, captureMsg, settings.pageId ?? undefined);
            await db.insert(conversationsTable).values({
              fbUserId: senderId, fbUserName: userName, fbProfileUrl: profileUrl,
              message: captureMsg, sender: "bot", timestamp: new Date(),
            });
          }
        }
      }
      } catch (msgErr) {
        console.error("❌ Unhandled error processing messaging event:", (msgErr as Error).message);
      }
    }

    // ── Comment handling ────────────────────────────────────────────────────────
    for (const change of entry.changes ?? []) {
      if (change.field !== "feed") continue;
      const val = change.value;
      if (val?.item !== "comment") continue;
      if (!config.botEnabled)      continue;
      if (!config.replyToComments) continue;

      // تجاهل تعليقات الصفحة نفسها — هذا يمنع chain reaction حيث البوت يرد على ردوده
      const fromId  = val.from?.id ?? val.sender_id ?? "";
      const ownPageId = settings.pageId ?? entry.id ?? "";
      if (ownPageId && fromId === ownPageId) {
        console.log(`[webhook] Skipping own page comment ${val.comment_id} — page replied`);
        continue;
      }

      // تجاهل العمليات غير "add" (تعديل أو حذف)
      if (val.verb && val.verb !== "add") {
        console.log(`[webhook] Skipping comment verb=${val.verb}`);
        continue;
      }

      // تجاهل التعليقات الأقدم من 10 دقائق
      const createdTime = (val as any).created_time as number | undefined;
      if (createdTime && Date.now() / 1000 - createdTime > 600) {
        console.log(`[webhook] Skipping stale comment (${Math.round(Date.now() / 1000 - createdTime)}s old)`);
        continue;
      }

      try {
        await handlePageComment(val, settings, config);
      } catch (commentErr) {
        console.error("❌ Unhandled error in comment handler:", (commentErr as Error).message);
      }
    }
  }
});

export default router;
