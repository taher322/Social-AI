# مساعد الصفحة الذكي — Facebook AI Agent

## Overview
مشروع متكامل من طرف لطرف (Full-Stack) للوحة تحكم ثنائية اللغة (عربي/إنجليزي) بدعم RTL كامل، مصمّم لأتمتة التفاعلات على Facebook Messenger وتعليقات المنشورات. يعتمد على نظام ذكاء اصطناعي متعدد المزودين قابل للضبط للرد التلقائي، وإدارة استفسارات العملاء، وتبسيط عمليات المبيعات.

---

## User Preferences
- أُفضّل التطوير التدريجي.
- يُرجى الاستفسار قبل إجراء تغييرات معمارية كبرى أو إدخال تبعيات خارجية جديدة.
- أُفضّل الشرح التفصيلي للمنطق المعقد أو قرارات التصميم.
- أُفضّل لغة بسيطة وواضحة في التواصل.
- `ai.ts` أصبح barrel file — لا تُضف كود مباشراً فيه. أضف في الملف الفرعي المناسب (aiEngine، aiPromptBuilder، aiMultimodal، إلخ).
- لا تُعدّل الملفات الفرعية للـ AI (aiEngine، aiPromptBuilder، aiMultimodal) بدون نقاش مسبق للتغييرات الكبيرة.
- لا تُعدّل `lib/db/src/schema/index.ts` بدون نقاش مسبق.

---

## System Architecture

**التقنيات المستخدمة:**
- **Monorepo**: pnpm workspaces
- **Node.js**: v24
- **TypeScript**: v5.9
- **API Framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod v4 + drizzle-zod
- **API Codegen**: Orval (React Query hooks + Zod schemas من OpenAPI)
- **Frontend**: React, Vite, TailwindCSS v4, shadcn/ui, Recharts, Framer Motion
- **Auth**: JWT (jsonwebtoken) + bcrypt
- **Build**: esbuild للـ API، Vite للـ dashboard

**هيكل المشروع:**
- `artifacts/api-server/` — Express 5 API (يخدم على `/api`)
  - `src/routes/webhook.ts` — معالج Webhook الرئيسي
  - `src/routes/orders.ts` — الطلبات + تصدير Excel (Ecotrack)
  - `src/lib/webhookUtils.ts` — دوال مساعدة نقية + re-export rate limiters
  - `src/lib/rateLimit.ts` — rate limiters هجينة Redis/in-memory
  - `src/lib/dbHelpers.ts` — دوال DB: getSettings، getConfig، isUserPaused، saveConversation
  - `src/lib/messengerUtils.ts` — sendFbQuickReplies، bufferMessage، getOrCreateSession
  - `src/lib/catalogFlow.ts` — sendDeliveryOptions، sendCatalogPage، sendCatalogCategoryMenu
    - **منطق الصور (3 مراحل):** 1) رابط CDN مخزّن في `fb_image_url` → 2) رفع الصورة لـ CDN عبر `uploadDataUrlToFbCdn` وتخزين الرابط → 3) Fallback بإرسال الصورة كرسالة منفصلة إذا فشل الرفع
  - `src/lib/orderFlow.ts` — handleProductPayload (كل تدفقات الطلبات)
  - `src/lib/ai.ts` — Barrel re-export فقط (واجهة موحدة لكل ملفات AI)
  - `src/lib/aiEngine.ts` — محرك مزودي AI: callAI، callAIWithMetadata، callAIWithLoadBalancing، retry logic
  - `src/lib/aiPromptBuilder.ts` — بناء System Prompt: buildSystemPrompt، buildCommentSystemPrompt، isWithinBusinessHours
  - `src/lib/aiMultimodal.ts` — تحليل الوسائط: analyzeAttachmentWithGemini، transcribeOrDescribeAttachment، classifyShoppingIntent، matchProductsFromAnalysis، summarizeProductForUser، getFreshAppointmentBlock
  - `src/lib/aiParsers.ts` — محللات JSON: parseOrderAction، parseAppointmentAction، إلخ
  - `src/lib/aiSafetyFilters.ts` — مرشحات الأمان: detectJailbreak، detectSalesTrigger، detectBookingIntent
  - `src/lib/aiFbApi.ts` — واجهة Facebook API: sendFbMessage، sendFbImageMessage، getFbUserName، **uploadDataUrlToFbCdn** (ترفع صورة كـ data URL لـ CDN فيسبوك وتُرجع رابطاً عاماً)، إلخ
  - `src/lib/vertexAi.ts` — مزود Vertex AI (callVertexAi، callVertexAiMultimodal، parseVertexConfig)
  - `src/lib/webhookAttachment.ts` — معالجة المرفقات (صور، صوت، فيديو)
  - `src/lib/cache.ts` — In-memory cache مع TTL
  - `src/lib/redisCache.ts` — Redis cache مع in-memory fallback
  - `public/ecotrack_template.xlsx` — قالب Ecotrack لتصدير الطلبات
- `artifacts/dashboard/` — React + Vite dashboard (يخدم على `/`)
- `lib/api-spec/` — OpenAPI 3.1 spec + Orval config
- `lib/api-client-react/` — Generated React Query hooks
- `lib/api-zod/` — Generated Zod schemas
- `lib/db/` — Drizzle ORM schema + database connection

**التوجيه:**
- مسارات API تعمل تحت `/api/*`
- Dashboard يعالج كل المسارات الأخرى `/*`

**التصميم:**
- تخطيط RTL كامل (يمين لليسار)
- Dark Mode كامل بنظام 3 طبقات لونية (CSS variables)
- خط Noto Kufi Arabic
- مكونات shadcn/ui
- تحريكات Framer Motion

---

## Core Features & Implementations

### 1. نظام المصادقة (Authentication)
- JWT مخزّن في `localStorage` كـ `fb_agent_token`
- `fetchWithAuth.ts` يُضيف التوكن تلقائياً لكل طلب
- `JWT_SECRET` مضبوط كمتغير بيئة دائم (64-char hex)
- **كلمة المرور**: تُولَّد تلقائياً عند أول تشغيل وتُطبَع في logs مرة واحدة — يُفضَّل ضبط `ADMIN_PASSWORD` كمتغير بيئة
- Middleware لحماية جميع مسارات الـ API

### 2. نظام الذكاء الاصطناعي متعدد المزودين
- مزودون مدعومون: **OpenAI، Anthropic، Gemini، DeepSeek، Groq، OpenRouter، Orbit، AgentRouter، Vertex AI، Custom**
- أولويات قابلة للضبط مع Failover وLoad Balancing تلقائي
- `buildSystemPrompt()` يبني Prompt ديناميكياً من: إعدادات الذكاء، المنتجات، FAQ، نطاقات العمل
- **`pageDescription`** مُحقَن في System Prompt — يُفيد الـ AI في تقديم النشاط للعميل
- **`pageFacebookUrl`** مُحقَن في System Prompt — الـ AI يُشارك الرابط عند سؤال العميل
- معالجة أخطاء 429 (Rate Limit) مع إعادة المحاولة عند المزود التالي
- صفحة `/providers` لإدارة المزودين من الواجهة

### 3. Vertex AI — دعم كامل
- **اتفاقية DB**: `provider.apiKey` → serviceAccountJson؛ `provider.baseUrl` → `"projectId|location"`؛ `provider.modelName` → اسم النموذج
- **bug المسافة**: دائماً `rawType.replace(/\s+/g, "")` قبل مقارنة `=== "vertexai"`
- **Token Cache**: `getVertexToken()` يُجدَّد كل 50 دقيقة بدون استدعاءات زائدة
- **callVertexAi()**: استدعاء نصي عادي مع تاريخ المحادثة
- **callVertexAiMultimodal()**: يُرسل نص + وسائط (صورة/صوت) عبر `inlineData` format
- **testVertexConnection()**: تُرجع `{ success, details }` — تُعالَج في providers.ts بشكل صحيح

### 4. معالجة المرفقات (Multimodal) — نظام Fallback مزدوج
**المنطق (Phase 7B + Vertex AI):**
- **صوت (Audio):** نسخ نصي بالعربية → النص يدخل pipeline الذكاء الطبيعي
- **صورة (Image):** وصف بالعربية → `[صورة]: <الوصف>` يدخل pipeline الذكاء
- **فيديو (Video):** نفس معالجة الصورة
- **`transcribeOrDescribeAttachment()` — نظام Fallback ثنائي:**
  - المرحلة 1: Gemini AI Studio (API Key مباشر)
  - المرحلة 2 (عند فشل Gemini أو غيابه):
    - **Vertex AI** → `callVertexAiMultimodal()` مباشرةً (يدعم صوت + صور)
    - **OpenAI-compatible** → vision URL (للصور فقط)
- **`analyzeImageWithActiveProvider()`:** يدعم Vertex AI عبر `rawTypeKey === "vertexai"` check قبل الفروع الأخرى
- حد 15MB للمرفقات
- `fromAttachment` flag يُعطّل `CATALOG_INTENT_PATTERNS` أثناء المرفقات الصوتية

### 5. فهرس المنتجات (Product Catalog)
- CRUD كامل للمنتجات مع صور ومخزون وإمكانية Pre-order
- نظام التصنيفات الهرمية (رئيسية + فرعية)
- جدول `productCategoriesTable`
- اختيار متعدد للتصنيفات
- **نظام المجلدات**: جدول `product_folders` + حقل `folderId` في المنتجات + tabs تصفية + bulk-assign
- **حقل `fb_image_url`**: مخزّن في جدول `products` — يُحفظ فيه رابط CDN فيسبوك بعد أول رفع لتجنب إعادة الرفع في كل طلب

### 6. نظام المواعيد (Appointments)
- حجز مواعيد مع خانات زمنية قابلة للضبط
- حد أقصى للحجوزات في اليوم
- تتبع الحالة: معلق، مؤكد، ملغى، مكتمل

### 7. إدارة الطلبات (Orders) + التوصيل
- حقول الطلب: الاسم، الهاتف، **البلدية** (customerCommune)، الولاية، العنوان
- **5 حقول** يجمعها AI تلقائياً من Messenger بالترتيب
- عرض خيارات التوصيل بعد اكتمال البيانات
- `sendDeliveryOptions()` تعرض أسعار المنزل/المكتب بالـ Quick Replies
- `deliveryEnabled` guard لحماية من الأزرار القديمة

### 8. تصدير Excel بصيغة Ecotrack
- **Endpoint**: `GET /api/orders/export?status=<filter>`
- **القالب**: `public/ecotrack_template.xlsx` — 18 عموداً بتنسيق Ecotrack الرسمي
- **الحقول المُملَّأة**: رقم الطلب، الاسم، الهاتف، كود الولاية (تلقائي من delivery_prices)، الولاية، البلدية، العنوان، المنتج، المبلغ، FRAGILE="OUI"
- **الحقول الفارغة**: هاتف 2، الوزن، ملاحظة، ECHANGE، PICK UP، RECOUVREMENT، STOP DESK، Lien map
- **إصلاح المسار في الإنتاج**: `findTemplate()` تجرب 4 مسارات لضمان العمل في كل البيئات
- **في الواجهة**: زر "تصدير Excel" يستخدم `fetch + blob` مع Authorization header

### 9. أسعار التوصيل لكل ولاية
- **69 ولاية جزائرية** بأسماء عربية وإنجليزية
- جدول `deliveryPricesTable`: سعر منزلي + سعر مكتبي
- البحث بـ `wilayaId` (رقم ثابت) لا بالاسم
- **تصحيحات الأسماء**: id=16 "الجزائر العاصمة"، id=64 "بئر العاتر"، id=69 "العريشة"
- ⚠️ `ensureWilayasExist()` تُزامن DB من `ALGERIA_WILAYAS` في الكود — دائماً عدّل الكود لا DB فقط

### 10. البث الجماعي (Broadcasts)
- حملات رسائل جماعية مع فلاتر استهداف
- تطبيق نافذة 24 ساعة لـ Messenger
- جدولة البث بوقت محدد
- **إرسال الصور**: رفع multipart مباشر عبر `sendFbImageFromDataUrl` (لا URL خارجي)
- **منطق مستقل**: فشل الصورة لا يلغي الرسالة النصية — كل منهما في `try/catch` مستقل

### 11. العملاء المحتملون (Leads)
- التقاط تلقائي لمعلومات التواصل — مُتحكَّم به بإعداد `leadCaptureFields`
- تصدير CSV

### 12. الأسئلة الشائعة (FAQ)
- CRUD للأسئلة والأجوبة
- مُضاف تلقائياً في System Prompt

### 13. إدارة المحادثات
- عرض مزدوج (قائمة + Chat view)
- إيقاف/استئناف الـ AI لكل محادثة
- تسمية العملاء، تحليل المشاعر، ملاحظات المشغل

### 14. لوحة التحليلات (Analytics Dashboard)
- بطاقات إحصائية: محادثات اليوم، رسائل اليوم، الطلبات المعلقة، المواعيد القادمة
- مخطط ساعات الذروة
- دعم الأوقات التي تتخطى منتصف الليل

### 15. Catalog Browser (Phase 6)
- الـ AI يُصدر `{"action":"browse_catalog"}` ← يُعترض في الـ webhook
- يعرض أزرار Quick Reply للفئات وعروض المنتجات
- أزرار المخزون الذكية + تكامل Pre-order

### 16. إعدادات الصفحة والنظام
- صفحة `/settings` للإعدادات الكاملة
- صفحة `/providers` لإدارة مزودي الذكاء الاصطناعي مع دعم Vertex AI
- Human Handoff مع منطق توقف/إعادة تفعيل الذكاء

### 17. Dark Mode كامل
- نظام 3 طبقات CSS variables:
  - خلفية رئيسية (8% brightness)
  - بطاقات/كاردات (13%)
  - أسطح مخففة (19%)
- 252 لوناً ثابتاً تم استبدالها بمتغيرات CSS عبر 18 ملف

### 18. تأمين الـ Webhook (Security Hardening)
- Layer 1 — IP Rate Limit: 120 طلب/دقيقة/IP (HTTP 429)
- Layer 2 — Signature Verification: `X-Hub-Signature-256` HMAC-SHA256
- Layer 3 — Replay Attack Prevention: أحداث أعمر من 10 دقائق تُرفض (يشمل التعليقات)
- Layer 4 — Idempotency: جدول `processed_messages` يمنع تكرار المعالجة
- Layer 5 — Text Rate Limit: 30 رسالة/دقيقة/sender
- Layer 6 — Attachment Rate Limit: 5 مرفقات/دقيقتين/user

### 19. Idempotency — منع تكرار المعالجة
- جدول `processed_messages` يحفظ `mid` (Message ID)
- سجلات تُحذف تلقائياً بعد 2 ساعة

### 20. Redis Cache مع In-Memory Fallback (Hybrid)
- `redisCache.ts` — `rGet/rSet/rDel` تُجرّب Redis أولاً وتُرجع لـ in-memory عند غيابه
- TTLs: SETTINGS/CONFIG/FAQS = 365 يوم، PRODUCTS = 30 دقيقة، FB_USER = 30 دقيقة

### 21. ملخص المنتج بالذكاء الاصطناعي
- `summarizeProductForUser()` مُصدَّرة من `ai.ts`
- تُستدعى عند ضغط "تفصيل المنتج" (DETAILS payload)
- Fallback تلقائي للنص الكلاسيكي عند فشل الـ AI

### 22. معالجة التعليقات (Comments)
- اشتراك تلقائي في feed events عبر `subscribePageToFeedEvents()`
- تجاهل التعليقات الأقدم من 10 دقائق (Replay Protection)
- رد علني: "تم الرد في الخاص 📩" + رد تفصيلي في DM
- استخراج Page Token تلقائياً من `/me/accounts` عند الحاجة

---

## Database Tables
| الجدول | الوصف |
|--------|--------|
| `fb_settings` | إعدادات الصفحة والتكامل |
| `ai_config` | إعدادات الذكاء الاصطناعي والـ System Prompt |
| `ai_providers` | مزودو الذكاء الاصطناعي وأولوياتهم |
| `products` | كتالوج المنتجات (يشمل folderId + fb_image_url لتخزين رابط CDN فيسبوك) |
| `product_categories` | تصنيفات المنتجات الهرمية |
| `product_folders` | مجلدات تنظيم المنتجات |
| `conversations` | سجل المحادثات الكامل |
| `conversation_sessions` | جلسات المحادثة النشطة |
| `orders` | الطلبات المكتملة (يشمل customerCommune) |
| `order_sessions` | جلسات جمع بيانات الطلب (يشمل customerCommune) |
| `appointments` | المواعيد |
| `available_slots` | الخانات الزمنية المتاحة |
| `leads` | العملاء المحتملون |
| `faqs` | الأسئلة الشائعة |
| `delivery_prices` | أسعار التوصيل لكل ولاية (69 ولاية) |
| `comments_log` | تعليقات المنشورات |
| `platform_events` | أحداث النظام للتتبع والتشخيص |
| `user_product_context` | آخر منتج تفاعل معه كل مستخدم |
| `pre_orders` | الطلبات المسبقة |
| `pre_order_sessions` | جلسات الطلب المسبق |
| `product_inquiries` | استفسارات المنتجات |
| `broadcasts` | حملات البث الجماعي |
| `broadcast_templates` | قوالب رسائل البث الجاهزة |
| `admins` | حسابات المشرفين |
| `subscription_plans` | خطط الاشتراك |
| `subscription_usage` | استهلاك المحادثات والبث الشهري |
| `domain_templates` | قوالب جاهزة لكل مجال تجاري |
| `processed_messages` | معرّفات الرسائل المعالجة (Idempotency) |
| `provider_usage_log` | سجل أداء مزودي AI |
| `user_counters` | عدادات المستخدمين (offTopicCount) |

---

## متغيرات البيئة المطلوبة (Environment Variables)

| المتغير | الإلزامية | الوصف |
|---|---|---|
| `DATABASE_URL` | إلزامي | رابط اتصال PostgreSQL |
| `PORT` | إلزامي | منفذ الـ API server (8080) |
| `BASE_PATH` | إلزامي | مسار الـ dashboard (`/`) |
| `ENCRYPTION_KEY` | إلزامي | مفتاح AES-256 لتشفير API keys (≥32 حرف) |
| `JWT_SECRET` | مهم جداً | مفتاح توقيع توكنات الدخول |
| `ADMIN_USERNAME` | أمني | اسم مدير لوحة التحكم (افتراضي: admin) |
| `ADMIN_PASSWORD` | أمني | كلمة مرور المدير — **إذا غاب: تُولَّد عشوائياً وتُطبَع في logs مرة واحدة** |
| `APP_URL` | إلزامي في الإنتاج | رابط التطبيق الكامل |
| `ALLOWED_ORIGINS` | اختياري | نطاقات CORS إضافية مفصولة بفواصل |
| `REDIS_URL` | موصى به | رابط Redis للـ cache الدائم |
| `VITE_API_URL` | اختياري | رابط الـ API للـ dashboard |
| `GITHUB_TOKEN` | للنشر فقط | Personal Access Token لـ GitHub |

> انظر ملف `.env.example` في جذر المشروع للتفاصيل الكاملة.

---

## External Dependencies
- **Facebook Graph API** — إرسال الرسائل، الـ Webhooks، التفاعل مع الصفحات
- **PostgreSQL** — قاعدة البيانات الرئيسية
- **مزودو الذكاء الاصطناعي (مدعومون):**
  - OpenAI / OpenAI-compatible
  - Anthropic (Claude)
  - Google Gemini (AI Studio API Key)
  - Google Vertex AI (Service Account JSON)
  - DeepSeek
  - Groq
  - OpenRouter
  - Orbit
  - AgentRouter
  - Custom (قابل للضبط)

---

## Important Implementation Notes
- **Sidebar**: يستخدم `inline styles` وليس Tailwind responsive classes
- **Route registration**: المسارات الجديدة تحتاج إعادة تشغيل السيرفر
- **Workflow الحقيقي**: `artifacts/api-server: API Server` يخدم على port 8080
- **JWT**: `JWT_SECRET` مضبوط كـ environment variable دائم
- **Vertex AI bug المسافة**: `rawType.replace(/\s+/g, "")` → `"vertexai"` قبل أي مقارنة
- **botEnabled**: يُوقِف الرسائل النصية + المرفقات + الأزرار + التعليقات
- **Dark Mode**: متغيرات CSS في `artifacts/dashboard/src/index.css` — لا تستخدم ألواناً ثابتة
- **Delivery wilayaId search**: البحث بـ `wilayaId` أولاً لا بالاسم
- **shopctx cache key**: `shopctx:{senderId}` — TTL ديناميكي (20 دقيقة للتصفح، 5 دقائق للـ DROP)
- **offTopicCount**: مُخزَّن في `user_counters` (DB) — يبقى عبر إعادة تشغيل الخادم
- **product description in catalog**: أول **300 حرف** في قائمة المنتجات للـ AI
- **Ecotrack template path**: `findTemplate()` في `orders.ts` تجرب 4 مسارات (dev + prod + cwd variants)
- **Export auth**: زر التصدير يستخدم `fetch + blob` مع `Authorization: Bearer <token>` وليس رابطاً مباشراً
- **customerCommune**: حقل البلدية مُضاف لـ `ordersTable` و`orderSessionsTable` — AI يطلبه بين الولاية والعنوان
- **GitHub Push**: دائماً orphan branch strategy مع `--force` → `tahersa21/Social-AI`
- **بيئتا الإنتاج والتطوير**: قاعدتا بيانات منفصلتان — البيانات لا تنتقل تلقائياً بين البيئتين
