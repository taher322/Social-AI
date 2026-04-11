import express, { type Express, type Request, type Response, type NextFunction } from "express";
import cors from "cors";
import sharp from "sharp";
import router from "./routes/index.js";
import { authMiddleware } from "./middleware/authMiddleware.js";
import { db, productsTable, broadcastsTable } from "@workspace/db";
import { eq } from "drizzle-orm";

function buildAllowedOrigins(): string[] {
  const origins: string[] = [
    "http://localhost:3000",
    "http://localhost:5173",
  ];

  if (process.env["APP_URL"]) {
    origins.push(process.env["APP_URL"].replace(/\/$/, ""));
  }

  if (process.env["ALLOWED_ORIGINS"]) {
    process.env["ALLOWED_ORIGINS"]
      .split(",")
      .map((o) => o.trim())
      .filter(Boolean)
      .forEach((o) => origins.push(o.replace(/\/$/, "")));
  }

  if (process.env["REPLIT_DOMAINS"]) {
    process.env["REPLIT_DOMAINS"]
      .split(",")
      .map((d) => d.trim())
      .filter(Boolean)
      .forEach((d) => origins.push(`https://${d}`));
  }

  if (process.env["REPLIT_DEV_DOMAIN"]) {
    origins.push(`https://${process.env["REPLIT_DEV_DOMAIN"]}`);
  }

  return [...new Set(origins)];
}

const ALLOWED_ORIGINS = buildAllowedOrigins();

const dashboardLimiter = new Map<string, number[]>();
const DASHBOARD_MAX       = 200;
const DASHBOARD_WINDOW_MS = 60 * 1000;

setInterval(() => {
  const now = Date.now();
  const windowStart = now - DASHBOARD_WINDOW_MS;
  for (const [ip, timestamps] of dashboardLimiter.entries()) {
    const fresh = timestamps.filter((t) => t >= windowStart);
    if (fresh.length === 0) dashboardLimiter.delete(ip);
    else dashboardLimiter.set(ip, fresh);
  }
}, 5 * 60 * 1000);

function apiRateLimit(req: Request, res: Response, next: NextFunction): void {
  if (req.path.startsWith("/webhook")) return next();

  const ip  = (req.headers["x-forwarded-for"] as string | undefined)?.split(",")[0]?.trim()
            ?? req.socket.remoteAddress
            ?? "unknown";
  const now         = Date.now();
  const windowStart = now - DASHBOARD_WINDOW_MS;
  const prev        = (dashboardLimiter.get(ip) ?? []).filter((t) => t >= windowStart);

  if (prev.length >= DASHBOARD_MAX) {
    res.status(429).json({ message: "Too Many Requests — slow down" });
    return;
  }
  dashboardLimiter.set(ip, [...prev, now]);
  next();
}

const app: Express = express();

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin) return callback(null, true);
      if (ALLOWED_ORIGINS.includes(origin)) return callback(null, true);
      callback(null, false);
    },
  })
);
app.use(
  express.json({
    verify: (req, _res, buf) => {
      (req as import("express").Request).rawBody = buf;
    },
  })
);
app.use(express.urlencoded({ extended: true }));

// ── Public image routes — NO auth required ────────────────────────────────────
// Registered directly on the app BEFORE authMiddleware so Facebook's servers
// can fetch product/broadcast images without any token.

app.get("/api/products/image/:id/:index", async (req: Request, res: Response): Promise<void> => {
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
});

app.get("/api/broadcasts/image/:id", async (req: Request, res: Response): Promise<void> => {
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
});

// ─────────────────────────────────────────────────────────────────────────────

app.use(authMiddleware);
app.use("/api", apiRateLimit, router);

export default app;
