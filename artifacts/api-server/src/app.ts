import express, { type Express, type Request, type Response, type NextFunction } from "express";
import cors from "cors";
import router from "./routes/index.js";
import { authMiddleware } from "./middleware/authMiddleware.js";

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

app.use(authMiddleware);
app.use("/api", apiRateLimit, router);

export default app;
