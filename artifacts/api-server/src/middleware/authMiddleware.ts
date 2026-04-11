import { Request, Response, NextFunction } from "express";
import { verifyToken } from "../lib/auth.js";

export interface AuthenticatedRequest extends Request {
  user?: { id: number; username: string };
}

const PUBLIC_PREFIXES = ["/api/webhook", "/api/auth", "/api/notifications/stream", "/api/products/image", "/api/broadcasts/image", "/api/healthz"];

export function authMiddleware(req: AuthenticatedRequest, res: Response, next: NextFunction): void {
  const fullPath = req.originalUrl.split("?")[0] ?? "";

  if (PUBLIC_PREFIXES.some((prefix) => fullPath.startsWith(prefix))) {
    next();
    return;
  }

  if (!fullPath.startsWith("/api")) {
    next();
    return;
  }

  const authHeader = req.headers["authorization"];
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    res.status(401).json({ message: "Unauthorized" });
    return;
  }

  const token = authHeader.slice(7);
  const payload = verifyToken(token);
  if (!payload) {
    res.status(401).json({ message: "Invalid or expired token" });
    return;
  }

  req.user = payload;
  next();
}
