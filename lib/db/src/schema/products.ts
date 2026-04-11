import { pgTable, serial, text, integer, real, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const productsTable = pgTable("products", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  description: text("description"),
  originalPrice: real("original_price"),
  discountPrice: real("discount_price"),
  stockQuantity: integer("stock_quantity").notNull().default(0),
  lowStockThreshold: integer("low_stock_threshold").notNull().default(5),
  status: text("status").notNull().default("available"),
  images: text("images"),
  mainImageIndex: integer("main_image_index").notNull().default(0),
  category: text("category"),
  brand: text("brand"),
  itemType: text("item_type"),
  priceTier: text("price_tier"),
  externalUrl: text("external_url"),
  folderId:    integer("folder_id"),
  fbImageUrl:  text("fb_image_url"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const insertProductSchema = createInsertSchema(productsTable).omit({ id: true, createdAt: true });
export type InsertProduct = z.infer<typeof insertProductSchema>;
export type Product = typeof productsTable.$inferSelect;
