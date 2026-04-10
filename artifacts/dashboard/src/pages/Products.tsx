import React, { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { Package, Plus, Edit, Trash2, Star, X, Image as ImageIcon, Loader2, Minus, Tag, FolderOpen, ChevronRight, Folder, FolderPlus, Check, CheckSquare, Square, Pencil, Sparkles } from "lucide-react";
import { useListProducts, useDeleteProduct, useAdjustProductStock, Product } from "@workspace/api-client-react";
import { createProductWithImages, updateProductWithImages } from "@/lib/api-overrides";
import { toast } from "@/hooks/use-toast";
import { useQueryClient } from "@tanstack/react-query";

import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";

const BASE = import.meta.env.BASE_URL?.replace(/\/$/, "") || "";

interface ProductCategory {
  id: number;
  name: string;
  parentId: number | null;
}

async function fetchCategories(): Promise<ProductCategory[]> {
  const res = await fetch(`${BASE}/api/product-categories`, { credentials: "include" });
  if (!res.ok) throw new Error("فشل تحميل التصنيفات");
  return res.json();
}

async function createCategory(name: string, parentId?: number | null): Promise<ProductCategory> {
  const res = await fetch(`${BASE}/api/product-categories`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ name, parentId: parentId ?? null }),
  });
  if (!res.ok) throw new Error("فشل الإضافة");
  return res.json();
}

async function updateCategory(id: number, name: string, parentId?: number | null): Promise<ProductCategory> {
  const res = await fetch(`${BASE}/api/product-categories/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ name, parentId: parentId ?? null }),
  });
  if (!res.ok) throw new Error("فشل التعديل");
  return res.json();
}

async function deleteCategory(id: number): Promise<void> {
  const res = await fetch(`${BASE}/api/product-categories/${id}`, {
    method: "DELETE",
    credentials: "include",
  });
  if (!res.ok) throw new Error("فشل الحذف");
}

// ── Folder interfaces & API ──────────────────────────────────────────────────
interface ProductFolder { id: number; name: string; createdAt: string; }
type ProductExt = Product & { folderId?: number | null };

async function fetchFolders(): Promise<ProductFolder[]> {
  const res = await fetch(`${BASE}/api/product-folders`, { credentials: "include" });
  if (!res.ok) throw new Error("فشل تحميل المجلدات");
  return res.json();
}
async function createFolder(name: string): Promise<ProductFolder> {
  const res = await fetch(`${BASE}/api/product-folders`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    credentials: "include", body: JSON.stringify({ name }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { message?: string }).message || `HTTP ${res.status}`);
  }
  return res.json();
}
async function renameFolder(id: number, name: string): Promise<ProductFolder> {
  const res = await fetch(`${BASE}/api/product-folders/${id}`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    credentials: "include", body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error("فشل التعديل");
  return res.json();
}
async function deleteFolder(id: number): Promise<void> {
  const res = await fetch(`${BASE}/api/product-folders/${id}`, {
    method: "DELETE", credentials: "include",
  });
  if (!res.ok) throw new Error("فشل الحذف");
}
async function bulkAssignFolder(productIds: number[], folderId: number | null): Promise<void> {
  const res = await fetch(`${BASE}/api/product-folders/bulk-assign`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    credentials: "include", body: JSON.stringify({ productIds, folderId }),
  });
  if (!res.ok) throw new Error("فشل التعيين");
}

export default function Products() {
  const queryClient = useQueryClient();
  const { data: products = [], isLoading } = useListProducts();
  const deleteMut = useDeleteProduct();
  const stockMut = useAdjustProductStock();

  // Product form dialog
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingProduct, setEditingProduct] = useState<Product | null>(null);

  const [formData, setFormData] = useState({
    name: "", description: "", originalPrice: "", discountPrice: "",
    stockQuantity: "10", lowStockThreshold: "5", status: "available",
    brand: "", itemType: "", externalUrl: "",
  });
  const [selectedCategory, setSelectedCategory] = useState<string>("");
  const [selectedFolderId, setSelectedFolderId] = useState<string>("");
  const [files, setFiles] = useState<File[]>([]);
  const [existingImages, setExistingImages] = useState<string[]>([]);
  const [mainImageIndex, setMainImageIndex] = useState(0);
  const [isSummarizing, setIsSummarizing] = useState(false);

  const handleSummarize = async () => {
    if (!formData.description.trim()) {
      toast({ title: "الوصف فارغ", description: "أضف وصفاً أولاً ثم اضغط تلخيص", variant: "destructive" });
      return;
    }
    setIsSummarizing(true);
    try {
      const res = await fetch(`${BASE}/api/products/summarize-description`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ description: formData.description }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "فشل التلخيص");
      setFormData(prev => ({ ...prev, description: data.summary }));
      toast({ title: "تم التلخيص", description: "تم استخراج المواصفات المهمة بنجاح" });
    } catch (err: any) {
      toast({ title: "خطأ", description: err.message, variant: "destructive" });
    } finally {
      setIsSummarizing(false);
    }
  };
  const [isSaving, setIsSaving] = useState(false);

  // Categories management dialog
  const [catsOpen, setCatsOpen] = useState(false);
  const [allCategories, setAllCategories] = useState<ProductCategory[]>([]);
  const [catsLoading, setCatsLoading] = useState(false);
  const [newCatName, setNewCatName] = useState("");
  const [newCatParent, setNewCatParent] = useState<string>("none");
  const [editingCat, setEditingCat] = useState<ProductCategory | null>(null);
  const [editCatName, setEditCatName] = useState("");
  const [editCatParent, setEditCatParent] = useState<string>("none");
  const [catSaving, setCatSaving] = useState(false);

  // Category picker popover in product form
  const [catPickerOpen, setCatPickerOpen] = useState(false);

  // Folders state
  const [allFolders, setAllFolders] = useState<ProductFolder[]>([]);
  const [foldersOpen, setFoldersOpen] = useState(false);
  const [activeFolderTab, setActiveFolderTab] = useState<number | null>(null);
  const [newFolderName, setNewFolderName] = useState("");
  const [folderSaving, setFolderSaving] = useState(false);
  const [editingFolder, setEditingFolder] = useState<ProductFolder | null>(null);
  const [editFolderName, setEditFolderName] = useState("");

  // Bulk select state
  const [bulkMode, setBulkMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [bulkTargetFolder, setBulkTargetFolder] = useState<string>("");

  const loadCategories = async () => {
    setCatsLoading(true);
    try {
      const cats = await fetchCategories();
      setAllCategories(cats);
    } catch {
      toast({ title: "خطأ", description: "تعذر تحميل التصنيفات", variant: "destructive" });
    } finally {
      setCatsLoading(false);
    }
  };

  const loadFolders = async () => {
    try {
      const folders = await fetchFolders();
      setAllFolders(folders);
    } catch {
      toast({ title: "خطأ", description: "تعذر تحميل المجلدات", variant: "destructive" });
    }
  };

  const handleCreateFolder = async () => {
    if (!newFolderName.trim()) return;
    setFolderSaving(true);
    try {
      await createFolder(newFolderName.trim());
      setNewFolderName("");
      await loadFolders();
      queryClient.invalidateQueries({ queryKey: ["/api/products"] });
      toast({ title: "تم إنشاء المجلد" });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "فشل إنشاء المجلد";
      toast({ title: "خطأ", description: msg, variant: "destructive" });
    } finally {
      setFolderSaving(false);
    }
  };

  const handleRenameFolder = async () => {
    if (!editingFolder || !editFolderName.trim()) return;
    setFolderSaving(true);
    try {
      await renameFolder(editingFolder.id, editFolderName.trim());
      setEditingFolder(null);
      await loadFolders();
      toast({ title: "تم التعديل" });
    } catch {
      toast({ title: "خطأ", description: "فشل التعديل", variant: "destructive" });
    } finally {
      setFolderSaving(false);
    }
  };

  const handleDeleteFolder = async (id: number) => {
    if (!confirm("هل أنت متأكد؟ سيتم إزالة المنتجات من هذا المجلد.")) return;
    try {
      await deleteFolder(id);
      if (activeFolderTab === id) setActiveFolderTab(null);
      await loadFolders();
      queryClient.invalidateQueries({ queryKey: ["/api/products"] });
      toast({ title: "تم الحذف" });
    } catch {
      toast({ title: "خطأ", description: "فشل الحذف", variant: "destructive" });
    }
  };

  const handleBulkAssign = async () => {
    if (selectedIds.size === 0) return;
    const fid = bulkTargetFolder === "none" ? null : parseInt(bulkTargetFolder, 10);
    try {
      await bulkAssignFolder([...selectedIds], fid);
      queryClient.invalidateQueries({ queryKey: ["/api/products"] });
      setSelectedIds(new Set());
      setBulkMode(false);
      setBulkTargetFolder("");
      toast({ title: "تم التعيين", description: `${selectedIds.size} منتج تم نقله` });
    } catch {
      toast({ title: "خطأ", description: "فشل التعيين", variant: "destructive" });
    }
  };

  const toggleSelect = (id: number) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  useEffect(() => {
    loadCategories();
    loadFolders();
  }, []);

  const openDialog = (prod?: Product) => {
    if (prod) {
      setEditingProduct(prod);
      setFormData({
        name: prod.name,
        description: prod.description || "",
        originalPrice: prod.originalPrice?.toString() || "",
        discountPrice: prod.discountPrice?.toString() || "",
        stockQuantity: prod.stockQuantity.toString(),
        lowStockThreshold: prod.lowStockThreshold.toString(),
        status: prod.status,
        brand: prod.brand || "",
        itemType: prod.itemType || "",
        externalUrl: prod.externalUrl || "",
      });
      const catStr = prod.category || "";
      setSelectedCategory(catStr.trim());
      setSelectedFolderId((prod as ProductExt).folderId?.toString() ?? "");
      const imgs = prod.images ? JSON.parse(prod.images) : [];
      setExistingImages(imgs);
      setFiles([]);
      setMainImageIndex(prod.mainImageIndex || 0);
    } else {
      setEditingProduct(null);
      setFormData({
        name: "", description: "", originalPrice: "", discountPrice: "",
        stockQuantity: "10", lowStockThreshold: "5", status: "available",
        brand: "", itemType: "", externalUrl: "",
      });
      setSelectedCategory("");
      setSelectedFolderId(activeFolderTab ? activeFolderTab.toString() : "");
      setExistingImages([]);
      setFiles([]);
      setMainImageIndex(0);
    }
    setDialogOpen(true);
  };

  const buildCategoryPath = (catId: number): string => {
    const cat = allCategories.find(c => c.id === catId);
    if (!cat) return "";
    if (!cat.parentId) return cat.name;
    const parentPath = buildCategoryPath(cat.parentId);
    return parentPath ? `${parentPath}/${cat.name}` : cat.name;
  };

  const selectCategory = (cat: ProductCategory) => {
    const path = buildCategoryPath(cat.id);
    setSelectedCategory(prev => prev === path ? "" : path);
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const totalCurrent = existingImages.length + files.length;
    const remaining = 5 - totalCurrent;
    if (remaining <= 0) { toast({ title: "عفواً", description: "الحد الأقصى 5 صور", variant: "destructive" }); return; }
    setFiles(prev => [...prev, ...acceptedFiles.slice(0, remaining)]);
  }, [existingImages.length, files.length]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': [] }
  });

  const removeExisting = (index: number) => {
    setExistingImages(prev => prev.filter((_, i) => i !== index));
    if (mainImageIndex === index) setMainImageIndex(0);
    else if (mainImageIndex > index) setMainImageIndex(mainImageIndex - 1);
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
    const globalIdx = existingImages.length + index;
    if (mainImageIndex === globalIdx) setMainImageIndex(0);
    else if (mainImageIndex > globalIdx) setMainImageIndex(mainImageIndex - 1);
  };

  const setMain = (globalIndex: number) => {
    setMainImageIndex(globalIndex);
  };

  const handleSave = async () => {
    if (!formData.name) { toast({ title: "خطأ", description: "الاسم مطلوب", variant: "destructive" }); return; }
    setIsSaving(true);
    try {
      const payload = {
        ...formData,
        category: selectedCategory,
        folderId: selectedFolderId || "",
        mainImageIndex: mainImageIndex.toString(),
        keepImages: existingImages
      };

      if (editingProduct) {
        await updateProductWithImages(editingProduct.id, payload, files);
        toast({ title: "تم الحفظ", description: "تم تحديث المنتج بنجاح" });
      } else {
        await createProductWithImages(payload, files);
        toast({ title: "تمت الإضافة", description: "تمت إضافة المنتج بنجاح" });
      }
      queryClient.invalidateQueries({ queryKey: ["/api/products"] });
      setDialogOpen(false);
    } catch (err: any) {
      toast({ title: "خطأ", description: err.message, variant: "destructive" });
    } finally {
      setIsSaving(false);
    }
  };

  const handleStock = (id: number, change: number) => {
    stockMut.mutate({ id, data: { quantityChange: change } }, {
      onSuccess: () => queryClient.invalidateQueries({ queryKey: ["/api/products"] })
    });
  };

  const discountPercent = formData.originalPrice && formData.discountPrice
    ? Math.round((1 - parseFloat(formData.discountPrice) / parseFloat(formData.originalPrice)) * 100)
    : 0;

  // Categories management
  const handleAddCategory = async () => {
    if (!newCatName.trim()) return;
    setCatSaving(true);
    try {
      await createCategory(newCatName.trim(), newCatParent !== "none" ? Number(newCatParent) : null);
      setNewCatName("");
      setNewCatParent("none");
      await loadCategories();
      toast({ title: "تمت الإضافة" });
    } catch {
      toast({ title: "خطأ", description: "فشل إضافة التصنيف", variant: "destructive" });
    } finally {
      setCatSaving(false);
    }
  };

  const handleEditCatSave = async () => {
    if (!editingCat || !editCatName.trim()) return;
    setCatSaving(true);
    try {
      await updateCategory(editingCat.id, editCatName.trim(), editCatParent !== "none" ? Number(editCatParent) : null);
      setEditingCat(null);
      await loadCategories();
      toast({ title: "تم التعديل" });
    } catch {
      toast({ title: "خطأ", description: "فشل التعديل", variant: "destructive" });
    } finally {
      setCatSaving(false);
    }
  };

  const handleDeleteCat = async (id: number) => {
    if (!confirm("هل أنت متأكد من حذف هذا التصنيف؟")) return;
    try {
      await deleteCategory(id);
      await loadCategories();
      toast({ title: "تم الحذف" });
    } catch {
      toast({ title: "خطأ", description: "فشل الحذف", variant: "destructive" });
    }
  };

  const parentCategories = allCategories.filter(c => !c.parentId);
  const getChildren = (parentId: number) => allCategories.filter(c => c.parentId === parentId);

  const getDescendantIds = (id: number): number[] => {
    const children = allCategories.filter(c => c.parentId === id);
    return [id, ...children.flatMap(c => getDescendantIds(c.id))];
  };

  const buildFlatCatOptions = (excludeId?: number): { id: number; label: string; depth: number }[] => {
    const excluded = excludeId ? new Set(getDescendantIds(excludeId)) : new Set<number>();
    const result: { id: number; label: string; depth: number }[] = [];
    const traverse = (parentId: number | null, depth: number) => {
      allCategories
        .filter(c => c.parentId === parentId && !excluded.has(c.id))
        .forEach(c => {
          result.push({ id: c.id, label: c.name, depth });
          traverse(c.id, depth + 1);
        });
    };
    traverse(null, 0);
    return result;
  };

  // Filter products by active folder tab
  const displayProducts = (activeFolderTab === null
    ? (products as ProductExt[])
    : (products as ProductExt[]).filter(p => p.folderId === activeFolderTab));

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6 max-w-7xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">المنتجات</h1>
          <p className="text-muted-foreground mt-1">إدارة الكتالوج والمخزون المتاح للبوت</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant={bulkMode ? "default" : "outline"}
            onClick={() => { setBulkMode(b => !b); setSelectedIds(new Set()); }}
            className="gap-2 h-11 px-4 rounded-xl"
          >
            <CheckSquare className="w-4 h-4" /> {bulkMode ? "إلغاء التحديد" : "تحديد متعدد"}
          </Button>
          <Button
            variant="outline"
            onClick={() => { setFoldersOpen(true); }}
            className="gap-2 h-11 px-4 rounded-xl border-amber-400/50 text-amber-700 hover:bg-amber-50"
          >
            <FolderPlus className="w-4 h-4" /> المجلدات
          </Button>
          <Button
            variant="outline"
            onClick={() => { setCatsOpen(true); loadCategories(); }}
            className="gap-2 h-11 px-5 rounded-xl border-primary/30 text-primary hover:bg-primary/5"
          >
            <FolderOpen className="w-4 h-4" /> التصنيفات
          </Button>
          <Button
            onClick={() => openDialog()}
            className="gap-2 h-11 px-6 rounded-xl shadow-lg shadow-primary/20 hover:-translate-y-0.5 transition-transform"
          >
            <Plus className="w-4 h-4" /> إضافة منتج
          </Button>
        </div>
      </div>

      {/* ── Folder Tabs ── */}
      {allFolders.length > 0 && (
        <div className="flex flex-wrap gap-2 items-center">
          <button
            onClick={() => setActiveFolderTab(null)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
              activeFolderTab === null
                ? "bg-primary text-primary-foreground shadow"
                : "bg-muted text-muted-foreground hover:bg-muted/80"
            }`}
          >
            📦 الكل ({(products as ProductExt[]).length})
          </button>
          {allFolders.map(f => {
            const count = (products as ProductExt[]).filter(p => p.folderId === f.id).length;
            return (
              <button
                key={f.id}
                onClick={() => setActiveFolderTab(f.id)}
                className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors flex items-center gap-1.5 ${
                  activeFolderTab === f.id
                    ? "bg-amber-500 text-white shadow"
                    : "bg-amber-50 text-amber-700 hover:bg-amber-100 border border-amber-200"
                }`}
              >
                <Folder className="w-3.5 h-3.5" /> {f.name} <span className="opacity-70">({count})</span>
              </button>
            );
          })}
        </div>
      )}

      {/* ── Bulk Select Bar ── */}
      {bulkMode && (
        <div className="flex flex-wrap items-center gap-3 p-3 bg-primary/5 border border-primary/20 rounded-xl">
          <span className="text-sm font-medium text-primary">{selectedIds.size} منتج محدد</span>
          <div className="flex-1 flex items-center gap-2">
            <Select value={bulkTargetFolder} onValueChange={setBulkTargetFolder}>
              <SelectTrigger className="h-9 max-w-[200px] bg-card">
                <SelectValue placeholder="نقل إلى مجلد..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">— بدون مجلد —</SelectItem>
                {allFolders.map(f => (
                  <SelectItem key={f.id} value={String(f.id)}>
                    <Folder className="w-3 h-3 inline me-1 text-amber-500" />{f.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button size="sm" className="h-9 rounded-lg" disabled={selectedIds.size === 0 || !bulkTargetFolder} onClick={handleBulkAssign}>
              نقل
            </Button>
          </div>
          <Button size="sm" variant="ghost" className="h-9 text-muted-foreground" onClick={() => { setSelectedIds(new Set()); setBulkMode(false); }}>
            إلغاء
          </Button>
        </div>
      )}

      {isLoading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {[1,2,3,4].map(i => <div key={i} className="h-80 rounded-2xl bg-muted animate-pulse" />)}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          <AnimatePresence>
            {displayProducts.map(product => {
              const imgs = product.images ? JSON.parse(product.images) : [];
              const mainImgUrl = imgs[product.mainImageIndex || 0] || "https://placehold.co/400x400/f8fafc/94a3b8?text=No+Image";
              const stockPercent = Math.min(100, (product.stockQuantity / (product.lowStockThreshold * 4)) * 100);
              const progressColor = stockPercent > 50 ? 'bg-emerald-500' : stockPercent > 20 ? 'bg-amber-500' : 'bg-red-500';
              const productCats = (product.category || "").split(",").map((c: string) => c.trim()).filter(Boolean);
              const productFolder = allFolders.find(f => f.id === product.folderId);
              const isSelected = selectedIds.has(product.id);

              return (
                <motion.div key={product.id} layout initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }}
                  onClick={() => bulkMode && toggleSelect(product.id)}
                  className={bulkMode ? "cursor-pointer" : ""}
                >
                  <Card className={`overflow-hidden border-2 shadow-sm hover:shadow-xl transition-all duration-300 group rounded-2xl ${
                    isSelected ? "border-primary shadow-primary/20" : "border-border/50 hover:border-primary/20"
                  }`}>
                    <div className="relative aspect-square bg-muted overflow-hidden">
                      {/* Checkbox overlay in bulk mode */}
                      {bulkMode && (
                        <div className="absolute top-2 right-2 z-10">
                          {isSelected
                            ? <CheckSquare className="w-6 h-6 text-primary bg-white rounded shadow" />
                            : <Square className="w-6 h-6 text-white/80 bg-black/20 rounded shadow" />
                          }
                        </div>
                      )}
                      <img src={mainImgUrl} alt={product.name} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
                      {imgs.length > 1 && (
                        <Badge variant="secondary" className="absolute top-2 left-2 bg-card/90 backdrop-blur-sm shadow-sm">
                          1/{imgs.length} <ImageIcon className="w-3 h-3 ms-1" />
                        </Badge>
                      )}
                      {product.status === 'out_of_stock' && (
                        <div className="absolute inset-0 bg-card/60 backdrop-blur-[2px] flex items-center justify-center">
                          <span className="bg-red-500 text-white px-4 py-2 rounded-lg font-bold shadow-lg rotate-12">نفد المخزون</span>
                        </div>
                      )}
                      {product.discountPrice && product.originalPrice && (
                        <Badge className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 shadow-md px-2 py-1 text-sm font-bold">
                          {Math.round((1 - product.discountPrice / product.originalPrice) * 100)}% خفض
                        </Badge>
                      )}
                    </div>
                    <CardContent className="p-4 space-y-4 bg-card">
                      <div>
                        <h3 className="font-bold text-lg leading-tight truncate" title={product.name}>{product.name}</h3>
                        {(productCats.length > 0 || product.brand || productFolder) && (
                          <div className="flex flex-wrap gap-1 mt-1.5">
                            {productFolder && (
                              <span className="text-[10px] bg-amber-50 text-amber-700 border border-amber-200 rounded-md px-1.5 py-0.5 font-medium flex items-center gap-0.5">
                                <Folder className="w-2.5 h-2.5" /> {productFolder.name}
                              </span>
                            )}
                            {productCats.map((cat: string) => (
                              <span key={cat} className="text-[10px] bg-blue-50 text-blue-700 border border-blue-200 rounded-md px-1.5 py-0.5 font-medium">
                                🏷️ {cat}
                              </span>
                            ))}
                            {product.brand && (
                              <span className="text-[10px] bg-violet-50 text-violet-700 border border-violet-200 rounded-md px-1.5 py-0.5 font-medium">
                                🔖 {product.brand}
                              </span>
                            )}
                          </div>
                        )}
                        <div className="mt-1 flex items-baseline gap-2">
                          <span className="text-xl font-bold text-emerald-600">
                            {product.discountPrice || product.originalPrice || 0} د.ج
                          </span>
                          {product.discountPrice && product.originalPrice && (
                            <span className="text-sm text-muted-foreground line-through decoration-red-500/50">
                              {product.originalPrice}
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="space-y-2 bg-muted/40 p-3 rounded-xl border border-border/50">
                        <div className="flex justify-between text-sm font-medium">
                          <span className="text-muted-foreground">المخزون</span>
                          <span className={product.stockQuantity <= product.lowStockThreshold ? "text-red-500" : ""}>
                            {product.stockQuantity} قطعة
                          </span>
                        </div>
                        <div className="h-1.5 w-full bg-border rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${progressColor} transition-all duration-500`} style={{ width: `${Math.max(2, stockPercent)}%` }} />
                        </div>
                        {product.stockQuantity <= product.lowStockThreshold && product.stockQuantity > 0 && (
                          <p className="text-[10px] text-amber-600 font-bold text-center">⚠️ مخزون منخفض</p>
                        )}

                        <div className="flex items-center justify-between gap-2 pt-2 border-t border-border/50 mt-2">
                          <Button variant="outline" size="icon" className="h-7 w-7 rounded-lg" disabled={product.stockQuantity <= 0 || stockMut.isPending} onClick={() => handleStock(product.id, -1)}>
                            <Minus className="w-3 h-3" />
                          </Button>
                          <span className="text-xs font-mono font-medium">{product.stockQuantity}</span>
                          <Button variant="outline" size="icon" className="h-7 w-7 rounded-lg" disabled={stockMut.isPending} onClick={() => handleStock(product.id, 1)}>
                            <Plus className="w-3 h-3" />
                          </Button>
                        </div>
                      </div>

                      <div className="flex gap-2 pt-2">
                        <Button variant="outline" className="flex-1 rounded-xl h-9" onClick={() => openDialog(product)}>
                          <Edit className="w-4 h-4 me-1" /> تعديل
                        </Button>
                        <Button variant="ghost" className="rounded-xl h-9 w-9 p-0 text-red-500 hover:text-red-600 hover:bg-red-50" onClick={() => {
                          if (confirm("هل أنت متأكد من حذف المنتج؟")) {
                            deleteMut.mutate({ id: product.id }, {
                              onSuccess: () => {
                                queryClient.invalidateQueries({ queryKey: ["/api/products"] });
                                toast({ title: "تم الحذف" });
                              }
                            });
                          }
                        }}>
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      )}

      {/* ===== PRODUCT DIALOG ===== */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent dir="rtl" className="max-w-3xl max-h-[90vh] overflow-y-auto rounded-2xl p-0 gap-0">
          <div className="p-6 bg-muted/40 border-b border-border/50 sticky top-0 z-10">
            <DialogTitle className="text-xl">{editingProduct ? "تعديل منتج" : "إضافة منتج جديد"}</DialogTitle>
          </div>

          <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-5">
              <div className="space-y-1.5">
                <Label>الاسم <span className="text-red-500">*</span></Label>
                <Input value={formData.name} onChange={e => setFormData({...formData, name: e.target.value})} className="bg-muted/40 h-11" placeholder="اسم المنتج..." />
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label>الوصف</Label>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleSummarize}
                    disabled={isSummarizing || !formData.description.trim()}
                    className="h-7 gap-1.5 text-xs border-violet-200 text-violet-700 hover:bg-violet-50 hover:text-violet-800 disabled:opacity-40"
                  >
                    {isSummarizing
                      ? <><Loader2 className="h-3 w-3 animate-spin" />جاري التلخيص...</>
                      : <><Sparkles className="h-3 w-3" />تلخيص بالذكاء الاصطناعي</>
                    }
                  </Button>
                </div>
                <Textarea
                  value={formData.description}
                  onChange={e => setFormData({...formData, description: e.target.value})}
                  className="bg-muted/40 min-h-[100px]"
                  placeholder="الصق الوصف الكامل هنا ثم اضغط &quot;تلخيص&quot; لاستخراج المواصفات تلقائياً..."
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>السعر الأصلي</Label>
                  <Input type="number" value={formData.originalPrice} onChange={e => setFormData({...formData, originalPrice: e.target.value})} className="bg-muted/40 h-11 font-mono text-left" dir="ltr" placeholder="0.00" />
                </div>
                <div className="space-y-1.5">
                  <Label>سعر التخفيض</Label>
                  <Input type="number" value={formData.discountPrice} onChange={e => setFormData({...formData, discountPrice: e.target.value})} className="bg-muted/40 h-11 font-mono text-left" dir="ltr" placeholder="0.00" />
                </div>
              </div>
              {discountPercent > 0 && (
                <div className="text-sm text-emerald-600 font-medium flex items-center gap-1">
                  <Tag className="w-4 h-4" /> نسبة التخفيض: {discountPercent}%
                </div>
              )}

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>الكمية الحالية</Label>
                  <Input type="number" value={formData.stockQuantity} onChange={e => setFormData({...formData, stockQuantity: e.target.value})} className="bg-muted/40 h-11 font-mono text-center" />
                </div>
                <div className="space-y-1.5">
                  <Label>تنبيه نقص المخزون عند</Label>
                  <Input type="number" value={formData.lowStockThreshold} onChange={e => setFormData({...formData, lowStockThreshold: e.target.value})} className="bg-muted/40 h-11 font-mono text-center" />
                </div>
              </div>

              <div className="space-y-1.5">
                <Label>حالة المنتج</Label>
                <Select value={formData.status} onValueChange={v => setFormData({...formData, status: v})}>
                  <SelectTrigger className="h-11 bg-muted/40"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="available">🟢 متاح (Available)</SelectItem>
                    <SelectItem value="out_of_stock">🔴 نفد (Out of Stock)</SelectItem>
                    <SelectItem value="paused">⏸️ موقوف (Paused)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="pt-2 border-t border-border/40 space-y-4">
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">معلومات الكتالوج (للتصفح في Messenger)</p>

                {/* Folder selector in form */}
                {allFolders.length > 0 && (
                  <div className="space-y-1.5">
                    <Label className="flex items-center gap-1.5">
                      <Folder className="w-3.5 h-3.5 text-amber-500" /> المجلد
                    </Label>
                    <Select value={selectedFolderId || "none"} onValueChange={v => setSelectedFolderId(v === "none" ? "" : v)}>
                      <SelectTrigger className="bg-card">
                        <SelectValue placeholder="اختر مجلداً..." />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">— بدون مجلد —</SelectItem>
                        {allFolders.map(f => (
                          <SelectItem key={f.id} value={String(f.id)}>
                            <Folder className="w-3 h-3 inline me-1 text-amber-500" />{f.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                {/* Category multi-select */}
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <Label>التصنيف <span className="text-xs text-muted-foreground font-normal">(اختر واحداً فقط)</span></Label>
                    <button
                      type="button"
                      onClick={() => { setCatsOpen(true); loadCategories(); }}
                      className="text-[11px] text-primary hover:underline"
                    >
                      + إدارة التصنيفات
                    </button>
                  </div>

                  {allCategories.length === 0 ? (
                    <div className="border border-dashed border-border rounded-xl p-3 text-center text-sm text-muted-foreground">
                      لا توجد تصنيفات — أضف تصنيفات أولاً
                    </div>
                  ) : (
                    <div className="border border-border rounded-xl bg-muted/40 p-3 space-y-2 max-h-52 overflow-y-auto">
                      {parentCategories.map(parent => {
                        const children = getChildren(parent.id);
                        return (
                          <div key={parent.id}>
                            {/* Parent category */}
                            <button
                              type="button"
                              onClick={() => selectCategory(parent)}
                              className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm font-medium transition-colors text-right ${
                                selectedCategory === buildCategoryPath(parent.id)
                                  ? "bg-primary/10 text-primary border border-primary/20"
                                  : "hover:bg-muted text-foreground/80"
                              }`}
                            >
                              {selectedCategory === buildCategoryPath(parent.id)
                                ? <Check className="w-3.5 h-3.5 shrink-0 text-primary" />
                                : <Folder className="w-3.5 h-3.5 shrink-0 text-muted-foreground/70" />
                              }
                              <span className="flex-1">{parent.name}</span>
                            </button>
                            {/* Sub-categories */}
                            {children.map(child => (
                              <button
                                key={child.id}
                                type="button"
                                onClick={() => selectCategory(child)}
                                className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm transition-colors text-right mr-4 ${
                                  selectedCategory === buildCategoryPath(child.id)
                                    ? "bg-primary/10 text-primary border border-primary/20"
                                    : "hover:bg-muted text-muted-foreground"
                                }`}
                              >
                                <ChevronRight className="w-3 h-3 shrink-0 text-muted-foreground/40" />
                                {selectedCategory === buildCategoryPath(child.id)
                                  ? <Check className="w-3.5 h-3.5 shrink-0 text-primary" />
                                  : <Tag className="w-3.5 h-3.5 shrink-0 text-muted-foreground/70" />
                                }
                                <span className="flex-1">{child.name}</span>
                              </button>
                            ))}
                          </div>
                        );
                      })}
                      {/* Categories with no parent (orphaned children shown as root) */}
                      {allCategories.filter(c => c.parentId && !allCategories.find(p => p.id === c.parentId)).map(orphan => (
                        <button
                          key={orphan.id}
                          type="button"
                          onClick={() => selectCategory(orphan)}
                          className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm transition-colors text-right ${
                            selectedCategory === buildCategoryPath(orphan.id)
                              ? "bg-primary/10 text-primary border border-primary/20"
                              : "hover:bg-muted text-foreground/80"
                          }`}
                        >
                          {selectedCategory === buildCategoryPath(orphan.id)
                            ? <Check className="w-3.5 h-3.5 shrink-0 text-primary" />
                            : <Tag className="w-3.5 h-3.5 shrink-0 text-muted-foreground/70" />
                          }
                          <span className="flex-1">{orphan.name}</span>
                        </button>
                      ))}
                    </div>
                  )}

                  {selectedCategory && (
                    <div className="flex flex-wrap gap-1 pt-1">
                      <span className="inline-flex items-center gap-1 text-[11px] bg-primary/10 text-primary border border-primary/20 rounded-full px-2 py-0.5 font-medium">
                        {selectedCategory}
                        <button type="button" onClick={() => setSelectedCategory("")} className="hover:text-red-500">
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <Label>العلامة التجارية (Brand)</Label>
                    <Input value={formData.brand} onChange={e => setFormData({...formData, brand: e.target.value})} className="bg-muted/40 h-10" placeholder="مثال: Samsung, Apple..." />
                  </div>
                  <div className="space-y-1.5">
                    <Label>النوع (Type)</Label>
                    <Input value={formData.itemType} onChange={e => setFormData({...formData, itemType: e.target.value})} className="bg-muted/40 h-10" placeholder="مثال: smartphone, laptop..." />
                  </div>
                </div>

                <div className="space-y-1.5">
                  <Label>رابط خارجي (External URL)</Label>
                  <Input value={formData.externalUrl} onChange={e => setFormData({...formData, externalUrl: e.target.value})} className="bg-muted/40 h-10" dir="ltr" placeholder="https://..." />
                </div>
              </div>
            </div>

            {/* Image Upload Area */}
            <div className="space-y-4">
              <Label>صور المنتج (الحد الأقصى 5)</Label>
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-colors ${isDragActive ? 'border-primary bg-primary/5' : 'border-border hover:bg-muted/40'}`}
              >
                <input {...getInputProps()} />
                <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mx-auto mb-3">
                  <ImageIcon className="w-6 h-6 text-muted-foreground/70" />
                </div>
                <p className="font-medium text-sm">اسحب وأفلت الصور هنا أو اضغط للاختيار</p>
                <p className="text-xs text-muted-foreground mt-1">PNG, JPG حتى 5 ميجابايت</p>
              </div>

              <div className="grid grid-cols-3 gap-3">
                {existingImages.map((url, idx) => (
                  <div key={`ex-${idx}`} className={`relative aspect-square rounded-xl overflow-hidden group border-2 ${mainImageIndex === idx ? 'border-primary shadow-md' : 'border-transparent'}`}>
                    <img src={url} alt="img" className="w-full h-full object-cover" />
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                      <Button size="icon" variant="ghost" className="h-7 w-7 text-white hover:text-white hover:bg-card/20" onClick={(e) => { e.stopPropagation(); setMain(idx); }}>
                        <Star className={`w-4 h-4 ${mainImageIndex === idx ? 'fill-yellow-400 text-yellow-400' : ''}`} />
                      </Button>
                      <Button size="icon" variant="ghost" className="h-7 w-7 text-white hover:text-red-400 hover:bg-card/20" onClick={(e) => { e.stopPropagation(); removeExisting(idx); }}>
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                ))}
                {files.map((file, fileIdx) => {
                  const globalIdx = existingImages.length + fileIdx;
                  return (
                    <div key={`new-${fileIdx}`} className={`relative aspect-square rounded-xl overflow-hidden group border-2 ${mainImageIndex === globalIdx ? 'border-primary shadow-md' : 'border-transparent'}`}>
                      <img src={URL.createObjectURL(file)} alt="img" className="w-full h-full object-cover" />
                      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                        <Button size="icon" variant="ghost" className="h-7 w-7 text-white hover:text-white hover:bg-card/20" onClick={(e) => { e.stopPropagation(); setMain(globalIdx); }}>
                          <Star className={`w-4 h-4 ${mainImageIndex === globalIdx ? 'fill-yellow-400 text-yellow-400' : ''}`} />
                        </Button>
                        <Button size="icon" variant="ghost" className="h-7 w-7 text-white hover:text-red-400 hover:bg-card/20" onClick={(e) => { e.stopPropagation(); removeFile(fileIdx); }}>
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="p-6 bg-muted/40 border-t border-border/50 sticky bottom-0 z-10 flex gap-3 justify-end">
            <Button variant="outline" onClick={() => setDialogOpen(false)} className="rounded-xl h-11 px-6">إلغاء</Button>
            <Button onClick={handleSave} disabled={isSaving} className="rounded-xl h-11 px-8 shadow-lg shadow-primary/20">
              {isSaving ? <><Loader2 className="w-4 h-4 me-2 animate-spin" /> جاري الحفظ...</> : "حفظ المنتج"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* ===== CATEGORIES MANAGEMENT DIALOG ===== */}
      <Dialog open={catsOpen} onOpenChange={setCatsOpen}>
        <DialogContent dir="rtl" className="max-w-xl max-h-[85vh] overflow-y-auto rounded-2xl p-0 gap-0">
          <div className="p-5 bg-muted/40 border-b border-border/50 sticky top-0 z-10">
            <DialogTitle className="text-xl flex items-center gap-2">
              <FolderOpen className="w-5 h-5 text-primary" /> إدارة التصنيفات
            </DialogTitle>
            <p className="text-sm text-muted-foreground mt-0.5">أضف وعدّل وأحذف التصنيفات للمنتجات</p>
          </div>

          <div className="p-5 space-y-5">
            {/* Add new category */}
            <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 space-y-3">
              <p className="text-sm font-semibold text-blue-800 flex items-center gap-1.5">
                <FolderPlus className="w-4 h-4" /> إضافة تصنيف جديد
              </p>
              <Input
                value={newCatName}
                onChange={e => setNewCatName(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleAddCategory()}
                placeholder="اسم التصنيف..."
                className="bg-card h-10"
              />
              <Select value={newCatParent} onValueChange={setNewCatParent}>
                <SelectTrigger className="h-10 bg-card">
                  <SelectValue placeholder="تصنيف أب (اختياري)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">— بدون تصنيف أب —</SelectItem>
                  {buildFlatCatOptions().map(opt => (
                    <SelectItem key={opt.id} value={String(opt.id)}>
                      {"\u00A0\u00A0\u00A0\u00A0".repeat(opt.depth)}{opt.depth > 0 ? "└ " : ""}{opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                onClick={handleAddCategory}
                disabled={!newCatName.trim() || catSaving}
                className="w-full h-10 rounded-xl"
              >
                {catSaving ? <Loader2 className="w-4 h-4 animate-spin" /> : <><Plus className="w-4 h-4 me-1" /> إضافة</>}
              </Button>
            </div>

            {/* List categories */}
            {catsLoading ? (
              <div className="text-center py-6 text-muted-foreground">
                <Loader2 className="w-5 h-5 animate-spin mx-auto" />
              </div>
            ) : allCategories.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground border border-dashed border-border rounded-xl">
                <Folder className="w-8 h-8 mx-auto mb-2 opacity-40" />
                <p className="text-sm">لا توجد تصنيفات بعد</p>
              </div>
            ) : (
              <div className="space-y-2">
                {(() => {
                  const renderNode = (node: ProductCategory, depth: number): React.ReactNode => {
                    const nodeChildren = getChildren(node.id);
                    const isRoot = depth === 0;
                    const indent = depth * 16;
                    return (
                      <div key={node.id}>
                        <div
                          className={`flex items-center gap-2 p-3 ${isRoot ? "bg-card" : "bg-muted/30"} ${depth > 0 ? "border-t border-border/50" : ""}`}
                          style={{ paddingRight: `${12 + indent}px` }}
                        >
                          {editingCat?.id === node.id ? (
                            <div className="flex-1 flex items-center gap-2">
                              <Input
                                value={editCatName}
                                onChange={e => setEditCatName(e.target.value)}
                                className="h-8 flex-1"
                                autoFocus
                              />
                              <Select value={editCatParent} onValueChange={setEditCatParent}>
                                <SelectTrigger className="h-8 w-40">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="none">— بدون أب —</SelectItem>
                                  {buildFlatCatOptions(node.id).map(opt => (
                                    <SelectItem key={opt.id} value={String(opt.id)}>
                                      {"\u00A0\u00A0\u00A0\u00A0".repeat(opt.depth)}{opt.depth > 0 ? "└ " : ""}{opt.label}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                              <Button size="sm" className="h-8 px-3" onClick={handleEditCatSave} disabled={catSaving}>
                                {catSaving ? <Loader2 className="w-3 h-3 animate-spin" /> : "حفظ"}
                              </Button>
                              <Button size="sm" variant="ghost" className="h-8 px-2" onClick={() => setEditingCat(null)}>
                                <X className="w-3 h-3" />
                              </Button>
                            </div>
                          ) : (
                            <>
                              {!isRoot && <ChevronRight className="w-3 h-3 text-muted-foreground/40 shrink-0" />}
                              {nodeChildren.length > 0
                                ? <Folder className={`w-4 h-4 shrink-0 ${isRoot ? "text-amber-500" : "text-amber-400"}`} />
                                : <Tag className="w-3.5 h-3.5 text-blue-400 shrink-0" />
                              }
                              <span className={`flex-1 text-sm ${isRoot ? "font-medium" : "text-foreground/80"}`}>{node.name}</span>
                              {nodeChildren.length > 0 && (
                                <span className="text-[10px] text-muted-foreground bg-muted px-1.5 py-0.5 rounded-full">{nodeChildren.length} فرعي</span>
                              )}
                              <Button
                                size="icon" variant="ghost" className="h-7 w-7 text-muted-foreground/70 hover:text-primary"
                                onClick={() => { setEditingCat(node); setEditCatName(node.name); setEditCatParent(node.parentId ? String(node.parentId) : "none"); }}
                              >
                                <Edit className="w-3.5 h-3.5" />
                              </Button>
                              <Button
                                size="icon" variant="ghost" className="h-7 w-7 text-muted-foreground/70 hover:text-red-500"
                                onClick={() => handleDeleteCat(node.id)}
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </Button>
                            </>
                          )}
                        </div>
                        {nodeChildren.map(child => renderNode(child, depth + 1))}
                      </div>
                    );
                  };
                  return parentCategories.map(root => (
                    <div key={root.id} className="border border-border rounded-xl overflow-hidden">
                      {renderNode(root, 0)}
                    </div>
                  ));
                })()}
              </div>
            )}
          </div>

          <div className="p-5 bg-muted/40 border-t border-border/50 sticky bottom-0 flex justify-end">
            <Button variant="outline" onClick={() => setCatsOpen(false)} className="rounded-xl h-10 px-6">إغلاق</Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* ===== FOLDERS MANAGEMENT DIALOG ===== */}
      <Dialog open={foldersOpen} onOpenChange={open => { setFoldersOpen(open); if (!open) setEditingFolder(null); }}>
        <DialogContent dir="rtl" className="max-w-md max-h-[80vh] overflow-y-auto rounded-2xl p-0 gap-0">
          <div className="p-5 bg-amber-50/60 border-b border-amber-100 sticky top-0 z-10">
            <DialogTitle className="text-xl flex items-center gap-2">
              <FolderPlus className="w-5 h-5 text-amber-600" /> إدارة المجلدات
            </DialogTitle>
            <p className="text-sm text-muted-foreground mt-0.5">نظّم منتجاتك في مجلدات لسهولة التصفح</p>
          </div>

          <div className="p-5 space-y-5">
            {/* Add new folder */}
            <div className="bg-amber-50 border border-amber-100 rounded-xl p-4 space-y-3">
              <p className="text-sm font-semibold text-amber-800 flex items-center gap-1.5">
                <FolderPlus className="w-4 h-4" /> مجلد جديد
              </p>
              <div className="flex gap-2">
                <Input
                  value={newFolderName}
                  onChange={e => setNewFolderName(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleCreateFolder()}
                  placeholder="اسم المجلد..."
                  className="bg-card h-10"
                />
                <Button
                  onClick={handleCreateFolder}
                  disabled={!newFolderName.trim() || folderSaving}
                  className="h-10 px-4 rounded-xl bg-amber-500 hover:bg-amber-600 text-white"
                >
                  {folderSaving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
                </Button>
              </div>
            </div>

            {/* Folders list */}
            {allFolders.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground text-sm">
                <Folder className="w-10 h-10 mx-auto mb-2 opacity-20" />
                لا توجد مجلدات بعد
              </div>
            ) : (
              <div className="space-y-2">
                {allFolders.map(folder => {
                  const count = (products as ProductExt[]).filter(p => p.folderId === folder.id).length;
                  const isEditing = editingFolder?.id === folder.id;
                  return (
                    <div key={folder.id} className="flex items-center gap-2 p-3 bg-card border border-border rounded-xl group">
                      <Folder className="w-5 h-5 text-amber-500 shrink-0" />
                      {isEditing ? (
                        <div className="flex flex-1 gap-2">
                          <Input
                            autoFocus
                            value={editFolderName}
                            onChange={e => setEditFolderName(e.target.value)}
                            onKeyDown={e => { if (e.key === "Enter") handleRenameFolder(); if (e.key === "Escape") setEditingFolder(null); }}
                            className="h-8 text-sm"
                          />
                          <Button size="sm" onClick={handleRenameFolder} disabled={folderSaving} className="h-8 px-3 bg-amber-500 hover:bg-amber-600 text-white">
                            {folderSaving ? <Loader2 className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
                          </Button>
                          <Button size="sm" variant="ghost" onClick={() => setEditingFolder(null)} className="h-8 px-2">
                            <X className="w-3 h-3" />
                          </Button>
                        </div>
                      ) : (
                        <>
                          <span className="flex-1 font-medium text-sm">{folder.name}</span>
                          <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded-full">{count} منتج</span>
                          <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            <Button size="icon" variant="ghost" className="w-7 h-7 text-amber-600 hover:bg-amber-50"
                              onClick={() => { setEditingFolder(folder); setEditFolderName(folder.name); }}>
                              <Pencil className="w-3.5 h-3.5" />
                            </Button>
                            <Button size="icon" variant="ghost" className="w-7 h-7 text-destructive hover:bg-destructive/10"
                              onClick={() => handleDeleteFolder(folder.id)}>
                              <Trash2 className="w-3.5 h-3.5" />
                            </Button>
                          </div>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="p-4 bg-muted/40 border-t border-border/50 sticky bottom-0 flex justify-end">
            <Button variant="outline" onClick={() => setFoldersOpen(false)} className="rounded-xl h-10 px-6">إغلاق</Button>
          </div>
        </DialogContent>
      </Dialog>
    </motion.div>
  );
}
