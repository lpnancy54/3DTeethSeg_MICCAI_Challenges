#!/usr/bin/env python3
"""GUI locale pour segmenter et étiqueter des scans STL/OBJ de mâchoires.

Ce script fournit :
- un mode GUI (Tkinter) pour utilisateurs non techniques ;
- un mode CLI pour automatisation/validation.

La segmentation proposée est une baseline géométrique (non deep learning).
"""

from __future__ import annotations

import argparse
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import trimesh

if not hasattr(np, "product"):
    np.product = np.prod

UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


def _kmeans(points: np.ndarray, k: int, n_iter: int = 30, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """K-means léger en NumPy (sans dépendance sklearn)."""
    rng = np.random.default_rng(seed)
    if k <= 1:
        center = points.mean(axis=0, keepdims=True)
        labels = np.zeros(points.shape[0], dtype=np.int32)
        return labels, center

    initial_idx = rng.choice(points.shape[0], size=min(k, points.shape[0]), replace=False)
    centers = points[initial_idx]

    if centers.shape[0] < k:
        pad = np.repeat(centers[-1][None, :], k - centers.shape[0], axis=0)
        centers = np.vstack([centers, pad])

    labels = np.zeros(points.shape[0], dtype=np.int32)
    for _ in range(n_iter):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(new_labels, labels):
            break

        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = points[mask].mean(axis=0)

    return labels, centers


def _label_order_for_jaw(jaw: str, n_teeth: int) -> list[int]:
    jaw = jaw.lower().strip()
    if jaw == "upper":
        return UPPER_FDI[:n_teeth]
    if jaw == "lower":
        return LOWER_FDI[:n_teeth]
    raise ValueError("La mâchoire doit être 'upper' ou 'lower'.")


def segment_mesh_vertices(vertices: np.ndarray, jaw: str, n_teeth: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Retourne labels FDI et instances pour chaque sommet du mesh."""
    n_points = vertices.shape[0]
    n_teeth = int(max(1, min(16, n_teeth)))

    xy = vertices[:, :2]
    cluster_ids, centers = _kmeans(xy, n_teeth)

    order = np.argsort(centers[:, 0])
    fdi_list = _label_order_for_jaw(jaw, n_teeth)

    labels = np.zeros(n_points, dtype=np.int32)
    instances = np.zeros(n_points, dtype=np.int32)

    distances = np.linalg.norm(xy - centers[cluster_ids], axis=1)
    gingiva_threshold = np.quantile(distances, 0.70)

    instance_id = 1
    for ordered_pos, cluster_id in enumerate(order):
        mask = cluster_ids == cluster_id
        tooth_mask = mask & (distances <= gingiva_threshold)

        labels[tooth_mask] = fdi_list[ordered_pos]
        instances[tooth_mask] = instance_id
        instance_id += 1

    return labels, instances


def _build_vertex_colors(instances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(labels), 4), dtype=np.uint8)
    colors[:, 3] = 255
    colors[labels == 0, :3] = np.array([170, 170, 170], dtype=np.uint8)

    unique_instances = [i for i in np.unique(instances) if i != 0]
    rng = np.random.default_rng(42)
    palette = rng.integers(30, 255, size=(max(1, len(unique_instances)), 3), dtype=np.uint8)
    for idx, inst in enumerate(unique_instances):
        colors[instances == inst, :3] = palette[idx]

    return colors


def _compute_prediction(input_path: str, jaw: str, n_teeth: int = 14) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray, dict]:
    mesh = trimesh.load(input_path, process=False)
    if not hasattr(mesh, "vertices"):
        raise ValueError("Le fichier chargé ne contient pas de vertices exploitables.")

    labels, instances = segment_mesh_vertices(mesh.vertices, jaw=jaw, n_teeth=n_teeth)
    result = {
        "id_patient": os.path.splitext(os.path.basename(input_path))[0],
        "jaw": jaw,
        "labels": labels.tolist(),
        "instances": instances.tolist(),
    }
    return mesh, labels, instances, result


def _save_outputs(mesh: trimesh.Trimesh, labels: np.ndarray, instances: np.ndarray, result: dict, output_json: str, output_ply: str | None = None) -> None:
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f)

    if output_ply:
        colored = mesh.copy()
        colored.visual.vertex_colors = _build_vertex_colors(instances, labels)
        colored.export(output_ply)


def run_segmentation(input_path: str, jaw: str, output_json: str, output_ply: str | None = None, n_teeth: int = 14) -> dict:
    mesh, labels, instances, result = _compute_prediction(input_path=input_path, jaw=jaw, n_teeth=n_teeth)
    _save_outputs(mesh, labels, instances, result, output_json, output_ply)
    return result


class SegmentationGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("3DTeethSeg - GUI STL/OBJ")
        self.root.geometry("860x640")

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar(value="dental-labels.json")
        self.ply_var = tk.StringVar(value="segmented-preview.ply")
        self.jaw_var = tk.StringVar(value="upper")
        self.teeth_var = tk.IntVar(value=14)
        self.status_var = tk.StringVar(value="Prêt")

        self.current_mesh: trimesh.Trimesh | None = None
        self.current_labels: np.ndarray | None = None
        self.current_instances: np.ndarray | None = None
        self.current_result: dict | None = None

        self._build_layout()

    def _build_layout(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Scan STL/OBJ").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.input_var, width=80).grid(row=1, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Parcourir", command=self._select_input).grid(row=1, column=1, sticky="e")

        ttk.Label(frm, text="JSON de sortie").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(frm, textvariable=self.output_var, width=80).grid(row=3, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Choisir", command=self._select_output).grid(row=3, column=1, sticky="e")

        ttk.Label(frm, text="Preview mesh coloré (PLY, optionnel)").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(frm, textvariable=self.ply_var, width=80).grid(row=5, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Choisir", command=self._select_ply).grid(row=5, column=1, sticky="e")

        options = ttk.Frame(frm)
        options.grid(row=6, column=0, columnspan=2, sticky="we", pady=(14, 0))
        ttk.Label(options, text="Mâchoire").grid(row=0, column=0, sticky="w")
        ttk.Combobox(options, textvariable=self.jaw_var, values=["upper", "lower"], state="readonly", width=10).grid(row=0, column=1, sticky="w", padx=(8, 20))
        ttk.Label(options, text="Nb dents estimé").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(options, from_=1, to=16, textvariable=self.teeth_var, width=5).grid(row=0, column=3, sticky="w", padx=8)

        action_bar = ttk.Frame(frm)
        action_bar.grid(row=7, column=0, columnspan=2, sticky="we", pady=(16, 6))
        ttk.Button(action_bar, text="1) Prévisualiser", command=self._preview).grid(row=0, column=0, sticky="we", padx=(0, 6))
        ttk.Button(action_bar, text="2) Exporter", command=self._export).grid(row=0, column=1, sticky="we", padx=(6, 6))
        ttk.Button(action_bar, text="Ouvrir dossier export", command=self._open_export_folder).grid(row=0, column=2, sticky="we", padx=(6, 0))
        action_bar.columnconfigure(0, weight=1)
        action_bar.columnconfigure(1, weight=1)
        action_bar.columnconfigure(2, weight=1)

        self.canvas = tk.Canvas(frm, width=800, height=360, bg="white", highlightthickness=1, highlightbackground="#bfbfbf")
        self.canvas.grid(row=8, column=0, columnspan=2, sticky="nsew", pady=(8, 6))

        ttk.Label(frm, textvariable=self.status_var, foreground="navy").grid(row=9, column=0, columnspan=2, sticky="w")

        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(8, weight=1)
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(400, 180, text="Cliquez sur '1) Prévisualiser' pour voir la segmentation avant export.", fill="#6b6b6b", font=("Arial", 12))

    def _select_input(self):
        p = filedialog.askopenfilename(filetypes=[("Meshes", "*.stl *.obj"), ("Tous", "*.*")])
        if p:
            self.input_var.set(p)
            base_dir = os.path.dirname(p)
            base = os.path.splitext(os.path.basename(p))[0]
            self.output_var.set(os.path.join(base_dir, f"{base}_dental-labels.json"))
            self.ply_var.set(os.path.join(base_dir, f"{base}_segmented-preview.ply"))

    def _select_output(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if p:
            self.output_var.set(p)

    def _select_ply(self):
        p = filedialog.asksaveasfilename(defaultextension=".ply", filetypes=[("PLY", "*.ply")])
        if p:
            self.ply_var.set(p)

    def _validate_input(self) -> str | None:
        input_path = self.input_var.get().strip()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Erreur", "Veuillez choisir un fichier STL/OBJ valide.")
            return None
        return input_path

    def _preview(self):
        input_path = self._validate_input()
        if not input_path:
            return

        try:
            self.status_var.set("Prévisualisation en cours...")
            self.root.update_idletasks()
            mesh, labels, instances, result = _compute_prediction(
                input_path=input_path,
                jaw=self.jaw_var.get(),
                n_teeth=int(self.teeth_var.get()),
            )
            self.current_mesh = mesh
            self.current_labels = labels
            self.current_instances = instances
            self.current_result = result
            self._draw_segmentation_preview(mesh.vertices, labels, instances)
            self.status_var.set(
                f"Prévisualisation OK : {len(labels)} sommets, {len(set(instances.tolist())) - (1 if 0 in instances else 0)} instances."
            )
        except Exception as exc:
            self.status_var.set("Erreur pendant la prévisualisation.")
            messagebox.showerror("Erreur", str(exc))

    def _draw_segmentation_preview(self, vertices: np.ndarray, labels: np.ndarray, instances: np.ndarray):
        self.canvas.delete("all")
        width = int(self.canvas.winfo_width() or 800)
        height = int(self.canvas.winfo_height() or 360)

        sample_size = min(6000, len(vertices))
        idx = np.random.default_rng(0).choice(len(vertices), size=sample_size, replace=False)
        sampled = vertices[idx]
        sampled_labels = labels[idx]
        sampled_instances = instances[idx]

        x = sampled[:, 0]
        y = sampled[:, 1]
        x_span = max(1e-6, float(x.max() - x.min()))
        y_span = max(1e-6, float(y.max() - y.min()))

        x_canvas = 20 + ((x - x.min()) / x_span) * (width - 40)
        y_canvas = 20 + ((y - y.min()) / y_span) * (height - 40)
        y_canvas = height - y_canvas

        color_cache: dict[int, str] = {0: "#aaaaaa"}
        rng = np.random.default_rng(42)
        unique_inst = [int(v) for v in np.unique(sampled_instances) if v != 0]
        for inst in unique_inst:
            rgb = rng.integers(30, 255, size=3)
            color_cache[inst] = f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"

        for px, py, inst in zip(x_canvas, y_canvas, sampled_instances):
            c = color_cache.get(int(inst), "#aaaaaa")
            self.canvas.create_oval(px, py, px + 1.6, py + 1.6, fill=c, outline=c)

        gingiva_pct = float(np.mean(sampled_labels == 0) * 100.0)
        self.canvas.create_rectangle(8, 8, 440, 46, fill="#ffffff", outline="#d0d0d0")
        self.canvas.create_text(
            16,
            18,
            anchor="w",
            text=f"Prévisualisation 2D (projection XY) — {sample_size} points affichés",
            fill="#202020",
            font=("Arial", 10, "bold"),
        )
        self.canvas.create_text(
            16,
            34,
            anchor="w",
            text=f"Instances visibles: {len(unique_inst)} | Gingiva approx: {gingiva_pct:.1f}%",
            fill="#303030",
            font=("Arial", 9),
        )

    def _export(self):
        if self.current_result is None or self.current_mesh is None or self.current_labels is None or self.current_instances is None:
            messagebox.showwarning("Prévisualisation requise", "Veuillez d'abord cliquer sur '1) Prévisualiser'.")
            return

        output_json = self.output_var.get().strip()
        output_ply = self.ply_var.get().strip() or None
        if not output_json:
            messagebox.showerror("Erreur", "Veuillez renseigner un chemin JSON de sortie.")
            return

        try:
            _save_outputs(
                mesh=self.current_mesh,
                labels=self.current_labels,
                instances=self.current_instances,
                result=self.current_result,
                output_json=output_json,
                output_ply=output_ply,
            )
            saved = [output_json]
            if output_ply:
                saved.append(output_ply)
            self.status_var.set("Export terminé : " + " | ".join(saved))
            messagebox.showinfo("Export terminé", "Fichiers exportés :\n" + "\n".join(saved))
        except Exception as exc:
            self.status_var.set("Erreur pendant l'export.")
            messagebox.showerror("Erreur", str(exc))

    def _open_export_folder(self):
        output_json = self.output_var.get().strip()
        base_dir = os.path.dirname(output_json) if output_json else ""
        if not base_dir or not os.path.isdir(base_dir):
            messagebox.showwarning("Dossier introuvable", "Le dossier d'export n'existe pas encore.")
            return

        # Cross-platform best effort.
        try:
            if os.name == "nt":
                os.startfile(base_dir)
            elif os.uname().sysname == "Darwin":
                os.system(f'open "{base_dir}"')
            else:
                os.system(f'xdg-open "{base_dir}" >/dev/null 2>&1 &')
        except Exception:
            messagebox.showinfo("Dossier export", f"Dossier: {base_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI/CLI baseline pour segmentation 3DTeethSeg.")
    parser.add_argument("--input", help="Fichier STL/OBJ d'entrée")
    parser.add_argument("--jaw", choices=["upper", "lower"], help="Mâchoire: upper/lower")
    parser.add_argument("--output-json", default="dental-labels.json", help="Chemin du JSON de sortie")
    parser.add_argument("--output-ply", default="", help="Chemin du mesh PLY coloré (optionnel)")
    parser.add_argument("--n-teeth", type=int, default=14, help="Nombre de dents à estimer (1-16)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input:
        if not args.jaw:
            raise SystemExit("En mode CLI, --jaw est obligatoire.")
        run_segmentation(
            input_path=args.input,
            jaw=args.jaw,
            output_json=args.output_json,
            output_ply=args.output_ply or None,
            n_teeth=args.n_teeth,
        )
        print(f"OK - résultat écrit dans {args.output_json}")
        return

    root = tk.Tk()
    SegmentationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
