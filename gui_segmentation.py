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

    # Clustering sur coordonnées XY pour découper l'arcade en régions dentaires.
    xy = vertices[:, :2]
    cluster_ids, centers = _kmeans(xy, n_teeth)

    # Ordonnancement droite->gauche en X pour mapper vers FDI.
    order = np.argsort(centers[:, 0])
    fdi_list = _label_order_for_jaw(jaw, n_teeth)

    labels = np.zeros(n_points, dtype=np.int32)
    instances = np.zeros(n_points, dtype=np.int32)

    # Proxy gingiva : sommets les plus éloignés du centre de cluster (30%).
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


def run_segmentation(input_path: str, jaw: str, output_json: str, output_ply: str | None = None, n_teeth: int = 14) -> dict:
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

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f)

    if output_ply:
        rng = np.random.default_rng(42)
        colors = np.zeros((len(labels), 4), dtype=np.uint8)
        colors[:, 3] = 255
        colors[labels == 0, :3] = np.array([170, 170, 170], dtype=np.uint8)

        unique_instances = [i for i in np.unique(instances) if i != 0]
        palette = rng.integers(30, 255, size=(max(1, len(unique_instances)), 3), dtype=np.uint8)
        for idx, inst in enumerate(unique_instances):
            colors[instances == inst, :3] = palette[idx]

        colored = mesh.copy()
        colored.visual.vertex_colors = colors
        colored.export(output_ply)

    return result


class SegmentationGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("3DTeethSeg - GUI STL/OBJ")
        self.root.geometry("720x380")

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar(value="dental-labels.json")
        self.ply_var = tk.StringVar(value="segmented-preview.ply")
        self.jaw_var = tk.StringVar(value="upper")
        self.teeth_var = tk.IntVar(value=14)
        self.status_var = tk.StringVar(value="Prêt")

        self._build_layout()

    def _build_layout(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Scan STL/OBJ").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.input_var, width=68).grid(row=1, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Parcourir", command=self._select_input).grid(row=1, column=1, sticky="e")

        ttk.Label(frm, text="JSON de sortie").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(frm, textvariable=self.output_var, width=68).grid(row=3, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Choisir", command=self._select_output).grid(row=3, column=1, sticky="e")

        ttk.Label(frm, text="Preview mesh coloré (PLY, optionnel)").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(frm, textvariable=self.ply_var, width=68).grid(row=5, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Choisir", command=self._select_ply).grid(row=5, column=1, sticky="e")

        options = ttk.Frame(frm)
        options.grid(row=6, column=0, columnspan=2, sticky="we", pady=(14, 0))
        ttk.Label(options, text="Mâchoire").grid(row=0, column=0, sticky="w")
        ttk.Combobox(options, textvariable=self.jaw_var, values=["upper", "lower"], state="readonly", width=10).grid(row=0, column=1, sticky="w", padx=(8, 20))
        ttk.Label(options, text="Nb dents estimé").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(options, from_=1, to=16, textvariable=self.teeth_var, width=5).grid(row=0, column=3, sticky="w", padx=8)

        ttk.Button(frm, text="Lancer la segmentation", command=self._run, style="Accent.TButton").grid(row=7, column=0, columnspan=2, sticky="we", pady=(20, 6))
        ttk.Label(frm, textvariable=self.status_var, foreground="navy").grid(row=8, column=0, columnspan=2, sticky="w")

        frm.columnconfigure(0, weight=1)

    def _select_input(self):
        p = filedialog.askopenfilename(filetypes=[("Meshes", "*.stl *.obj"), ("Tous", "*.*")])
        if p:
            self.input_var.set(p)
            base = os.path.splitext(os.path.basename(p))[0]
            self.output_var.set(f"{base}_dental-labels.json")
            self.ply_var.set(f"{base}_segmented-preview.ply")

    def _select_output(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if p:
            self.output_var.set(p)

    def _select_ply(self):
        p = filedialog.asksaveasfilename(defaultextension=".ply", filetypes=[("PLY", "*.ply")])
        if p:
            self.ply_var.set(p)

    def _run(self):
        input_path = self.input_var.get().strip()
        output_json = self.output_var.get().strip()
        output_ply = self.ply_var.get().strip() or None

        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Erreur", "Veuillez choisir un fichier STL/OBJ valide.")
            return

        try:
            self.status_var.set("Segmentation en cours...")
            self.root.update_idletasks()
            result = run_segmentation(
                input_path=input_path,
                jaw=self.jaw_var.get(),
                output_json=output_json,
                output_ply=output_ply,
                n_teeth=int(self.teeth_var.get()),
            )
            self.status_var.set(
                f"Terminé. {len(result['labels'])} sommets traités. JSON: {output_json}"
            )
            messagebox.showinfo("Succès", "Segmentation terminée.")
        except Exception as exc:
            self.status_var.set("Erreur pendant la segmentation.")
            messagebox.showerror("Erreur", str(exc))


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
