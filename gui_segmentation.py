#!/usr/bin/env python3
"""GUI locale pour segmenter et étiqueter des scans STL/OBJ de mâchoires."""

from __future__ import annotations

import argparse
import json
import os
import tkinter as tk
from collections import deque
from tkinter import filedialog, messagebox, ttk

import numpy as np
import trimesh

if not hasattr(np, "product"):
    np.product = np.prod

try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


def _label_order_for_jaw(jaw: str, n_teeth: int) -> list[int]:
    jaw = jaw.lower().strip()
    if jaw == "upper":
        return UPPER_FDI[:n_teeth]
    if jaw == "lower":
        return LOWER_FDI[:n_teeth]
    raise ValueError("La mâchoire doit être 'upper' ou 'lower'.")


def _kmeans_numpy(points: np.ndarray, k: int, n_iter: int = 35, seed: int = 13) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros((k, points.shape[1]), dtype=np.float32)

    k = max(1, min(k, n))
    centers = points[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int32)

    for _ in range(n_iter):
        d = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = points[m].mean(axis=0)
            else:
                centers[c] = points[rng.integers(0, n)]

    return labels, centers


def _pca_align(vertices: np.ndarray) -> np.ndarray:
    center = vertices.mean(axis=0)
    x = vertices - center
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    aligned = x @ vt.T
    return aligned


def _vertex_curvature_score(mesh: trimesh.Trimesh) -> np.ndarray:
    v = mesh.vertices
    neighbors = mesh.vertex_neighbors
    curv = np.zeros(len(v), dtype=np.float32)

    for i, nbs in enumerate(neighbors):
        if len(nbs) == 0:
            continue
        mean_nb = v[np.asarray(nbs)].mean(axis=0)
        curv[i] = np.linalg.norm(v[i] - mean_nb)

    lo, hi = np.quantile(curv, [0.02, 0.98])
    curv = np.clip(curv, lo, hi)
    curv = (curv - curv.min()) / max(1e-8, (curv.max() - curv.min()))
    return curv


def _largest_cc_mask(mesh: trimesh.Trimesh, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask

    neighbors = mesh.vertex_neighbors
    visited = np.zeros(len(mask), dtype=bool)
    best_component = []

    starts = np.where(mask)[0]
    for s in starts:
        if visited[s]:
            continue
        q = deque([int(s)])
        visited[s] = True
        comp = [int(s)]
        while q:
            u = q.popleft()
            for nb in neighbors[u]:
                nb = int(nb)
                if mask[nb] and not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
                    comp.append(nb)
        if len(comp) > len(best_component):
            best_component = comp

    out = np.zeros(len(mask), dtype=bool)
    out[np.asarray(best_component, dtype=np.int32)] = True
    return out


def segment_mesh_vertices(mesh: trimesh.Trimesh, jaw: str, n_teeth: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Segmentation heuristique plus stable:
    - détecte une bande dentaire externe (évite le palais/lingual)
    - découpe l'arcade selon l'angle PCA
    - extrait une composante connexe principale par dent
    """
    n_teeth = int(max(8, min(16, n_teeth)))
    vertices = mesh.vertices
    aligned = _pca_align(vertices)
    xy = aligned[:, :2]
    z = aligned[:, 2]

    r = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(xy[:, 1], xy[:, 0])

    # Scores géométriques
    r_n = (r - r.min()) / max(1e-8, (r.max() - r.min()))
    z_dev = np.abs(z - np.median(z))
    z_n = (z_dev - z_dev.min()) / max(1e-8, (z_dev.max() - z_dev.min()))
    c_n = _vertex_curvature_score(mesh)
    tooth_score = 0.50 * r_n + 0.30 * c_n + 0.20 * z_n

    # 1) Bande dentaire externe par secteur angulaire (adaptatif)
    n_angle_bins = 180
    bins = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
    angle_idx = np.clip(np.digitize(theta, bins) - 1, 0, n_angle_bins - 1)

    radial_thr = np.zeros(n_angle_bins, dtype=np.float32)
    for b in range(n_angle_bins):
        m = angle_idx == b
        if not m.any():
            radial_thr[b] = np.quantile(r, 0.70)
            continue
        # on garde la partie extérieure du bin
        radial_thr[b] = np.quantile(r[m], 0.62)

    outer_band = r >= radial_thr[angle_idx]

    # combine avec score dent adaptatif
    q = 0.58
    candidate = outer_band & (tooth_score >= np.quantile(tooth_score, q))
    while candidate.sum() < n_teeth * 220 and q > 0.30:
        q -= 0.04
        candidate = outer_band & (tooth_score >= np.quantile(tooth_score, q))

    # fallback: juste bande externe
    if candidate.sum() < n_teeth * 100:
        candidate = outer_band

    # 2) Arc dentaire utile (évite les zones hors arcade)
    th_c = theta[candidate]
    if len(th_c) < 200:
        th_c = theta
    t_min = np.quantile(th_c, 0.03)
    t_max = np.quantile(th_c, 0.97)
    if (t_max - t_min) < 0.8:
        t_min = np.quantile(theta, 0.05)
        t_max = np.quantile(theta, 0.95)

    # 3) Découpe de l'arc en dents (ordre FDI)
    tooth_bins = np.linspace(t_min, t_max, n_teeth + 1)
    sector = np.clip(np.digitize(theta, tooth_bins) - 1, 0, n_teeth - 1)

    labels = np.zeros(len(vertices), dtype=np.int32)
    instances = np.zeros(len(vertices), dtype=np.int32)

    fdi = _label_order_for_jaw(jaw, n_teeth)

    min_size = max(130, int(0.00035 * len(vertices)))

    for i in range(n_teeth):
        m = sector == i
        if not m.any():
            continue

        local = np.zeros(len(vertices), dtype=bool)

        # Zone dent: externe + score suffisant dans le secteur
        local_score = tooth_score[m]
        score_thr = np.quantile(local_score, 0.34)
        local[m] = outer_band[m] & (local_score >= score_thr)

        # Limite en distance radiale interne (anti-palais)
        r_loc = r[m]
        r_cut = np.quantile(r_loc, 0.36)
        radial_gate = np.zeros(len(vertices), dtype=bool)
        radial_gate[m] = r[m] >= r_cut
        local &= radial_gate

        local = _largest_cc_mask(mesh, local)

        if local.sum() < min_size:
            # fallback doux du secteur
            local = np.zeros(len(vertices), dtype=bool)
            local[m] = outer_band[m] & (r[m] >= np.quantile(r[m], 0.48))
            local = _largest_cc_mask(mesh, local)

        if local.sum() < min_size:
            continue

        instances[local] = i + 1
        labels[local] = fdi[i]

    # fallback global: si trop peu de dents, assigne secteur externe
    detected = len([x for x in np.unique(instances) if x != 0])
    if detected < max(6, n_teeth // 2):
        labels[:] = 0
        instances[:] = 0
        for i in range(n_teeth):
            m = sector == i
            local = np.zeros(len(vertices), dtype=bool)
            local[m] = outer_band[m] & (tooth_score[m] >= np.quantile(tooth_score[m], 0.22))
            local = _largest_cc_mask(mesh, local)
            if local.sum() < min_size:
                continue
            instances[local] = i + 1
            labels[local] = fdi[i]

    return labels, instances


def _instance_color_map(instances: np.ndarray) -> dict[int, np.ndarray]:
    cmap = {0: np.array([0.70, 0.70, 0.70, 1.0])}
    unique_instances = [int(i) for i in np.unique(instances) if i != 0]
    rng = np.random.default_rng(42)
    for inst in unique_instances:
        rgb = rng.random(3) * 0.75 + 0.2
        cmap[inst] = np.array([rgb[0], rgb[1], rgb[2], 1.0])
    return cmap


def _build_vertex_colors(instances: np.ndarray) -> np.ndarray:
    cmap = _instance_color_map(instances)
    out = np.zeros((len(instances), 4), dtype=np.uint8)
    for inst, rgba in cmap.items():
        out[instances == inst] = (rgba * 255).astype(np.uint8)
    return out


def _compute_prediction(input_path: str, jaw: str, n_teeth: int = 14) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray, dict]:
    mesh = trimesh.load(input_path, process=False)
    if not hasattr(mesh, "vertices"):
        raise ValueError("Le fichier chargé ne contient pas de vertices exploitables.")

    labels, instances = segment_mesh_vertices(mesh, jaw=jaw, n_teeth=n_teeth)
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
        colored.visual.vertex_colors = _build_vertex_colors(instances)
        colored.export(output_ply)


def run_segmentation(input_path: str, jaw: str, output_json: str, output_ply: str | None = None, n_teeth: int = 14) -> dict:
    mesh, labels, instances, result = _compute_prediction(input_path=input_path, jaw=jaw, n_teeth=n_teeth)
    _save_outputs(mesh, labels, instances, result, output_json, output_ply)
    return result


class SegmentationGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("3DTeethSeg - GUI STL/OBJ")
        self.root.geometry("980x760")

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

        self.preview_frame: ttk.Frame | None = None
        self.preview_canvas = None

        self._build_layout()

    def _build_layout(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Scan STL/OBJ").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.input_var, width=92).grid(row=1, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Parcourir", command=self._select_input).grid(row=1, column=1, sticky="e")

        ttk.Label(frm, text="JSON de sortie").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=self.output_var, width=92).grid(row=3, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Choisir", command=self._select_output).grid(row=3, column=1, sticky="e")

        ttk.Label(frm, text="Preview mesh coloré (PLY, optionnel)").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=self.ply_var, width=92).grid(row=5, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Choisir", command=self._select_ply).grid(row=5, column=1, sticky="e")

        options = ttk.Frame(frm)
        options.grid(row=6, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Label(options, text="Mâchoire").grid(row=0, column=0, sticky="w")
        ttk.Combobox(options, textvariable=self.jaw_var, values=["upper", "lower"], state="readonly", width=10).grid(row=0, column=1, sticky="w", padx=(8, 16))
        ttk.Label(options, text="Nb dents estimé").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(options, from_=8, to=16, textvariable=self.teeth_var, width=5).grid(row=0, column=3, sticky="w", padx=8)

        action_bar = ttk.Frame(frm)
        action_bar.grid(row=7, column=0, columnspan=2, sticky="we", pady=(12, 6))
        ttk.Button(action_bar, text="1) Prévisualiser", command=self._preview).grid(row=0, column=0, sticky="we", padx=(0, 6))
        ttk.Button(action_bar, text="2) Exporter", command=self._export).grid(row=0, column=1, sticky="we", padx=(6, 6))
        ttk.Button(action_bar, text="Ouvrir dossier export", command=self._open_export_folder).grid(row=0, column=2, sticky="we", padx=(6, 0))
        action_bar.columnconfigure(0, weight=1)
        action_bar.columnconfigure(1, weight=1)
        action_bar.columnconfigure(2, weight=1)

        self.preview_frame = ttk.Frame(frm)
        self.preview_frame.grid(row=8, column=0, columnspan=2, sticky="nsew", pady=(8, 6))

        ttk.Label(frm, textvariable=self.status_var, foreground="navy").grid(row=9, column=0, columnspan=2, sticky="w")

        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(8, weight=1)
        self._draw_placeholder()

    def _draw_placeholder(self):
        for child in self.preview_frame.winfo_children():
            child.destroy()
        lbl = ttk.Label(
            self.preview_frame,
            text="Cliquez sur '1) Prévisualiser' pour voir un rendu surfacique de la segmentation.",
            anchor="center",
        )
        lbl.pack(fill="both", expand=True)

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
            self._draw_segmentation_preview(mesh, instances)
            n_inst = len([i for i in np.unique(instances) if i != 0])
            self.status_var.set(f"Prévisualisation OK : {len(labels)} sommets, {n_inst} dents détectées.")
        except Exception as exc:
            self.status_var.set("Erreur pendant la prévisualisation.")
            messagebox.showerror("Erreur", str(exc))

    def _draw_segmentation_preview(self, mesh: trimesh.Trimesh, instances: np.ndarray):
        for child in self.preview_frame.winfo_children():
            child.destroy()

        if not MATPLOTLIB_AVAILABLE or len(mesh.faces) == 0:
            ttk.Label(self.preview_frame, text="Matplotlib indisponible, prévisualisation surfacique non disponible.").pack(fill="both", expand=True)
            return

        aligned = _pca_align(mesh.vertices)
        xy = aligned[:, :2]

        face_inst = np.zeros(len(mesh.faces), dtype=np.int32)
        for i, f in enumerate(mesh.faces):
            vals, counts = np.unique(instances[f], return_counts=True)
            face_inst[i] = vals[np.argmax(counts)]

        cmap = _instance_color_map(instances)
        face_colors = np.array([cmap[int(v)] for v in face_inst])

        tri = xy[mesh.faces]
        fig = Figure(figsize=(8.8, 5.2), dpi=100)
        ax = fig.add_subplot(111)
        poly = PolyCollection(tri, facecolors=face_colors, edgecolors="none", linewidths=0.0, antialiaseds=False)
        ax.add_collection(poly)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_title("Prévisualisation surfacique (projection 2D de la mâchoire)")
        ax.axis("off")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.preview_canvas = canvas

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
            saved = [output_json] + ([output_ply] if output_ply else [])
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
    parser.add_argument("--n-teeth", type=int, default=14, help="Nombre de dents à estimer (8-16)")
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
