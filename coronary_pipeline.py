"""
coronary_pipeline.py — Coronary Artery Detection & Segmentation Pipeline

Pipeline łączy:
  1. Klasyfikator strony (right / left coronary) — lekka sieć ResNet18
  2. RF-DETR-Seg — detekcja + segmentacja 25 segmentów tętnic
  3. Heurystyki domenowe — filtrowanie niemożliwych detekcji

Użycie:
    # --- TRENING ---
    from coronary_pipeline import CoronaryPipeline
    pipeline = CoronaryPipeline(device="cuda")
    pipeline.train_side_classifier(arcade_root="./dataset", epochs=20)
    pipeline.train_rfdetr_seg(dataset_dir="./arcade_coco_dataset_multiclass_seg", epochs=50)

    # --- INFERENCJA ---
    pipeline = CoronaryPipeline.load("./checkpoints", device="cuda")
    result = pipeline.predict(image, threshold=0.5)
    # result.side → "right" | "left"
    # result.detections → sv.Detections (filtered)
    # result.raw_detections → sv.Detections (unfiltered)
"""

from __future__ import annotations

import logging
# Patch for Python 3.13 where logging.warn is removed
if not hasattr(logging, "warn"):
    logging.warn = logging.warning

import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

try:
    import supervision as sv
except ImportError:
    sv = None

try:
    from rfdetr import RFDETRSegLarge
except ImportError:
    RFDETRSegLarge = None

# ════════════════════════════════════════════════════════════════════
# 1. DOMAIN KNOWLEDGE — mapowania segmentów wieńcowych
# ════════════════════════════════════════════════════════════════════

# Mapowanie segment_name → class_id (1-indexed, ARCADE convention)
SEGMENT_NAME_TO_CLASS_ID: Dict[str, int] = {
    "1":   1,   # RCA proximal
    "2":   2,   # RCA mid
    "3":   3,   # RCA distal
    "4":   4,   # PDA (Posterior Descending Artery)
    "5":   5,   # LM (Left Main)
    "6":   6,   # LAD proximal
    "7":   7,   # LAD mid
    "8":   8,   # LAD apical
    "9":   9,   # D1 (First Diagonal)
    "9a":  10,  # D1 branch
    "10":  11,  # D2 (Second Diagonal)
    "10a": 12,  # D2 branch
    "11":  13,  # LCx proximal
    "12":  14,  # OM1 (First Obtuse Marginal)
    "12a": 15,  # OM1 branch
    "12b": 24,  # OM1 branch b
    "13":  16,  # LCx mid
    "14":  17,  # OM2 (Second Obtuse Marginal)
    "14a": 18,  # OM2 branch a
    "14b": 25,  # OM2 branch b
    "15":  19,  # LCx distal
    "16":  20,  # PLB (Posterior Left Ventricular Branch)
    "16a": 21,  # PLB branch a
    "16b": 22,  # PLB branch b
    "16c": 23,  # PLB branch c
}

CLASS_ID_TO_SEGMENT_NAME: Dict[int, str] = {v: k for k, v in SEGMENT_NAME_TO_CLASS_ID.items()}

# CLASS_ID_TO_LABEL: Dict[int, str] = {
#     1: "1 - RCA_prox",   2: "2 - RCA_mid",     3: "RCA_dist",      4: "PDA",
#     5: "LM",         6: "LAD_prox",    7: "LAD_mid",       8: "LAD_apical",
#     9: "D1",        10: "D1_9a",      11: "D2",           12: "D2_10a",
#     13: "LCx_prox", 14: "OM1_12",     15: "OM1_12a",      16: "LCx_mid",
#     17: "OM2_14",   18: "OM2_14a",    19: "LCx_dist",     20: "PLB_16",
#     21: "PLB_16a",  22: "PLB_16b",    23: "PLB_16c",      24: "OM1_12b",
#     25: "OM2_14b",
# }

# CLASS_ID_TO_LABEL: Dict[int, str] = {
#     1: "1 - RCA_prox",   2: "2 - RCA_mid",     3: "3 - RCA_dist",    4: "4 - PDA",
#     5: "5 - LM",         6: "6 - LAD_prox",    7: "7 - LAD_mid",     8: "8 - LAD_apical",
#     9: "9 - D1",        10: "10 - D1_9a",     11: "11 - D2",        12: "12 - D2_10a",
#     13: "13 - LCx_prox", 14: "14 - OM1_12",    15: "15 - OM1_12a",   16: "16 - LCx_mid",
#     17: "17 - OM2_14",   18: "18 - OM2_14a",   19: "19 - LCx_dist",  20: "20 - PLB_16",
#     21: "21 - PLB_16a",  22: "22 - PLB_16b",   23: "23 - PLB_16c",   24: "24 - OM1_12b",
#     25: "25 - OM2_14b",
# }

CLASS_ID_TO_LABEL: Dict[int, str] = {
    1: "1: RCA prox",
    2: "2: RCA mid",
    3: "3: RCA dist",
    4: "4: PDA",
    5: "5: LM",
    6: "6: LAD prox",
    7: "7: LAD mid",
    8: "8: LAD apical",
    9: "9: D1",
    10: "10: D1 branch (9a)",
    11: "11: D2",
    12: "12: D2 branch (10a)",
    13: "13: LCx prox (11)",
    14: "14: OM1 (12)",
    15: "15: OM1 branch (12a)",
    16: "16: LCx mid (13)",
    17: "17: OM2 (14)",
    18: "18: OM2 branch (14a)",
    19: "19: LCx dist (15)",
    20: "20: PLB (16)",
    21: "21: PLB branch (16a)",
    22: "22: PLB branch (16b)",
    23: "23: PLB branch (16c)",
    24: "24: OM1 branch (12b)",
    25: "25: OM2 branch (14b)",
    26: "26: stenosis",
}

# ── Definicje stron ──────────────────────────────────────────────
RIGHT_SEGMENT_NAMES: Set[str] = {"1", "2", "3", "4", "16a", "16b", "16c"}
LEFT_SEGMENT_NAMES: Set[str] = {
    "5", "6", "7", "8", "9", "9a", "10", "10a",
    "11", "12", "12a", "12b", "13", "14", "14a", "14b", "15", "16",
}

RIGHT_CLASS_IDS: Set[int] = {SEGMENT_NAME_TO_CLASS_ID[s] for s in RIGHT_SEGMENT_NAMES}
LEFT_CLASS_IDS: Set[int] = {SEGMENT_NAME_TO_CLASS_ID[s] for s in LEFT_SEGMENT_NAMES}

# ── Anatomiczna adjacency — które segmenty są sąsiadami ─────────
# Używane do walidacji: jeśli wykryto segment X, jego sąsiedzi są bardziej prawdopodobni
ANATOMICAL_ADJACENCY: Dict[int, Set[int]] = {
    # RIGHT CORONARY
    1:  {2},           # RCA prox → RCA mid
    2:  {1, 3},        # RCA mid → RCA prox, RCA dist
    3:  {2, 4},        # RCA dist → RCA mid, PDA
    4:  {3},           # PDA → RCA dist
    21: {3, 22},       # PLB 16a → RCA dist, PLB 16b
    22: {21, 23},      # PLB 16b → PLB 16a, PLB 16c
    23: {22},          # PLB 16c → PLB 16b
    # LEFT CORONARY
    5:  {6, 13},       # LM → LAD prox, LCx prox
    6:  {5, 7, 9},     # LAD prox → LM, LAD mid, D1
    7:  {6, 8, 11},    # LAD mid → LAD prox, LAD apical, D2
    8:  {7},           # LAD apical → LAD mid
    9:  {6, 10},       # D1 → LAD prox, D1 branch
    10: {9},           # D1 branch 9a → D1
    11: {7, 12},       # D2 → LAD mid, D2 branch
    12: {11},          # D2 branch 10a → D2
    13: {5, 14, 16},   # LCx prox → LM, OM1, LCx mid
    14: {13, 15, 24},  # OM1 → LCx prox, OM1 12a, OM1 12b
    15: {14},          # OM1 12a → OM1
    24: {14},          # OM1 12b → OM1
    16: {13, 17, 19},  # LCx mid → LCx prox, OM2, LCx dist
    17: {16, 18, 25},  # OM2 → LCx mid, OM2 14a, OM2 14b
    18: {17},          # OM2 14a → OM2
    25: {17},          # OM2 14b → OM2
    19: {16, 20},      # LCx dist → LCx mid, PLB
    20: {19},          # PLB 16 → LCx dist
}


def distinguish_side(segments: List[str]) -> str:
    """Określ stronę na podstawie nazw wykrytych segmentów."""
    return "right" if any(seg in segments for seg in RIGHT_SEGMENT_NAMES) else "left"


def distinguish_side_from_class_ids(class_ids: List[int]) -> str:
    """Określ stronę na podstawie wykrytych class_id."""
    return "right" if any(cid in RIGHT_CLASS_IDS for cid in class_ids) else "left"


# ════════════════════════════════════════════════════════════════════
# 2. SIDE CLASSIFIER — lekki klasyfikator prawe / lewe naczynia
# ════════════════════════════════════════════════════════════════════

def SideClassifierNet(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """
    Creates a ResNet18 model compatible with the architecture used in Prefect training.
    
    Architecture:
      ResNet18 (torchvision)
      fc layer replaced with: Sequential(Dropout(0.5), Linear(in_features, num_classes))
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    return model

class SideClassificationDataset(Dataset):
    """
    Dataset do treningu klasyfikatora strony.

    Etykieta = 0 (right) lub 1 (left) — wyznaczana automatycznie
    z adnotacji COCO na podstawie segmentów obecnych w obrazie.
    """

    def __init__(
        self,
        coco_json: str,
        images_dir: str,
        transform: Optional[Any] = None,
    ):
        with open(coco_json, "r") as f:
            coco = json.load(f)

        # Zbuduj mapę category_id → segment_name
        cat_id_to_name: Dict[int, str] = {}
        for cat in coco["categories"]:
            cat_id_to_name[cat["id"]] = cat["name"]

        # Przyporządkuj obrazom zbiór segmentów
        img_id_to_segments: Dict[int, Set[str]] = defaultdict(set)
        for ann in coco["annotations"]:
            cat_name = cat_id_to_name.get(ann["category_id"], "")
            # Wyciągnij numer segmentu z nazwy kategorii
            # Nazwy mogą być "RCA_prox" → segment "1", albo po prostu "1"
            seg_name = self._resolve_segment_name(cat_name)
            if seg_name:
                img_id_to_segments[ann["image_id"]].add(seg_name)

        self.samples: List[Tuple[str, int]] = []
        img_id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

        for img_id, segments in img_id_to_segments.items():
            side = distinguish_side(list(segments))
            label = 0 if side == "right" else 1
            fname = img_id_to_filename.get(img_id, "")
            if fname:
                self.samples.append((os.path.join(images_dir, fname), label))

        # ── Walidacja: sprawdź czy kategorie zawierają rozpoznawalne segmenty ──
        all_cat_names = set(cat_id_to_name.values())
        resolved_any = any(self._resolve_segment_name(n) is not None for n in all_cat_names)
        if not resolved_any:
            warnings.warn(
                f"⚠️  SideClassificationDataset: żadna kategoria COCO ({all_cat_names}) "
                f"nie pasuje do znanych segmentów wieńcowych. "
                f"Upewnij się, że DATASET_DIR wskazuje na dataset MULTICLASS "
                f"(np. arcade_coco_detection2), a nie BINARY (np. arcade_coco_binary). "
                f"Plik: {coco_json}"
            )
        if len(self.samples) == 0 and len(coco['annotations']) > 0:
            warnings.warn(
                f"⚠️  SideClassificationDataset: 0 próbek wygenerowanych z {len(coco['annotations'])} adnotacji. "
                f"Kategorie w datasecie: {all_cat_names}. "
                f"Oczekiwane nazwy segmentów: {sorted(SEGMENT_NAME_TO_CLASS_ID.keys())}"
            )

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _resolve_segment_name(cat_name: str) -> Optional[str]:
        """
        Rozpoznawanie nazwy segmentu z nazwy kategorii COCO.
        Obsługuje zarówno format "1", "16a" jak i "RCA_prox", "PLB_16a".
        """
        # Bezpośrednie dopasowanie (format numeryczny)
        if cat_name in SEGMENT_NAME_TO_CLASS_ID:
            return cat_name
        # Format opisowy → klucz z CLASS_ID_TO_LABEL
        for class_id, label in CLASS_ID_TO_LABEL.items():
            if label == cat_name:
                return CLASS_ID_TO_SEGMENT_NAME.get(class_id)
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def train_side_classifier(
    train_json: str,
    train_images: str,
    val_json: Optional[str] = None,
    val_images: Optional[str] = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
    save_path: str = "side_classifier.pth",
) -> SideClassifierNet:
    """
    Trenuje klasyfikator strony (right / left) na danych ARCADE COCO.

    Returns:
        Wytrenowana sieć SideClassifierNet.
    """
    print("=" * 60)
    print("STAGE 1: Training Side Classifier (right / left)")
    print("=" * 60)

    train_ds = SideClassificationDataset(train_json, train_images)

    if len(train_ds) == 0:
        raise ValueError(
            f"❌ SideClassificationDataset ma 0 próbek — nie można trenować klasyfikatora.\n"
            f"   Sprawdź, czy DATASET_DIR wskazuje na dataset MULTICLASS z adnotacjami segmentów (np. arcade_coco_detection2),\n"
            f"   a nie dataset BINARY z jedną kategorią 'vessel' (np. arcade_coco_binary).\n"
            f"   Podany plik: {train_json}"
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = None
    if val_json and val_images:
        val_ds = SideClassificationDataset(val_json, val_images)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

    model = SideClassifierNet(pretrained=True, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Policz class balance
    rights = sum(1 for _, l in train_ds.samples if l == 0)
    lefts = len(train_ds.samples) - rights
    print(f"  Train samples: {len(train_ds)} (right={rights}, left={lefts})")

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        train_acc = correct / total
        train_loss = running_loss / total
        scheduler.step()

        # ── Validation ──
        val_msg = ""
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.long().to(device)
                    logits = model(images)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += images.size(0)
            val_acc = val_correct / val_total
            val_msg = f" | val_acc={val_acc:.4f}"

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
        else:
            if train_acc > best_val_acc:
                best_val_acc = train_acc
                torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} — loss={train_loss:.4f} acc={train_acc:.4f}{val_msg}")

    print(f"  ✓ Best accuracy: {best_val_acc:.4f}")
    print(f"  ✓ Saved: {save_path}")
    return model


# ════════════════════════════════════════════════════════════════════
# 3. DOMAIN HEURISTIC FILTER — filtrowanie detekcji reguły medyczne
# ════════════════════════════════════════════════════════════════════

@dataclass
class HeuristicConfig:
    """Parametry heurystyki domenowej."""
    # Mnożnik confidence dla segmentów z NIEWŁAŚCIWEJ strony
    wrong_side_penalty: float = 0.0     # 0 = kompletne usunięcie
    # Minimalny confidence po penalizacji, żeby zachować detekcję
    min_confidence_after_penalty: float = 0.1
    # Bonus confidence jeśli segment ma wykrytego sąsiada anatomicznego
    adjacency_bonus: float = 0.05
    # Maksymalny confidence po bonusie
    max_confidence: float = 0.99
    # Minimalna ilość detekcji z danej strony, żeby uznać stronę za pewną
    min_side_votes: int = 2
    # Confidence threshold dla głosowania strony
    side_vote_threshold: float = 0.3
    # Jeśli True, używaj klasyfikatora; jeśli False, tylko głosowanie detekcji
    use_classifier: bool = True
    # Waga klasyfikatora vs głosowanie (0-1, 1 = tylko klasyfikator)
    classifier_weight: float = 0.7


@dataclass
class PipelineResult:
    """Wynik inferencji pipeline'u."""
    side: str                          # "right" | "left"
    side_confidence: float             # Pewność klasyfikacji strony
    detections: Any                    # sv.Detections — przefiltrowane
    raw_detections: Any                # sv.Detections — przed filtrowaniem
    suppressed_class_ids: List[int]    # Class ID usunięte przez heurystykę
    adjacency_bonuses: Dict[int, float]  # Bonusy za sąsiedztwo
    heuristic_log: List[str]           # Log decyzji heurystyki


class DomainHeuristicFilter:
    """
    Filtr heurystyczny oparty na wiedzy domenowej o anatomii tętnic wieńcowych.

    Reguły:
    1. Jeśli obraz = RIGHT → segmenty LEFT dostają penalty (lub usunięcie)
    2. Jeśli obraz = LEFT → segmenty RIGHT dostają penalty (lub usunięcie)
    3. Jeśli segment ma wykrytego sąsiada → bonus confidence
    4. Izolowane segmenty bez sąsiadów → ostrzeżenie
    """

    def __init__(self, config: Optional[HeuristicConfig] = None):
        self.config = config or HeuristicConfig()

    def determine_side_from_detections(
        self,
        class_ids: np.ndarray,
        confidences: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Głosowanie strony na podstawie detekcji RF-DETR-Seg.

        Każda detekcja z confidence > threshold głosuje na swoją stronę.
        Zwraca (side, confidence).
        """
        right_votes = 0.0
        left_votes = 0.0

        for cid, conf in zip(class_ids, confidences):
            if conf < self.config.side_vote_threshold:
                continue
            if cid in RIGHT_CLASS_IDS:
                right_votes += conf
            elif cid in LEFT_CLASS_IDS:
                left_votes += conf

        total = right_votes + left_votes
        if total < 1e-8:
            return "left", 0.5  # brak pewności, domyślnie left (częściej)

        if right_votes > left_votes:
            return "right", right_votes / total
        return "left", left_votes / total

    def apply_adjacency_bonus(
        self,
        class_ids: np.ndarray,
        confidences: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[int, float]]:
        """
        Dodaj bonus confidence jeśli segment ma wykrytego sąsiada.
        """
        detected_set = set(class_ids.tolist())
        new_confidences = confidences.copy()
        bonuses: Dict[int, float] = {}

        for i, cid in enumerate(class_ids):
            neighbors = ANATOMICAL_ADJACENCY.get(cid, set())
            if neighbors & detected_set:
                bonus = self.config.adjacency_bonus
                new_conf = min(new_confidences[i] + bonus, self.config.max_confidence)
                if new_conf != new_confidences[i]:
                    bonuses[int(cid)] = bonus
                    new_confidences[i] = new_conf

        return new_confidences, bonuses

    def filter(
        self,
        detections: Any,  # sv.Detections
        side: str,
        side_confidence: float,
    ) -> PipelineResult:
        """
        Zastosuj pełen zestaw heurystyk domenowych.

        Args:
            detections: Surowe detekcje z RF-DETR-Seg
            side: Określona strona ("right" | "left")
            side_confidence: Pewność klasyfikacji strony

        Returns:
            PipelineResult z przefiltrowanymi detekcjami
        """
        log: List[str] = []
        suppressed: List[int] = []

        if detections is None or len(detections) == 0:
            return PipelineResult(
                side=side,
                side_confidence=side_confidence,
                detections=detections,
                raw_detections=detections,
                suppressed_class_ids=[],
                adjacency_bonuses={},
                heuristic_log=["No detections to filter."],
            )

        class_ids = detections.class_id.copy()
        confidences = detections.confidence.copy()

        # ── 1. Penalizacja segmentów z niewłaściwej strony ──
        wrong_side_ids = LEFT_CLASS_IDS if side == "right" else RIGHT_CLASS_IDS
        keep_mask = np.ones(len(detections), dtype=bool)

        for i, (cid, conf) in enumerate(zip(class_ids, confidences)):
            if cid in wrong_side_ids:
                if self.config.wrong_side_penalty == 0.0:
                    # Całkowite usunięcie
                    keep_mask[i] = False
                    suppressed.append(int(cid))
                    seg_name = CLASS_ID_TO_SEGMENT_NAME.get(cid, "?")
                    log.append(
                        f"SUPPRESS: segment {seg_name} (class={cid}, conf={conf:.3f}) "
                        f"— wrong side ({side})"
                    )
                else:
                    new_conf = conf * self.config.wrong_side_penalty
                    if new_conf < self.config.min_confidence_after_penalty:
                        keep_mask[i] = False
                        suppressed.append(int(cid))
                        log.append(
                            f"SUPPRESS: segment {CLASS_ID_TO_SEGMENT_NAME.get(cid, '?')} "
                            f"(class={cid}, conf={conf:.3f}→{new_conf:.3f}) — below threshold"
                        )
                    else:
                        confidences[i] = new_conf
                        log.append(
                            f"PENALIZE: segment {CLASS_ID_TO_SEGMENT_NAME.get(cid, '?')} "
                            f"(class={cid}, conf={conf:.3f}→{new_conf:.3f})"
                        )

        # ── 2. Adjacency bonus (na zachowanych detekcjach) ──
        remaining_ids = class_ids[keep_mask]
        remaining_confs = confidences[keep_mask]
        adj_confs, adj_bonuses = self.apply_adjacency_bonus(remaining_ids, remaining_confs)

        # Wstaw z powrotem
        conf_iter = iter(adj_confs)
        for i in range(len(confidences)):
            if keep_mask[i]:
                confidences[i] = next(conf_iter)

        if adj_bonuses:
            log.append(f"ADJACENCY BONUS: {adj_bonuses}")

        # ── 3. Zbuduj przefiltrowane detekcje ──
        filtered = detections[keep_mask]
        # Nadpisz confidence z uwzględnionymi bonusami/penalty
        if hasattr(filtered, 'confidence') and len(filtered) > 0:
            filtered.confidence = confidences[keep_mask]

        log.append(
            f"RESULT: {len(detections)} → {len(filtered)} detections "
            f"(suppressed {len(suppressed)} wrong-side)"
        )

        return PipelineResult(
            side=side,
            side_confidence=side_confidence,
            detections=filtered,
            raw_detections=detections,
            suppressed_class_ids=suppressed,
            adjacency_bonuses=adj_bonuses,
            heuristic_log=log,
        )


# ════════════════════════════════════════════════════════════════════
# 4. GŁÓWNY PIPELINE — łączy wszystkie komponenty
# ════════════════════════════════════════════════════════════════════

def _remap_features_classifier_to_resnet(state_dict: dict) -> dict:
    """Map state_dict with keys features.*/classifier.* (Prefect/ml-workflows) to torchvision ResNet keys."""
    prefix_map = {
        "features.0.": "conv1.",
        "features.1.": "bn1.",
        "features.4.0.": "layer1.0.",
        "features.4.1.": "layer1.1.",
        "features.5.0.": "layer2.0.",
        "features.5.1.": "layer2.1.",
        "features.6.0.": "layer3.0.",
        "features.6.1.": "layer3.1.",
        "features.7.0.": "layer4.0.",
        "features.7.1.": "layer4.1.",
        "classifier.1.": "fc.1.",
    }
    out = {}
    for k, v in state_dict.items():
        for old_pref, new_pref in prefix_map.items():
            if k.startswith(old_pref):
                out[new_pref + k[len(old_pref):]] = v
                break
    return out


class CoronaryPipeline:
    """
    End-to-end pipeline: Side Classification → RF-DETR-Seg → Domain Heuristics.

    Training:
        pipeline = CoronaryPipeline(device="cuda")
        pipeline.train_side_classifier(...)
        pipeline.train_rfdetr_seg(...)

    Inference:
        pipeline = CoronaryPipeline.load(checkpoint_dir, device="cuda")
        result = pipeline.predict(image, threshold=0.5)
    """

    def __init__(
        self,
        device: str = "cuda",
        heuristic_config: Optional[HeuristicConfig] = None,
    ):
        self.device = device
        self.heuristic = DomainHeuristicFilter(heuristic_config)

        self.side_classifier: Optional[SideClassifierNet] = None
        self.rfdetr_seg: Optional[Any] = None

        self._side_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ── Training Methods ─────────────────────────────────────────

    def train_side_classifier(
        self,
        dataset_dir: str,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-4,
        save_path: str = "checkpoints/side_classifier.pth",
    ) -> None:
        """
        Trenuj klasyfikator strony (Stage 1).
        
        Uwaga: Używa nowej architektury (CrossEntropyLoss, 2 output classes).
        """
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        
        # Override SideClassifierNet training logic to use CrossEntropy
        print("STAGE 1: Training Side Classifier (Architecture: ResNet18 -> 2 classes)")
        
        # ... (setup loaders) ...
        train_json = os.path.join(dataset_dir, "train", "_annotations.coco.json")
        train_imgs = os.path.join(dataset_dir, "train")

        val_json = os.path.join(dataset_dir, "valid", "_annotations.coco.json")
        val_imgs = os.path.join(dataset_dir, "valid")

        train_ds = SideClassificationDataset(train_json, train_imgs)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

        val_loader = None
        if os.path.exists(val_json):
            val_ds = SideClassificationDataset(val_json, val_imgs)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
            
        model = SideClassifierNet(pretrained=True, num_classes=2).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.long().to(self.device)  # CE expects long
                
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            
            # Validation
            if val_loader:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        logits = model(images)
                        preds = torch.argmax(logits, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                        
                acc = correct / total
                print(f"Epoch {epoch+1}/{epochs} - Val Acc: {acc:.4f}")
                
                if acc > best_val_acc:
                    best_val_acc = acc
                    torch.save(model.state_dict(), save_path)
            else:
                 torch.save(model.state_dict(), save_path)
                 
        self.side_classifier = model
        print(f"Done. Saved to {save_path}")

    def train_rfdetr_seg(
        self,
        dataset_dir: str,
        epochs: int = 50,
        batch_size: int = 4,
        lr: float = 5e-4,
        checkpoint_dir: str = "checkpoints",
        model_size: str = "large",
        **kwargs,
    ) -> None:
        """
        Trenuj RF-DETR-Seg na multiclass segmentacji (Stage 2).

        Args:
            dataset_dir: Katalog COCO dataset z 25 kategoriami
            model_size: "large" (default)
            **kwargs: Dodatkowe argumenty do model.train()
        """
        if RFDETRSegLarge is None:
            raise ImportError("rfdetr package not installed. pip install rfdetr>=1.4.0")

        print("=" * 60)
        print("STAGE 2: Training RF-DETR-Seg (multiclass segmentation)")
        print("=" * 60)

        model = RFDETRSegLarge(num_classes=25)

        train_kwargs = dict(
            dataset_dir=dataset_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lr_encoder=lr * 0.1,
            lr_scheduler="cosine",
            warmup_epochs=3,
            weight_decay=1e-4,
            use_ema=True,
            early_stopping=True,
            early_stopping_patience=15,
            checkpoint_interval=5,
            output_dir=checkpoint_dir,
        )
        train_kwargs.update(kwargs)

        model.train(**train_kwargs)
        model.optimize_for_inference()

        self.rfdetr_seg = model
        print("  ✓ RF-DETR-Seg training complete")

    # ── Loading ──────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        checkpoint_dir: str,
        rfdetr_checkpoint: Optional[str] = None,
        side_classifier_checkpoint: Optional[str] = None,
        device: str = "cuda",
        heuristic_config: Optional[HeuristicConfig] = None,
        num_classes: int = 25,
    ) -> "CoronaryPipeline":
        """
        Załaduj wytrenowany pipeline z checkpointów.

        Args:
            checkpoint_dir: Katalog z checkpointami
            rfdetr_checkpoint: Ścieżka do wag RF-DETR-Seg (lub None → auto-detect)
            side_classifier_checkpoint: Ścieżka do wag side classifier (lub None → auto-detect)
            device: "cuda" | "cpu"
        """
        pipeline = cls(device=device, heuristic_config=heuristic_config)

        # ── Load side classifier ──
        sc_path = side_classifier_checkpoint or os.path.join(checkpoint_dir, "side_classifier.pth")
        if os.path.exists(sc_path):
            pipeline.side_classifier = SideClassifierNet(pretrained=False, num_classes=2)
            raw = torch.load(sc_path, map_location=device, weights_only=False)
            state_dict = raw.get("model_state_dict", raw)
            if any(k.startswith("features.") for k in state_dict):
                state_dict = _remap_features_classifier_to_resnet(state_dict)
            pipeline.side_classifier.load_state_dict(state_dict, strict=True)
            pipeline.side_classifier.to(device).eval()
            print(f"  ✓ Loaded side classifier: {sc_path}")
        else:
            warnings.warn(f"Side classifier not found at {sc_path}. Will use detection-based voting.")

        # ── Load RF-DETR-Seg ──
        rfdetr_path = rfdetr_checkpoint
        if rfdetr_path is None:
            # Auto-detect: checkpoint_best_total.pth w checkpoint_dir
            candidates = [
                os.path.join(checkpoint_dir, "checkpoint_best_total.pth"),
                os.path.join(checkpoint_dir, "checkpoint_last.pth"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    rfdetr_path = c
                    break

        if rfdetr_path and os.path.exists(rfdetr_path):
            if RFDETRSegLarge is None:
                raise ImportError("rfdetr package required")
            pipeline.rfdetr_seg = RFDETRSegLarge(
                num_classes=num_classes,
                pretrain_weights=rfdetr_path,
            )
            pipeline.rfdetr_seg.optimize_for_inference()
            print(f"  ✓ Loaded RF-DETR-Seg: {rfdetr_path}")
        else:
            raise FileNotFoundError(
                f"RF-DETR-Seg checkpoint not found. Searched: {rfdetr_path or checkpoint_dir}"
            )

        return pipeline

    # ── Inference ────────────────────────────────────────────────

    def classify_side(self, image: Image.Image) -> Tuple[str, float]:
        """
        Klasyfikuj stronę obrazu za pomocą sieci ResNet18.

        Returns:
            (side, confidence) — "right" | "left", 0.0-1.0
        """
        if self.side_classifier is None:
            raise RuntimeError("Side classifier not loaded. Use detection-based voting instead.")

        self.side_classifier.eval()
        x = self._side_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.side_classifier(x)
            probs = torch.softmax(logits, dim=1).squeeze(0) # [p0, p1]
            
            # Assuming class 0 = right, class 1 = left (based on SideClassificationDataset logic)
            prob_left = probs[1].item()
            prob_right = probs[0].item()

        # If prob_left > 0.5 -> left
        if prob_left > 0.5:
            return "left", prob_left
        else:
            return "right", prob_right

    def predict(
        self,
        image: Union[Image.Image, np.ndarray, str],
        threshold: float = 0.5,
        verbose: bool = False,
    ) -> PipelineResult:
        """
        Pełna inferencja: Side Classification → Detection → Heuristic Filtering.

        Args:
            image: Obraz wejściowy (PIL, numpy, albo ścieżka do pliku)
            threshold: Próg confidence dla RF-DETR-Seg
            verbose: Czy drukować logi heurystyki

        Returns:
            PipelineResult z przefiltrowanymi detekcjami
        """
        # ── Konwersja wejścia ──
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        # ── Step 1: Run RF-DETR-Seg ──
        if self.rfdetr_seg is None:
            raise RuntimeError("RF-DETR-Seg not loaded.")

        raw_detections = self.rfdetr_seg.predict(image, threshold=threshold)

        # ── Step 2: Determine side ──
        classifier_side, classifier_conf = None, 0.0
        detection_side, detection_conf = None, 0.0

        # 2a. Classifier-based
        if self.heuristic.config.use_classifier and self.side_classifier is not None:
            classifier_side, classifier_conf = self.classify_side(image)

        # 2b. Detection-based voting
        if raw_detections is not None and len(raw_detections) > 0:
            detection_side, detection_conf = self.heuristic.determine_side_from_detections(
                raw_detections.class_id, raw_detections.confidence
            )

        # 2c. Fuzja decyzji
        w = self.heuristic.config.classifier_weight
        if classifier_side is not None and detection_side is not None:
            if classifier_side == detection_side:
                # Zgodność — wysoka pewność
                final_side = classifier_side
                final_conf = w * classifier_conf + (1 - w) * detection_conf
            else:
                # Konflikt — użyj tego z wyższą wagowaną pewnością
                c_score = w * classifier_conf
                d_score = (1 - w) * detection_conf
                if c_score >= d_score:
                    final_side = classifier_side
                    final_conf = classifier_conf
                else:
                    final_side = detection_side
                    final_conf = detection_conf
        elif classifier_side is not None:
            final_side = classifier_side
            final_conf = classifier_conf
        elif detection_side is not None:
            final_side = detection_side
            final_conf = detection_conf
        else:
            final_side = "left"
            final_conf = 0.5

        # ── Step 3: Domain heuristic filtering ──
        result = self.heuristic.filter(raw_detections, final_side, final_conf)

        if verbose:
            print(f"\n{'='*50}")
            print(f"Pipeline Result:")
            print(f"  Side: {result.side} (confidence={result.side_confidence:.3f})")
            if classifier_side:
                print(f"    Classifier: {classifier_side} ({classifier_conf:.3f})")
            if detection_side:
                print(f"    Detection voting: {detection_side} ({detection_conf:.3f})")
            print(f"  Detections: {len(result.raw_detections)} → {len(result.detections)}")
            if result.suppressed_class_ids:
                names = [CLASS_ID_TO_SEGMENT_NAME.get(c, str(c)) for c in result.suppressed_class_ids]
                print(f"  Suppressed: {names}")
            if result.adjacency_bonuses:
                print(f"  Adjacency bonuses: {result.adjacency_bonuses}")
            for line in result.heuristic_log:
                print(f"    {line}")
            print(f"{'='*50}\n")

        return result

    # ── Visualization Helper ─────────────────────────────────────

    def visualize(
        self,
        image: Union[Image.Image, np.ndarray, str],
        result: PipelineResult,
        show_raw: bool = True,
    ) -> np.ndarray:
        """
        Wizualizacja wyniku pipeline'u z kolorami per-class.

        Returns:
            Obraz numpy z adnotacjami (lub siatka porównawcza jeśli show_raw=True).
        """
        if sv is None:
            raise ImportError("supervision package required for visualization")

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        palette = sv.ColorPalette.from_hex([
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
            "#00FFFF", "#FF8000", "#8000FF", "#00FF80", "#FF0080",
            "#80FF00", "#0080FF", "#FF8080", "#80FF80", "#8080FF",
            "#FFFF80", "#FF80FF", "#80FFFF", "#C00000", "#00C000",
            "#0000C0", "#C0C000", "#C000C0", "#00C0C0", "#FFA500",
        ])

        mask_ann = sv.MaskAnnotator(color=palette, opacity=0.4, color_lookup=sv.ColorLookup.CLASS)
        label_ann = sv.LabelAnnotator(
            color=palette,
            text_color=sv.Color.BLACK,
            text_scale=sv.calculate_optimal_text_scale(resolution_wh=image.size),
            color_lookup=sv.ColorLookup.CLASS,
        )

        def annotate(img, dets, title_prefix=""):
            out = img.copy()
            if dets is not None and len(dets) > 0:
                if dets.mask is not None:
                    out = mask_ann.annotate(out, dets)
                labels = [
                    f"{CLASS_ID_TO_LABEL.get(cid, str(cid))} {conf:.2f}"
                    for cid, conf in zip(dets.class_id, dets.confidence)
                ]
                out = label_ann.annotate(out, dets, labels)
            return out

        filtered_img = annotate(image, result.detections)

        if show_raw and result.raw_detections is not None:
            raw_img = annotate(image, result.raw_detections)
            return sv.plot_images_grid(
                [raw_img, filtered_img],
                grid_size=(1, 2),
                titles=[
                    f"Raw ({len(result.raw_detections)} dets)",
                    f"Filtered [{result.side}] ({len(result.detections)} dets)",
                ],
            )

        return filtered_img


# ════════════════════════════════════════════════════════════════════
# 5. BATCH EVALUATION — ewaluacja pipeline'u na zbiorze testowym
# ════════════════════════════════════════════════════════════════════

def evaluate_pipeline(
    pipeline: CoronaryPipeline,
    test_json: str,
    test_images: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Ewaluacja pipeline'u: mierzy accuracy klasyfikacji strony
    oraz wpływ heurystyk na metryki detekcji.
    """
    with open(test_json, "r") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    img_id_to_segments: Dict[int, Set[str]] = defaultdict(set)

    for ann in coco["annotations"]:
        name = cat_id_to_name.get(ann["category_id"], "")
        seg = SideClassificationDataset._resolve_segment_name(name)
        if seg:
            img_id_to_segments[ann["image_id"]].add(seg)

    # Ewaluacja
    side_correct = 0
    side_total = 0
    total_raw_dets = 0
    total_filtered_dets = 0
    total_suppressed = 0

    for img_info in tqdm(coco["images"], desc="Evaluating pipeline"):
        img_path = os.path.join(test_images, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        gt_segments = list(img_id_to_segments.get(img_info["id"], set()))
        if not gt_segments:
            continue

        gt_side = distinguish_side(gt_segments)

        result = pipeline.predict(img_path, threshold=threshold)

        if result.side == gt_side:
            side_correct += 1
        side_total += 1

        raw_n = len(result.raw_detections) if result.raw_detections is not None else 0
        filt_n = len(result.detections) if result.detections is not None else 0
        total_raw_dets += raw_n
        total_filtered_dets += filt_n
        total_suppressed += len(result.suppressed_class_ids)

    metrics = {
        "side_accuracy": side_correct / max(side_total, 1),
        "side_correct": side_correct,
        "side_total": side_total,
        "avg_raw_detections": total_raw_dets / max(side_total, 1),
        "avg_filtered_detections": total_filtered_dets / max(side_total, 1),
        "total_suppressed": total_suppressed,
        "avg_suppressed_per_image": total_suppressed / max(side_total, 1),
    }

    print(f"\n{'='*50}")
    print("Pipeline Evaluation Results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics
