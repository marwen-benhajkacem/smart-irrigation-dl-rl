"""Utilise un modèle de prévision entraîné DANS la boucle de l'agent (agent v2)."""
from collections import deque

import numpy as np
import torch

from .models import build_model


class ForecasterRuntime:
    def __init__(self, ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.features = ck["features"]
        self.mean = np.asarray(ck["mean"], dtype=np.float32)
        self.std = np.asarray(ck["std"], dtype=np.float32)
        self.sw_std, self.K, self.H = ck["sw_std"], ck["K"], ck["H"]
        self.model = build_model(ck["model"], len(self.features), self.H, ck["hidden"], ck["layers"], ck["dropout"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        torch.set_num_threads(1)
        self.hist = deque(maxlen=self.K)
        self.ckpt_path = str(ckpt_path)

    def reset(self):
        self.hist.clear()

    def update(self, row):
        self.hist.append([row[f] for f in self.features])
        self._sw_now = row["sw_root"]

    def predict(self):
        if len(self.hist) < self.K:                      # début de saison : persistance
            return np.full(self.H, self._sw_now, dtype=float)
        x = (np.asarray(self.hist, dtype=np.float32) - self.mean) / self.std
        with torch.no_grad():
            d = self.model(torch.from_numpy(x[None])).numpy()[0]
        return self._sw_now + d * self.sw_std
