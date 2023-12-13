# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
build graph from SMILES string(s)
"""
from typing import Dict, List, Tuple, Union, Optional
from rdkit import Chem
import torch
import torch.nn.functional as F
from torch import Tensor


electron_config = torch.tensor(
    [
        # 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p 7s 5f 6d 7p 6f 7d 7f
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # proton
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # H
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # He
        [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Li
        [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Be
        [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B
        [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C
        [2, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # N
        [2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # O
        [2, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # F
        [2, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ne
        [2, 2, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Na
        [2, 2, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Mg
        [2, 2, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Al
        [2, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Si
        [2, 2, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # P
        [2, 2, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # S
        [2, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Cl
        [2, 2, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ar
        [2, 2, 6, 2, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # K
        [2, 2, 6, 2, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ca
        [2, 2, 6, 2, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 5, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 7, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 8, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 9, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 1, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 2, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 3, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 4, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 5, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 1, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 0, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 1, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 2, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 3, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 4, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 5, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 6, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 7, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 8, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 9, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 10, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 11, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 12, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 13, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 0, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 1, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 2, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 3, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 4, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 5, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 6, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 7, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 8, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 9, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 0, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 1, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 2, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 3, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 4, 0, 0, 0],
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 5, 0, 0, 0],  # Ts
        [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 6, 0, 0, 0],  # Og
    ],
    dtype=torch.float32,
)  # ground state electron configurations
degrees = torch.tensor(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 3
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 6
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 7
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 9
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
    ],
    dtype=torch.float32,
)  # one-hot degree (0 - 10) representation
implicit_hs = torch.tensor(
    [
        [1, 0, 0, 0, 0, 0, 0],  # 0
        [0, 1, 0, 0, 0, 0, 0],  # 1
        [0, 0, 1, 0, 0, 0, 0],  # 2
        [0, 0, 0, 1, 0, 0, 0],  # 3
        [0, 0, 0, 0, 1, 0, 0],  # 4
        [0, 0, 0, 0, 0, 1, 0],  # 5
        [0, 0, 0, 0, 0, 0, 1],  # 6
    ],
    dtype=torch.float32,
)  # one-hot implicit hydrogen atom numbers (0 - 6)
hybridise = torch.tensor(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],  # unspecified
        [0, 1, 0, 0, 0, 0, 0, 0, 0],  # S
        [0, 0, 1, 0, 0, 0, 0, 0, 0],  # SP
        [0, 0, 0, 1, 0, 0, 0, 0, 0],  # SP2
        [0, 0, 0, 0, 1, 0, 0, 0, 0],  # SP3
        [0, 0, 0, 0, 0, 1, 0, 0, 0],  # SP2D
        [0, 0, 0, 0, 0, 0, 1, 0, 0],  # SP3D
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  # SP3D2
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # other
    ],
    dtype=torch.float32,
)  # one-hot hybridisation type (0 - 8)
chiral_types = torch.tensor(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],  # unspecified
        [0, 1, 0, 0, 0, 0, 0, 0, 0],  # tetra_CW
        [0, 0, 1, 0, 0, 0, 0, 0, 0],  # tetra_CCW
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # other
        [0, 0, 0, 1, 0, 0, 0, 0, 0],  # tetra
        [0, 0, 0, 0, 1, 0, 0, 0, 0],  # allene
        [0, 0, 0, 0, 0, 1, 0, 0, 0],  # squareplanar
        [0, 0, 0, 0, 0, 0, 1, 0, 0],  # trig
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  # oct
    ],
    dtype=torch.float32,
)  # one-hot chirality type (0 - 8)
bond_types = torch.tensor(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # unknow
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # single
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # double
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # triple
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # quadruple
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # quintuple
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # hextuple
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1-1/2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2-1/2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3-1/2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4-1/2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5-1/2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # aromatic
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # ionic
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # H-bond
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 3-centre
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # dativeone
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # dative
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # dativel
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # dativer
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # other
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # zero
    ],
    dtype=torch.float32,
)  # one-hot bond type (0 - 21)
stero_type = torch.tensor(
    [
        [1, 0, 0, 0, 0, 0],  # none
        [0, 1, 0, 0, 0, 0],  # any
        [0, 0, 1, 0, 0, 0],  # Z
        [0, 0, 0, 1, 0, 0],  # E
        [0, 0, 0, 0, 1, 0],  # Cis
        [0, 0, 0, 0, 0, 1],  # Trans
    ],
    dtype=torch.float32,
)  # one-hot bond stero type (0 - 5)


def _electron_config_feature(atoms) -> Tensor:
    z = torch.tensor([i.GetAtomicNum() for i in atoms], dtype=torch.long)
    return electron_config[z]


def _valence_feature(atoms) -> Tensor:
    valence = torch.tensor(
        [i.GetTotalValence() for i in atoms], dtype=torch.float32
    ).unsqueeze(dim=-1)
    return valence


def _formal_charge_feature(atoms) -> Tensor:
    formal_charge = torch.tensor(
        [i.GetFormalCharge() for i in atoms], dtype=torch.float32
    ).unsqueeze(dim=-1)
    return formal_charge


def _mass_feature(atoms) -> Tensor:
    mass = torch.tensor([i.GetMass() for i in atoms], dtype=torch.float32).unsqueeze(
        dim=-1
    )
    return mass / 100


def _degree_feature(atoms) -> Tensor:
    degree = torch.tensor([i.GetDegree() for i in atoms], dtype=torch.long)
    return degrees[degree]


def _implicit_h_feature(atoms) -> Tensor:
    implicit_h = torch.tensor([i.GetNumImplicitHs() for i in atoms], dtype=torch.long)
    return implicit_hs[implicit_h]


def _aromaticity_feature(atoms) -> Tensor:
    aromaticity = torch.tensor(
        [i.GetIsAromatic() for i in atoms], dtype=torch.float32
    ).unsqueeze(dim=-1)
    return aromaticity


def _hybridisation_feature(atoms) -> Tensor:
    hybridisation = torch.tensor(
        [int(i.GetHybridization()) for i in atoms], dtype=torch.long
    )
    return hybridise[hybridisation]


def _chiral_feature(atoms) -> Tensor:
    chiral_type = torch.tensor([int(i.GetChiralTag()) for i in atoms], dtype=torch.long)
    return chiral_types[chiral_type]


def _bond_type_feature(bonds) -> Tensor:
    bond_type = torch.tensor([int(i.GetBondType()) for i in bonds], dtype=torch.long)
    return bond_types[bond_type]


def _conjugation_feature(bonds) -> Tensor:
    conjugated = torch.tensor(
        [i.GetIsConjugated() for i in bonds], dtype=torch.float32
    ).unsqueeze(dim=-1)
    return conjugated


def _in_ring_feature(bonds) -> Tensor:
    ring = torch.tensor([i.IsInRing() for i in bonds], dtype=torch.float32).unsqueeze(
        dim=-1
    )
    return ring


def _stero_feature(bonds) -> Tensor:
    stero = torch.tensor([int(i.GetStereo()) for i in bonds], dtype=torch.long)
    return stero_type[stero]


def _bond_pair_idx(bonds) -> List:
    return [[i.GetBeginAtomIdx(), i.GetEndAtomIdx()] for i in bonds]


def _smiles2graph(smiles: str) -> Dict[str, Tensor]:
    """
    Generate graph features from one SMILES string

    :param smiles: SMILES string
    :return: {"node": nodes, "edge": edges};  shape: (n_a, 62) & (n_a, n_a, 30)
    """
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    num = len(atoms)
    e_config = _electron_config_feature(atoms)  #    ground state electron config
    valence = _valence_feature(atoms)  #             valence electron
    formal_charge = _formal_charge_feature(atoms)  # formal charge
    mass = _mass_feature(atoms)  #                   atomic mass / 100
    degree = _degree_feature(atoms)  #               degree
    h_count = _implicit_h_feature(atoms)  #          implicit Hs
    armoatic = _aromaticity_feature(atoms)  #        armoaticity
    hybridisation = _hybridisation_feature(atoms)  # hybridisation type
    chiral = _chiral_feature(atoms)  #               chirality
    node = torch.cat(
        [
            e_config,
            valence,
            formal_charge,
            mass,
            degree,
            h_count,
            armoatic,
            hybridisation,
            chiral,
        ],
        dim=-1,
    )
    bond_type = _bond_type_feature(bonds)  #    bond type
    conjugated = _conjugation_feature(bonds)  # conjugation
    in_ring = _in_ring_feature(bonds)  #        ring
    stero = _stero_feature(bonds)  #            bond stero type
    edge_feature = torch.cat([bond_type, conjugated, in_ring, stero], dim=-1)
    edge = torch.zeros(num, num, edge_feature.shape[-1])
    bond_pair_idx = _bond_pair_idx(bonds)
    for key, pair in enumerate(bond_pair_idx):
        edge[pair[0]][pair[1]] = edge_feature[key]
        edge[pair[1]][pair[0]] = edge_feature[key]
    graph = {"node": node, "edge": edge}
    return graph


def smiles2graph(
    smiles: Union[str, list],
    alphas: Optional[List[float]] = None,
    beta: Optional[float] = None,
    mol_sizes: bool = False,
) -> Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], List[int]]]:
    """
    Generate graph features from SMILES string(s)

    :param smiles: SMILES string(s)
    :param alphas: coefficients
    :param beta: bias
    :param mol_sizes: whether return the length(s) of molecule(s)
    :return: {"node": nodes, "edge": edges} (& [l1, l2, ...])
    """
    #-- NOTE: commented code is the second way to embed ratio and temperature #--
    sizes = []
    if isinstance(smiles, str):
        smiles = [smiles]
    # exclude empty string
    graphs = [_smiles2graph(i) for i in smiles if i != ""]
    nodes, edges = [], []
    # masses, ratios = [], []
    for key, graph in enumerate(graphs):
        node = graph["node"]
        if alphas:
            node *= alphas[key]
        nodes.append(node)
        sizes.append(node.shape[0])
        # masses.append(node[..., 24] * 100)
        # ratios.append(torch.ones(size) * alphas[key])
    nodes = torch.cat(nodes, dim=0)
    # masses = torch.cat(masses, dim=-1).unsqueeze(dim=-1)
    # ratios = torch.cat(ratios, dim=-1).unsqueeze(dim=-1)
    # mu_re = (1 / masses) + (1 / masses).transpose(-2, -1)
    # coeff = ratios * ratios.transpose(-2, -1)
    # freq_matrix = coeff * (8 * beta * 1.38 * 6.022 * mu_re / torch.pi).sqrt()
    if beta:
        nodes += beta
    n_total = nodes.shape[0]
    i = 0
    for graph in graphs:
        edge = graph["edge"]
        n = edge.shape[0]
        p1, p2 = torch.zeros(n, i, 30), torch.zeros(n, n_total - n - i, 30)
        edges.append(torch.cat([p1, edge, p2], dim=1))
        i += n
    edges = torch.cat(edges, dim=0)
    if mol_sizes:
        # return {"node": nodes, "edge": edges, "freq_matrix": freq_matrix}, sizes
        return {"node": nodes, "edge": edges}, sizes
    # return {"node": nodes, "edge": edges, "freq_matrix": freq_matrix}
    return {"node": nodes, "edge": edges}


def gather(batch: List) -> Dict[str, Tensor]:
    """
    Gathering different graph data with various length into one batch.

    :param batch: a list of data (one batch)
    :return: batched {"node": nodes, "edge": edges}
    """
    #-- NOTE: commented code is the second way to embed ratio and temperature --
    nodes, edges = [], []
    atom_numbers = [i["node"].shape[-2] for i in batch]
    max_n = max(atom_numbers)
    for item in batch:
        node, edge = item["node"], item["edge"]
        n = node.shape[-2]
        node = F.pad(node, (0, 0, 0, max_n - n), value=0)
        nodes.append(node.unsqueeze(dim=0))
        e_row_pad = torch.zeros(n, max_n - n, 30)
        edge = torch.cat([edge, e_row_pad], dim=-2)
        e_col_pad = torch.zeros(max_n - n, max_n, 30)
        edge = torch.cat([edge, e_col_pad], dim=-3)
        edges.append(edge.unsqueeze(dim=0))
        # if "freq_matrix" in item:
        #     freq_matrix = item["freq_matrix"]
        #     freq_matrix = F.pad(freq_matrix, (0, max_n - n, 0, max_n - n), value=0)
        #     freq_matrices.append(freq_matrix.unsqueeze(dim=0))
    nodes = torch.cat(nodes, dim=0)
    edges = torch.cat(edges, dim=0)
    batch = {"node": nodes, "edge": edges}
    # if freq_matrices:
    #     batch["freq_matrix"] = torch.cat(freq_matrices, dim=0)
    return batch


if __name__ == "__main__":
    ...
