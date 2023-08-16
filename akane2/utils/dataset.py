# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
DataSets class
The item returned is a python dictionary: {"mol": mol_dict, "label": label_dict},
where mol_dict is a dictionary as {
                                    "node": node state (Tensor), 
                                    "edge": edge stste (Tensor),
                                    }
and label_dict is dictionary as {
                                  "property": property (Tensor, optional),
                                  "token_input": input tokens (Tensor),
                                  "token_label" label tokens (Tensor),
                                  }.
"""
import csv
import random
from math import exp
from typing import Optional, List, Dict, Union
import torch
from torch import Tensor
from torch.utils.data import Dataset
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from .graph import smiles2graph
from .token import smiles2vec, protein2vec


class CSVData(Dataset):
    def __init__(
        self,
        file: str,
        limit: Optional[int] = None,
        label_idx: Optional[List[int]] = None,
    ) -> None:
        """
        Dataset stored in CSV file.

        :param file: dataset file name <file>
        :param limit: item limit
        :param label_idx: a list of indices indicating which value to be input
                          use 'None' for inputting all values
        """
        super().__init__()
        self.data = []
        with open(file, "r") as db:
            data = csv.reader(db)
            self.data = list(data)
        self.label_idx = label_idx
        self.smiles_idx, self.value_idx = [], []
        self.ratio_idx, self.temp_idx, self.seq_idx = None, None, None
        for key, i in enumerate(self.data[0]):
            i = i.lower()
            if i == "smiles":
                self.smiles_idx.append(key)
            if i == "value":
                self.value_idx.append(key)
            if i == "ratio":
                self.ratio_idx = key
            if i == "temperature":
                self.temp_idx = key
            if i == "seq":
                self.seq_idx = key
        self.data = self.data[1:]
        self.data = self.data[:limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: Union[int, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        d: List[str] = self.data[idx]
        alphas, beta = None, None
        smiles = [d[idx] for idx in self.smiles_idx]
        smiles4vec = ".".join(smiles)
        values = [
            float(d[idx]) if d[idx].strip() != "" else torch.inf
            for idx in self.value_idx
        ]
        if self.label_idx:
            values = [values[i] for i in self.label_idx]
        if self.seq_idx:
            values = protein2vec(d[self.seq_idx])
        if self.ratio_idx:
            if d[self.ratio_idx] != "":
                rs = d[self.ratio_idx].split(":")
                rs = torch.tensor([float(i) for i in rs])
                alphas = list((rs / rs.sum()).numpy())
        if self.temp_idx:
            if d[self.temp_idx] != "":
                beta = exp(-273.15 / float(d[self.temp_idx]))
        graph = smiles2graph(smiles=smiles, alphas=alphas, beta=beta)
        token = smiles2vec(smiles4vec)
        # start token: <start> = 1; end token: <esc> = 2
        token_input = torch.tensor([1] + token, dtype=torch.long)
        token_label = torch.tensor(token + [2], dtype=torch.long)
        label = {"token_input": token_input, "token_label": token_label}
        if len(values) != 0:
            label["property"] = torch.tensor(values, dtype=torch.float32)
        return {"mol": graph, "label": label}


def split_dataset(file: str, split_ratio: float = 0.8, method: str = "random") -> None:
    assert file.endswith(".csv")
    assert 0 < split_ratio < 1
    assert method in ("random", "scaffold")
    with open(file, "r") as f:
        data = list(csv.reader(f))
    header = data[0]
    raw_data = data[1:]
    smiles_idx = []  # only first index will be used
    for key, h in enumerate(header):
        if h == "smiles":
            smiles_idx.append(key)
    assert len(smiles_idx) > 0
    split_idx = int(len(raw_data) * split_ratio)
    if method == "random":
        random.shuffle(raw_data)
        train_set, test_set = raw_data[:split_idx], raw_data[split_idx:]
    if method == "scaffold":
        scaffolds = {}
        for key, d in enumerate(raw_data):
            # compute Bemis-Murcko scaffold
            scaffold = MurckoScaffoldSmiles(d[smiles_idx[0]])
            if scaffold in scaffolds:
                scaffolds[scaffold].append(key)
            else:
                scaffolds[scaffold] = [key]
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        train_set, test_set = [], []
        for idxs in scaffolds.values():
            if len(train_set) + len(idxs) > split_idx:
                test_set += [raw_data[i] for i in idxs]
            else:
                train_set += [raw_data[i] for i in idxs]
    with open(file.replace(".csv", "_train.csv"), "w", newline="") as ftr:
        writer = csv.writer(ftr)
        writer.writerows([header] + train_set)
    with open(file.replace(".csv", "_test.csv"), "w", newline="") as fte:
        writer = csv.writer(fte)
        writer.writerows([header] + test_set)
