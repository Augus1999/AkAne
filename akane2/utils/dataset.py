# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
DataSets classes
The item returned from these classes is a python 
dictionary: {"mol": mol_dict, "label": label_dict},
where mol_dict is a dictionary as {
                                    "node": node state (Tensor), 
                                    "edge": edge stste (Tensor),
                                    }
and label_dict is dictionary as {
                                  "property": property (Tensor),
                                  "token_input": input tokens (Tensor),
                                  "token_label" label tokens (Tensor),
                                  }.
"""
import os
import csv
import glob
from math import exp
from pathlib import Path
from typing import Optional, List, Dict, Union, Generator
import torch
from torch import Tensor
from torch.utils.data import Dataset
from .graph import smiles2graph
from .token import smiles2vec

valid_dataset_name = ("CMC", "ESOL", "FreeSolv", "Lipo")


class CSVData(Dataset):
    def __init__(
        self,
        file: str,
        limit: Optional[int] = None,
    ) -> None:
        """
        Dataset stored in CSV file.

        :param file: dataset file name <file>
        :param limit: item limit
        """
        super().__init__()
        self.data = []
        with open(file, "r") as db:
            data = csv.reader(db)
            self.data = list(data)
        self.smiles_idx, self.value_idx = [], []
        self.ratio_idx, self.temp_idx = None, None
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
        if self.ratio_idx:
            if d[self.ratio_idx] != "":
                rs = d[self.ratio_idx].split(":")
                rs = torch.tensor([float(i) for i in rs])
                alphas = list((rs / rs.sum()).numpy())
        if self.temp_idx:
            if d[self.temp_idx] != "":
                beta = exp(-273.15 / float(d[self.temp_idx]))
        graph = smiles2graph(smiles=smiles, alphas=alphas, beta=beta)
        property = torch.tensor(values, dtype=torch.float32)
        token = smiles2vec(smiles4vec)
        # start token: <start> = 1; end token: <esc> = 2
        token_input = torch.tensor([1] + token, dtype=torch.long)
        token_label = torch.tensor(token + [2], dtype=torch.long)
        label = {
            "property": property,
            "token_input": token_input,
            "token_label": token_label,
        }
        return {"mol": graph, "label": label}


class DataSet:
    """
    Import dataset. \n
    Recently support CMC, FreeSolv, Lipo and ESOL datasets.
    """

    def __init__(
        self,
        name: str,
        dir_: str,
        mode: str = "train",
        limit: Optional[int] = None,
    ) -> None:
        """
        Check the state of dataset.

        :param name: dataset name
        :param dir: where the dataset locates <path>
        :param mode: mode; "train" or "test"
        :param limit: item limit
        """
        assert (
            name in valid_dataset_name
        ), f"keyword 'name' should be one of {valid_dataset_name}"
        self.name = name.split(".")[0]
        assert mode in ("train", "test"), "invalid mode..."
        self.dir = Path(dir_)
        self.mode = mode
        self.limit = limit
        self._check()

    @property
    def _dataset(self) -> Dataset:
        """
        Return the required dataset class.
        """
        dataset = {
            "CMC": CSVData,
            "ESOL": CSVData,
            "FreeSolv": CSVData,
            "Lipo": CSVData,
        }
        return dataset[self.name]

    def _check(self) -> bool:
        """
        Check whether all required files exist.
        """
        if self.name in ("CMC", "ESOL", "FreeSolv", "Lipo"):
            d_files = list(glob.glob(str(self.dir / r"*.csv")))
            d_files = set([os.path.basename(i) for i in d_files])
            valid_files = {
                self.name.lower() + "_train.csv",
                self.name.lower() + "_test.csv",
            }
            if not valid_files & d_files == valid_files:
                raise FileNotFoundError(
                    f"No dataset was found under directory '{self.dir}'."
                )

    @property
    def data(self) -> Generator:
        """
        Return data from dataset.
        """
        if self.name in ("CMC", "ESOL", "FreeSolv", "Lipo"):
            file_names = {
                "train": f"{self.name.lower()}_train.csv",
                "test": f"{self.name.lower()}_test.csv",
            }
            file_name = file_names[self.mode]
        dataset = self._dataset(
            file=self.dir / file_name,
            limit=self.limit,
        )
        return dataset
