# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Tokenise SMILES string and protein sequences
"""
import re
from pathlib import Path
from typing import List

__filedir__ = Path(__file__).parent
with open(__filedir__ / "regex_pattern.txt", "r", encoding="utf-8") as rp:
    regex_pattern = rp.read().strip().replace("\n", "")
SMI_REGEX_PATTERN = f"({regex_pattern})"
AA_REGEX_PATTERN = r"(A|B|C|D|E|F|G|H|I|K|L|M|N|P|Q|R|S|T|V|W|Y|Z)"
smi_regex = re.compile(SMI_REGEX_PATTERN)
aa_regex = re.compile(AA_REGEX_PATTERN)
with open(__filedir__ / "vocab.txt", "r", encoding="utf-8") as v:
    lines = v.read().strip()
VOCAB_KEYS = lines.split("\n") + [f"%{i}" for i in range(10, 100)]
VOCAB_COUNT = len(VOCAB_KEYS)
VOCAB_DICT = dict(zip(VOCAB_KEYS, range(VOCAB_COUNT)))
AA_KEYS = "<pad> A B C D E F G H I K L M N P Q R S T V W Y Z".split()
AA_COUNT = len(AA_KEYS)
AA_DICT = dict(zip(AA_KEYS, range(AA_COUNT)))


def _tokenize(text: str, regex: re.Pattern) -> List[str]:
    """
    Run tokenisation (text) using a regex pattern.
    """
    return [token for token in regex.findall(text)]


def smiles2vec(smiles: str) -> List[int]:
    """
    SMILES tokenisation using a regex pattern stored in regex_pattern.txt file.

    :param smiles: SMILES string
    :return: tokens
    """
    tokens = _tokenize(smiles, smi_regex)
    vec = [VOCAB_DICT[token] for token in tokens]
    return vec


def protein2vec(sequence: str) -> List[int]:
    """
    Protein sequence tokenisation.

    :param sequence: amino acid sequence
    :return: tokens
    """
    tokens = _tokenize(sequence, aa_regex)
    vec = [AA_DICT[token] for token in tokens]
    return vec


if __name__ == "__main__":
    ...
