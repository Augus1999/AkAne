# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Model representation
"""
from copy import deepcopy
from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self
from .utils.token import VOCAB_COUNT, VOCAB_KEYS, FS_MASK
from .module.encoder import Encoder
from .module.decoder import Decoder
from .module.readout import Readout
from .module.diffusion import GaussianDiffusion, DiT
from .module.kamome.module import Interaction, MLP


__all__ = ["AkAne", "Kamome"]


class AkAne(nn.Module):
    def __init__(
        self,
        n_vocab: int = VOCAB_COUNT,
        channel: int = 512,
        num_head: int = 8,
        temperature_coeff: float = 2.0,
        n_encoder: int = 6,
        n_decoder: int = 6,
        n_readout: int = 2,
        n_diffuse: int = 6,
        num_task: int = 1,
        label_mode: str = "value:2",
    ) -> None:
        """
        AkAne (AsymmetriC AutoeNcodEr) model.

        :param n_vocab: number of vocabulary
        :param channel: number of atom basis
        :param num_head: number of heads per attention block
        :param temperature_coeff: attention temperature coefficient
        :param n_encoder: number of encoder layers
        :param n_decoder: number of decoder layers
        :param n_readout: number of read-out layers
        :param n_diffuse: number of DiT layers
        :param num_task: number of task output
        :param label_mode: label mode chosen from 'value:x', 'class:x' and 'text:x',
                           where x is the `maximal length` of label, e.g.,
                           if the label is consisted of 5 numbers, then 'value:5'
                           if the label is 3 classes, then 'class:3'
                           if the label is text-like with a vocabulary of 45 unique words, then 'text:45'
        """
        super().__init__()
        self.encoder = Encoder(channel, n_encoder, num_head, temperature_coeff)
        self.decoder = Decoder(n_vocab, channel, n_decoder, num_head, temperature_coeff)
        self.readout = Readout(channel, n_readout, num_task)
        self.dit = DiT(channel, n_diffuse, num_head, temperature_coeff, label_mode)
        self.diffusion_method = GaussianDiffusion()
        self.vocab_key = deepcopy(VOCAB_KEYS)  # required by TorchScript
        self.fs_mask = deepcopy(FS_MASK)  # required by TorchScript

    def autoencoder_train_step(
        self, mol: Dict[str, Tensor], token_input: Tensor, token_label: Tensor
    ) -> Tensor:
        """
        :param mol: molecule = {
            "node": node matrix;  shape: (n_b, n_a, 62)
            "edge": edge matrix;  shape: (n_b, n_a, n_a, 30)
        }
        :param token_input: input tokens to decoder;  shape: (n_b, n_t, n_f)
        :param token_label: label tokens to compare;  shape: (n_b, n_t, n_f)
        :return: cross entropy loss
        """
        self.readout.requires_grad_(False)
        self.dit.requires_grad_(False)
        h, node_mask = self.encoder(mol)  # graph --> latent vectors (l.v)
        z, mask = self.decoder(token_input, h, node_mask)  # l.v. --> tokens
        z = z.reshape(-1, z.shape[-1])
        token_label = token_label.reshape(-1)
        loss = F.cross_entropy(z, token_label, reduction="none") * mask.reshape(-1)
        return loss.sum() / mask.sum() + self.encoder.kl

    def predict_train_step(self, mol: Dict[str, Tensor], label: Tensor) -> Tensor:
        """
        :param mol: molecule = {
            "node": node matrix;  shape: (n_b, n_a, 62)
            "edge": edge matrix;  shape: (n_b, n_a, n_a, 30)
        }
        :param label: property label(s);  shape: (n_b, n_l)
        :return: MSE loss
        """
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.dit.requires_grad_(False)
        h, _ = self.encoder(mol, True)
        y = self.readout(h)
        label_mask = (label != torch.inf).float()  # find the unlabelled position(s)
        # masked the unlabelled position(s)
        label = label.masked_fill(label == torch.inf, 0)
        return F.mse_loss(y * label_mask, label, reduction="mean")

    def classify_train_step(self, mol: Dict[str, Tensor], label: Tensor) -> Tensor:
        """
        :param mol: molecule = {
            "node": node matrix;  shape: (n_b, n_a, 62)
            "edge": edge matrix;  shape: (n_b, n_a, n_a, 30)
        }
        :param label: property label(s);  shape: (n_b, 1)
        :return: MSE loss
        """
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.dit.requires_grad_(False)
        h, _ = self.encoder(mol, True)
        y = self.readout(h)
        return F.cross_entropy(y, label.reshape(-1))

    def diffusion_train_step(self, mol: Dict[str, Tensor], label: Tensor) -> Tensor:
        """
        :param mol: molecule = {
            "node": node matrix;  shape: (n_b, n_a, 62)
            "edge": edge matrix;  shape: (n_b, n_a, n_a, 30)
        }
        :param label: property label(s);  shape: (n_b, n_l)
        :return: MSE loss
        """
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.readout.requires_grad_(False)
        h, mask = self.encoder(mol)
        time = torch.randint(1000, (h.shape[0], 1), device=h.device)
        loss = self.diffusion_method.train_losses(self.dit, h, time, label, mask)
        return loss

    @torch.jit.export
    def inference(self, mol: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Predict properties from the graph(s) of molecule(s).

        :param mol: molecule = {
            "node": node matrix;  shape: (n_b, n_a, 62)
            "edge": edge matrix;  shape: (n_b, n_a, n_a, 30)
        }
        :return: {"prediction": properties}
        """
        h, _ = self.encoder(mol, True)
        y = self.readout(h)
        return {"prediction": y}

    @torch.jit.export
    def translate(
        self, mol: Dict[str, Tensor], beam_size: int = 5
    ) -> Dict[str, List[str]]:
        """
        Translate a graph of molecule(s) to SMILES string.

        :param mol: molecule = {
            "node": node matrix;  shape: (n_b, n_a, 62)
            "edge": edge matrix;  shape: (n_b, n_a, n_a, 30)
        }
        :param beam_size: number of beam search branch(es)
        :return: {"SMILES": [SMILES strings]}
        """
        h, m = self.encoder(mol)
        smiles = self._beam_search(h, m, beam_size)
        return {"SMILES": smiles}

    @torch.jit.export
    def generate(self, size: List[int], label: Tensor) -> Dict[str, List[str]]:
        """
        Generate molecule(s) from noise.

        :param size: [batch-size, target number of atoms in the molecule(s), beam-size]
        :param label: labels values;  shape: (1, n_l)
        :return: {"SMILES": [SMILES strings]}
        """
        assert len(size) == 3
        h_noisy = torch.randn(
            (size[0], size[1], self.decoder.position.d), device=label.device
        )
        m = torch.ones((size[0], size[1], 1), device=label.device)
        assert label.shape[0] == 1 or label.shape[0] == size[0]
        if label.shape[0] == 1:
            label = label.repeat(size[0], 1)
        h = self.diffusion_method.sample(self.dit, h_noisy, m, label)
        smiles = self._beam_search(h, m, size[2])
        return {"SMILES": smiles}

    def _greedy_search(self, y: Tensor, y_mask: Tensor) -> List[str]:
        """
        Greedy search until all entities meet the <esc> token.\n
        'Greediness is good.'

        :param y: node representation from encoder;  shape: (n_b, n_a, n_f)
        :param y_mask: y mask;                       shape: (n_b, n_a, 1)
        :return: a `list` instance containing SMILES string(s)
        """
        n_b = y.shape[0]
        count = torch.zeros((n_b, 1), device=y.device)
        token = torch.ones((n_b, 1), dtype=torch.long, device=y.device)
        smiles: List[str] = []
        while True:
            out, _ = self.decoder.forward(token, y, y_mask)
            out = torch.argmax(F.softmax(out, -1), -1)
            last_out = out[::, -1, None]
            count += torch.where(last_out == 2, 1, 0)
            if (count == 0).float().sum() == 0:
                break
            token = torch.cat([token, last_out], -1)
        for i in token:
            s = "".join([self.vocab_key[j.item()] for j in i[1:]])
            smiles.append(s.split("<esc>")[0])
        return smiles

    def _beam_search(self, y: Tensor, y_mask: Tensor, beam_size: int = 5) -> List[str]:
        """
        Beam search until all branches meet <esc> token.

        :param y: node representation from encoder;  shape: (n_b, n_a, n_f)
        :param y_mask: y mask;                       shape: (n_b, n_a, 1)
        :param beam_size: number of beam search branches
        :return: a `list` instance containing SMILES string(s)
        """
        n_b, n_a, n_f = y.shape
        mask = torch.tensor(self.fs_mask, device=y.device)[None, None, :]
        token = torch.ones((n_b, 1), dtype=torch.long, device=y.device)
        smiles: List[str] = []
        out, _ = self.decoder.forward(token, y, y_mask)
        out = F.softmax(out.masked_fill_(mask == 0, -torch.inf), -1)
        n_vocab = out.shape[-1]
        p = out.view(n_b, -1).topk(beam_size, -1)
        last_out = p.indices.view(-1, 1)
        prob = p.values.view(-1, 1)
        count = torch.where(last_out == 2, 1, 0)
        token = torch.cat([token.repeat(beam_size, 1), last_out], -1)
        y = y[:, None, ...].repeat(1, beam_size, 1, 1).view(-1, n_a, n_f)
        y_mask = y_mask[:, None, ...].repeat(1, beam_size, 1, 1).view(-1, n_a, 1)
        _id = torch.arange(beam_size * n_b, dtype=torch.long, device=y.device)
        _id = _id[:, None].repeat(1, n_vocab).flatten()
        id: Optional[Tensor] = None
        n = torch.arange(n_b, device=y.device)[:, None].repeat(1, beam_size) * beam_size
        c = n.flatten() * n_vocab  # stepped residual values of last_idx
        while True:
            out, _ = self.decoder.forward(token, y, y_mask)
            p = F.softmax(out, -1)[::, -1].reshape(n_b, -1).topk(beam_size, -1)
            last_idx = p.indices.flatten()
            if id is None:
                id = _id[last_idx + c]
            else:
                # lock the batches whose branches have all met <esc> token
                id_mask = ((count.view(n_b, -1) == 0).sum(1, True) == 0).float()
                id_mask = id_mask.repeat(1, beam_size).flatten()
                id = (_id[last_idx + c] * (1 - id_mask) + id * id_mask).to(torch.long)
            token = token[id]
            count = count[id]
            prob = prob[id] * p.values.view(-1, 1)
            last_out = torch.fmod(last_idx, n_vocab).view(beam_size * n_b, 1)
            count += torch.where(last_out == 2, 1, 0)
            if (count == 0).float().sum() == 0:
                break
            token = torch.cat([token, last_out], -1)
        # sort the outputs from highest total probability to the lowest one
        sort_idx = torch.sort(prob.view(n_b, -1), -1, True).indices
        token = token[sort_idx + n].view(token.shape)
        for i in token:
            s = "".join([self.vocab_key[j.item()] for j in i[1:]])
            smiles.append(s.split("<esc>")[0])
        return smiles

    def pretrained(self, file: str) -> Self:
        """
        Load pre-trained model.

        :param file: checkpoint file name <file>
        :return: loaded model
        """
        with open(file, mode="rb") as f:
            state_dict = torch.load(f, map_location="cpu")["nn"]
        if "encoder" in state_dict:
            self.encoder.load_state_dict(state_dict["encoder"])
        if "decoder" in state_dict:
            self.decoder.load_state_dict(state_dict["decoder"])
        if "readout" in state_dict:
            self.readout.load_state_dict(state_dict["readout"])
        if "dit" in state_dict:
            self.dit.load_state_dict(state_dict["dit"])
        return self


class Kamome(nn.Module):
    def __init__(
        self,
        n_atom_basis: int = 256,
        n_interaction: int = 6,
        num_head: int = 8,
        temperature_coeff: float = 2.0,
        n_readout: int = 2,
        num_task: int = 1,
        add_adjacency_matrix_into_attn: bool = True,
    ) -> None:
        """
        Kamome model.

        :param n_atom_basis: number of atom basis
        :param n_interaction: number of interaction blocks
        :param num_head: number of heads per self-attention block
        :param temperature_coeff: attention temperature coefficient
        :param n_readout: number of read-out layers
        :param num_task: number of task output
        :param add_adjacency_matrix_into_attn: whether to add the adjacency into attention machenism
        """
        super().__init__()
        self.add_adj = add_adjacency_matrix_into_attn
        self.embedding = nn.Linear(
            in_features=62, out_features=n_atom_basis, bias=False
        )
        self.interaction = nn.ModuleList(
            [
                Interaction(
                    hidden_dim=n_atom_basis,
                    num_head=num_head,
                    temperature_coeff=temperature_coeff,
                )
                for _ in range(n_interaction)
            ]
        )
        self.readout = MLP(hidden_dim=n_atom_basis, n_layer=n_readout)
        self.output = nn.Linear(
            in_features=n_atom_basis, out_features=num_task, bias=True
        )

    def forward(
        self,
        mol: Dict[str, Tensor],
        return_attn_matrix: bool = False,
        average_attn_matrix: bool = True,
    ) -> Dict[str, Tensor]:
        """
        :param mol: molecule = {
            "node": node matrix;  shape: (n_b, n_a, 62)
            "edge": edge matrix;  shape: (n_b, n_a, n_a, 30)
        }
        :param return_adj_matrix: whether to return the attention matrices
        :param average_attn_matrix: whether to average the returned attention matrix over layers
        :return: molecular properties (e.g. CMC)
        """
        h, e = mol["node"], mol["edge"]
        node_mask = (h.sum(dim=-1) != 0).float().unsqueeze(dim=-1)
        edge_mask = (e.sum(dim=-1) != 0).float().unsqueeze(dim=-1)
        attention_mask = (node_mask + node_mask.transpose(-2, -1) == 1).unsqueeze(dim=0)
        attn_matrix_m = (
            edge_mask.squeeze(dim=-1).unsqueeze(dim=0) if self.add_adj else None
        )
        attns = []
        h = self.embedding(h)
        for layer in self.interaction:
            h, attn = layer(
                h,
                e,
                node_mask,
                edge_mask,
                attention_mask,
                attn_matrix_m,
                return_attn_matrix,
            )
            if attn is not None:
                attns.append(attn)
        y = h.sum(dim=-2)  # sum pooling
        y = self.readout(y)
        y = self.output(y)
        if return_attn_matrix:
            attn = torch.cat(attns, dim=0)
            if average_attn_matrix:
                attn = attn.mean(dim=0)
            return {
                "prediction": y,
                "attn_matrix": attn,
            }
        return {"prediction": y}

    def pretrained(self, file: str) -> Self:
        with open(file, mode="rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        self.load_state_dict(state_dict=state_dict["nn"])
        return self
