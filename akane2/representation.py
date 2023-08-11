# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Model representation
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils.token import VOCAB_COUNT, VOCAB_KEYS
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
        :param num_head: number of heads per self-attention block
        :param temperature_coeff: attention temperature coefficient
        :param n_encoder: number of encoder layers
        :param n_decoder: number of decoder layers
        :param n_readout: number of read-out layers
        :param n_diffuse: number of DiT layers
        :param num_task: number of task output
        :param label_mode: label mode chosen from 'value:x' and 'text:x'
        """
        super().__init__()
        self.encoder = Encoder(channel, n_encoder, num_head, temperature_coeff)
        self.decoder = Decoder(n_vocab, channel, n_decoder, num_head, temperature_coeff)
        self.readout = Readout(channel, n_readout, num_task)
        self.dit = DiT(channel, n_diffuse, num_head, temperature_coeff, label_mode)
        self.diffusion_method = GaussianDiffusion()

    def autoencoder_train_step(
        self, mol: Dict[str, Tensor], token_input: Tensor, token_label: Tensor
    ) -> Tensor:
        """
        :param mol: molecule = {
            "node": node matrix;  shape: (n_b, n_a, n_f)
            "edge": edge matrix;  shape: (n_b, n_a, n_a, n_f)
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
            "node": node matrix;  shape: (n_b, n_a, n_f)
            "edge": edge matrix;  shape: (n_b, n_a, n_a, n_f)
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
        label = label.masked_fill(
            label == torch.inf, 0
        )  # masked the unlabelled position(s)
        return F.mse_loss(y * label_mask, label, reduction="mean")

    def classify_train_step(self, mol: Dict[str, Tensor], label: Tensor) -> Tensor:
        """
        :param mol: molecule = {
            "node": node matrix;  shape: (n_b, n_a, n_f)
            "edge": edge matrix;  shape: (n_b, n_a, n_a, n_f)
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
            "node": node matrix;  shape: (n_b, n_a, n_f)
            "edge": edge matrix;  shape: (n_b, n_a, n_a, n_f)
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

    @torch.no_grad()
    def inference(self, mol: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Predict properties from the graph(s) of molecule(s).

        :param mol: molecule = {
            "node": node matrix;  shape: (n_b, n_a, n_f)
            "edge": edge matrix;  shape: (n_b, n_a, n_a, n_f)
        }
        :return: {"prediction": properties}
        """
        h, _ = self.encoder(mol, True)
        y = self.readout(h)
        return {"prediction": y}

    @torch.no_grad()
    def translate(self, mol: Dict[str, Tensor]) -> Dict[str, str]:
        """
        Translate a graph of molecule(s) to SMILES string.

        :param mol: molecule = {
            "node": node matrix;  shape: (1, n_a, n_f)
            "edge": edge matrix;  shape: (1, n_a, n_a, n_f)
        }
        :return: {"SMILES": SMILES string}
        """
        h, m = self.encoder(mol)
        # <start> token
        token = torch.tensor([[1]], dtype=torch.long, device=h.device)
        smiles = []
        while True:
            out, _ = self.decoder(token, h.clone(), m)
            out = nn.functional.softmax(out, dim=-1)
            out = torch.argmax(out, dim=-1)
            last_out = out[0][-1]
            if last_out == 2:  # <esc> token
                return {"SMILES": "".join(smiles)}
            token = torch.cat([token, last_out.reshape(1, 1)], dim=-1)
            s = VOCAB_KEYS[last_out.detach().item()]
            smiles.append(s)

    @torch.no_grad()
    def generate(
        self, size: int, label: Tensor, progress_bar: bool = True
    ) -> Dict[str, str]:
        """
        Generate molecule(s) from noise.

        :param size: traget number of atoms in the molecule(s)
        :param label: labels values;  shape: (1, n_l)
        :param progress_bar: whether to show the progress bar
        :return: {"SMILES": SMILES string}
        """
        h_noisy = torch.randn(1, size, self.decoder.position.d, device=label.device)
        m = torch.ones(1, size, 1, device=label.device)
        h = self.diffusion_method.sample(self.dit, h_noisy, m, label, progress_bar)
        # <start> token
        token = torch.tensor([[1]], dtype=torch.long, device=h.device)
        smiles = []
        while True:
            out, _ = self.decoder(token, h.clone(), m)
            out = nn.functional.softmax(out, dim=-1)
            out = torch.argmax(out, dim=-1)
            last_out = out[0][-1]
            if last_out == 2:  # <esc> token
                return {"SMILES": "".join(smiles)}
            token = torch.cat([token, last_out.reshape(1, 1)], dim=-1)
            s = VOCAB_KEYS[last_out.detach().item()]
            smiles.append(s)

    def pretrained(self, file: str) -> nn.Module:
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
            "node": node matrix;  shape: (n_b, n_a, n_f)
            "edge": edge matrix;  shape: (n_b, n_a, n_a, n_f)
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
            if return_attn_matrix:
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

    def pretrained(self, file: str) -> nn.Module:
        with open(file, mode="rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        self.load_state_dict(state_dict=state_dict["nn"])
        return self
