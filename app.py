# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omomzawa Sueno)
"""
Web UI
"""
import math
import argparse
from pathlib import Path
from rdkit.Chem import Draw, MolFromSmiles
from mol2chemfigPy3 import mol2chemfig
import gradio as gr
import torch
from akane2.representation import Kamome, AkAne
from akane2.utils.graph import smiles2graph, gather
from akane2.utils.token import protein2vec

ptk = Path(__file__).parent / "model_kamome"
pta = Path(__file__).parent / "model_akane"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = Kamome().pretrained(ptk / "moleculenet/esol.pt").eval().to(device)
model2 = (
    Kamome(num_head=2).pretrained(ptk / "moleculenet/freesolv.pt").eval().to(device)
)
model3 = Kamome().pretrained(ptk / "moleculenet/lipo.pt").eval().to(device)
model4 = Kamome().pretrained(ptk / "qm9/qm9_homo.pt").eval().to(device)
model5 = Kamome().pretrained(ptk / "qm9/qm9_lumo.pt").eval().to(device)
model6 = Kamome().pretrained(ptk / "qm9/qm9_gap.pt").eval().to(device)
model7 = Kamome().pretrained(ptk / "qm9/qm9_zpve.pt").eval().to(device)
model8 = Kamome().pretrained(ptk / "qm9/qm9_u0.pt").eval().to(device)
model9 = Kamome().pretrained(ptk / "qm9/qm9_u.pt").eval().to(device)
model10 = Kamome().pretrained(ptk / "qm9/qm9_g.pt").eval().to(device)
model11 = Kamome().pretrained(ptk / "qm9/qm9_h.pt").eval().to(device)
model12 = Kamome().pretrained(ptk / "qm9/qm9_cv.pt").eval().to(device)
model13 = Kamome().pretrained(ptk / "photoswitch/e_iso_n.pt").eval().to(device)
model14 = Kamome().pretrained(ptk / "photoswitch/e_iso_pi.pt").eval().to(device)
model15 = Kamome().pretrained(ptk / "photoswitch/z_iso_n.pt").eval().to(device)
model16 = Kamome().pretrained(ptk / "photoswitch/z_iso_pi.pt").eval().to(device)

model17 = (
    AkAne(label_mode="text:23").pretrained(pta / "bind_generate.pt").eval().to(device)
)
model18 = (
    AkAne(label_mode="value:2").pretrained(pta / "des_generate.pt").eval().to(device)
)


def _count_iso(mol):
    counter = []
    bonds = mol.GetBonds()
    for bond in bonds:
        type = int(bond.GetBondType())
        if type == 2:
            stereo = int(bond.GetStereo())
            if stereo == 2 or stereo == 4:
                counter.append("Z")
            elif stereo == 3 or stereo == 5:
                counter.append("E")
            else:
                atom_b, atom_e = bond.GetBeginAtom(), bond.GetEndAtom()
                degree_b, degree_e = atom_b.GetDegree(), atom_e.GetDegree()
                if degree_b != 1 and degree_e != 1:
                    counter.append("any")
    return set(counter)


@torch.no_grad()
def process0(smiles: str):
    mol = MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, (500, 500))
    counter = _count_iso(mol)
    mol = gather([smiles2graph(smiles)])
    mol["node"] = mol["node"].to(device)
    mol["edge"] = mol["edge"].to(device)
    v1 = model1(mol)["prediction"].item()
    v1 = math.pow(10, v1)
    v2 = model2(mol)["prediction"].item()
    v3 = model3(mol)["prediction"].item()
    v4 = model4(mol)["prediction"].item()
    v5 = model5(mol)["prediction"].item()
    v6 = model6(mol)["prediction"].item()
    v7 = model7(mol)["prediction"].item()
    v8 = model8(mol)["prediction"].item()
    v9 = model9(mol)["prediction"].item()
    v10 = model10(mol)["prediction"].item()
    v11 = model11(mol)["prediction"].item()
    v12 = model12(mol)["prediction"].item()
    if "E" in counter:
        v13 = model13(mol)["prediction"].item()
        v14 = model14(mol)["prediction"].item()
    elif "Z" in counter:
        v13 = model15(mol)["prediction"].item()
        v14 = model16(mol)["prediction"].item()
    elif "any" in counter:
        v13 = v14 = "please specify the double bond stereochemistry in SMILES."
    else:
        v13 = v14 = "n.a."
    return img, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14


@torch.no_grad()
def process1(size, file):
    with open(file.name, "r", encoding="utf-8") as f:
        data = f.readlines()
    fasta = data[1]
    label = torch.tensor([protein2vec(fasta)], device=device)
    while True:
        smiles = model17.generate(int(size), label, False)
        mol = MolFromSmiles(smiles["SMILES"])
        if mol != None:
            break
    img = Draw.MolToImage(mol, (500, 500))
    smiles = smiles["SMILES"]
    chemfig = mol2chemfig(smiles, "-r", inline=True)
    return img, smiles, chemfig


@torch.no_grad()
def process2(size, x, mp):
    label = torch.tensor([[float(x), float(mp)]], device=device)
    while True:
        smiles = model18.generate(int(size), label, False)
        mol = MolFromSmiles(smiles["SMILES"])
        if mol != None:
            break
    img = Draw.MolToImage(mol, (500, 500))
    smiles = smiles["SMILES"]
    chemfig = mol2chemfig(smiles, "-r", inline=True)
    return img, smiles, chemfig


with gr.Blocks(title="AkAne") as app:
    gr.Markdown(
        "### This model is part of MSc Electrochemistry and Battery Technologies project (2022 - 2023), University of Southampton"
    )
    gr.Markdown("Author: Nianze Tao (Omozawa Sueno)")
    gr.Markdown("---")
    with gr.Tab(label="MOLECULAR PROPERTY PREDICTION"):
        with gr.Row():
            with gr.Column(scale=1):
                smiles0 = gr.Textbox(label="SMILES", placeholder="input SMILES here")
                btn0 = gr.Button("RUN", variant="primary")
                img0 = gr.Image(label="molecule")
            with gr.Column(scale=2):
                with gr.Tab(label="EXP"):
                    gr.Markdown(
                        "Predicted with A<span style='color:#CB4154'>k</span>Ane from MoleculeNet training data."
                    )
                    v1 = gr.Textbox(label="solubility / M")
                    v2 = gr.Textbox(label="solvation energy / kcal/mol")
                    v3 = gr.Textbox(label="lipophilicity (logD)")
                    gr.Markdown(
                        "Predicted with A<span style='color:#CB4154'>k</span>Ane from PhotoSwitch training data."
                    )
                    v13 = gr.Textbox(label="n-π* transition wavelength / nm")
                    v14 = gr.Textbox(label="π-π* transition wavelength / nm")
                with gr.Tab(label="QM"):
                    gr.Markdown(
                        "Predicted with A<span style='color:#CB4154'>k</span>Ane from QM9 training data."
                    )
                    v4 = gr.Textbox(label="HOMO / Hartree")
                    v5 = gr.Textbox(label="LUMO / Hartree")
                    v6 = gr.Textbox(label="HOMO-LUMO gap / Hartree")
                    v7 = gr.Textbox(label="zero-point vibrational energy / Hartree")
                    v8 = gr.Textbox(label="U at 0K / Hartree")
                    v9 = gr.Textbox(label="U at 298.15K / Hartree")
                    v10 = gr.Textbox(label="G at 298.15K / Hartree")
                    v11 = gr.Textbox(label="H at 298.15K / Hartree")
                    v12 = gr.Textbox(label="heat capacity / kcal/K/mol")
    with gr.Tab(label="DE NOVO STRUCTURE GENERATION"):
        with gr.Tab(label="protein ligand design"):
            gr.Markdown(
                "*de novo* ligand desiged with A<span style='color:#CB4154'>k</span>Ane from BindingDB training data."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    size1 = gr.Slider(10, 100, value=24, step=1, label="molecule size")
                    file = gr.File(file_types=[".fasta"], label="FASTA sequence")
                    btn1 = gr.Button("RUN", variant="primary")
                    img1 = gr.Image(label="molecule")
                with gr.Column(scale=2):
                    smiles1 = gr.Textbox(label="SMILES")
                    chemfig1 = gr.TextArea(label="LATEX ChemFig")
        with gr.Tab(label="deep eutective solvent design"):
            gr.Markdown(
                "*de novo* deep eutective solvent desiged with A<span style='color:#CB4154'>k</span>Ane."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    size2 = gr.Slider(5, 100, value=12, step=1, label="molecule size")
                    x = gr.Slider(0.1, 0.9, label="x(HBA)", value=0.5)
                    mp = gr.Slider(
                        -120, 200, label="melting point / °C", value=-20, step=0.2
                    )
                    btn2 = gr.Button("RUN", variant="primary")
                    img2 = gr.Image(label="molecule")
                with gr.Column(scale=2):
                    smiles2 = gr.Textbox(label="SMILES")
                    chemfig2 = gr.TextArea(label="LATEX ChemFig")
    btn0.click(
        fn=process0,
        inputs=smiles0,
        outputs=[img0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14],
    )
    btn1.click(fn=process1, inputs=[size1, file], outputs=[img1, smiles1, chemfig1])
    btn2.click(fn=process2, inputs=[size2, x, mp], outputs=[img2, smiles2, chemfig2])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", default=False, type=bool, help="open to public")
    args = parser.parse_args()
    app.launch(share=args.public)
