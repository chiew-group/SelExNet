<h1 align="center"> [MRM 2026] SelExNet: A Self-Supervised Physics-Informed Framework for Multi-Channel Joint RF and Gradient Waveform Optimization in 2D Spatially Selective Excitation </h1>
<p align="center">
<!-- <a href="https://aao-hnsfjournals.onlinelibrary.wiley.com/doi/10.1002/ohn.868"><img src="https://img.shields.io/badge/Wiley-Paper-red"></a> -->
<!-- <a href="https://pubmed.ncbi.nlm.nih.gov/38922721/"><img src="https://img.shields.io/badge/PubMed-Link-blue"></a> -->
<a href="https://github.com/chiew-group/SelExNet"><img src="https://img.shields.io/badge/Code-Page-magenta"></a>
</p>
<h5 align="center">Yuliang Xiao<sup>#</sup>, Jason Rock, Zhe Wu, Jamie Near, Mark Chiew, Simon J. Graham</em></h5>
<p align="center"> <sup>#</sup> Indicates Corresponding Author </p>
<p align="center">
  <a href="#news">News</a> |
  <a href="#abstract">Abstract</a> |
  <a href="#installation">Installation</a> |
  <a href="#train">Train</a> |
  <a href="#finetune">Finetune</a>
</p>

## News 

**2026.05.02** - Our paper is accepted by **Magnetic Resonance in Medicine 2026 (MRM 2026)**.

## Abstract
SelExNet is a self-supervised framework for 2D spatially selective excitation that jointly optimizes radiofrequency (RF) pulses and gradient waveforms, with extension to multi-channel transmission MRI. It couples neural RF and gradient generators with a differentiable Bloch simulator, enabling pulse optimization directly from desired excitation outcomes without requiring pre-designed target pulses.

The framework designs RF pulses and parameterized variable-density spiral gradient waveforms for both single- (sTx) and multi-channel (pTx) transmission, and supports patient-specific adaptation using measured, previously unseen <em>B</em><sub>0</sub> and
                <em>B</em><sub>1</sub><sup>+</sup> maps. Joint RF-Gradient optimization improves excitation fidelity over RF-only optimization. In phantom experiments, patient-specific fine-tuning restores target geometry and uniformity under field inhomogeneity. In-vivo studies demonstrate anatomically precise excitation, sharper target boundaries, and reduced off-target signal.

## Installation

## Train

## Finetune
