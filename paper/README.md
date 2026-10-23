# Academic Research Manuscript & Publication Suite 📄

This folder contains the complete, publication-ready research manuscript prepared for **IEEE Transactions on Medical Imaging (TMI)**.

## Manuscript Information
- **Title**: *Explainable Deep Foundation and Hybrid Vision Architectures for Multi-Organ Medical Image Diagnostics at the Edge and Centralized Clinical PACS*
- **Target Journal**: IEEE Transactions on Medical Imaging (Vol. 45, No. 10)
- **Primary Source File**: [`research_paper.tex`](file:///home/faith/folder/AI_medical_image_processing_multiorgan/paper/research_paper.tex)

## Figures
The manuscript incorporates publication-grade diagnostic figures (confusion matrices, ROC curves, multi-panel XAI diagnostic sheets, executive dashboard). Figures are located in the project's [`figures/`](file:///home/faith/folder/AI_medical_image_processing_multiorgan/figures) and [`outputs/`](file:///home/faith/folder/AI_medical_image_processing_multiorgan/outputs) directories.

## Compilation Instructions

### Using LaTeX CLI (`pdflatex` / `latexmk`):
From the project root:
```bash
cd paper
pdflatex research_paper.tex
bibtex research_paper
pdflatex research_paper.tex
pdflatex research_paper.tex
```

### Using Overleaf:
1. Upload `research_paper.tex`.
2. Upload the `figures/` directory containing the referenced visual artifacts.
3. Select the `IEEEtran` compiler engine.
