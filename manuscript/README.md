# Manuscript

LaTeX source for the paper accompanying the Java reference implementation.

## Build

```bash
cd manuscript/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or, with `latexmk`:

```bash
latexmk -pdf main.tex
```

## Files

- `main.tex` — manuscript source (~12 pages targeting *Expert Systems with Applications*)
- `references.bib` — BibTeX references (24 entries)

## Empirical numbers in the manuscript

All numbers in Tables 1, 2, and 3 are **placeholders** carried forward from a
preliminary development run. After you complete a local Java run via:

```bash
mvn clean package
java -jar target/ach-decisioning-1.0.0.jar --mode=experiment
```

the file `output/results.json` will contain the canonical AUROC, Operational
Cost, and latency numbers. Replace the placeholder values in `main.tex`
(search for the lines marked `\textbf{Placeholder values; ...}`) with those
canonical numbers before submission.

## Figures

Seven figures (`fig1_dataset_labels.png` through `fig7_*`) are emitted by
the Java pipeline into `output/figures/` upon running the experiment mode.
The current `main.tex` references them inline; ensure the figures directory
is on the `\graphicspath` or copy them into `manuscript/figures/`.

## Target venue

Primary: **Expert Systems with Applications** (Elsevier, Q1, IF ~8.5,
Scopus-indexed)

Alternate: **IEEE Transactions on Services Computing** (Q1, Scopus-indexed)

Both venues fit the contribution profile: applied AI in a domain-specific
service architecture, with empirical evaluation and reproducible artifacts.
