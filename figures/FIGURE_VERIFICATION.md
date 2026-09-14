# Figure verification record

## Scope

This record distinguishes numerical reproducibility from raster-pixel identity.
The committed publication assets were regenerated from archived experiment CSV
files without retraining models or changing results.

## Figure 1: training dynamics

- The historical plotting source at commit
  `7657a0cb41fb99ab16bd5071c863cfd6c9cde26b` runs successfully against the
  archived nine `curves.csv` files and recreates the legacy 870 x 635 PNG
  layout.
- Its aggregation is unchanged: at each model, evaluation step, and band it
  plots the median over ten seeds with first and third quartiles as the shaded
  interval.
- The publication script preserves that aggregation and changes only rendering
  parameters: final width, text sizing, font embedding, and output format.
- The legacy and regenerated raster PNGs are not asserted to be byte- or
  pixel-identical. Rasterization can differ with font rendering and graphics
  context even when the source arrays and plotting operations are the same.

## Figure 2: prediction profiles

- No historical dedicated Figure 2 script was recovered from the local
  workspaces or repository history.
- `scripts/plot_prediction_profiles.py` reconstructs the displayed quantity
  directly from the archived final `predictions.csv` records. It validates ten
  seeds per displayed model and every index from 0 through 60 before writing
  any figure.
- The reconstruction retains the five displayed models, the median-over-seeds
  summaries, the sign-preserving `log10(abs(value) + 1)` prediction view, the
  future indices 31--60, the near/mid/far bands, and the `5e-2` far-success
  threshold visible in the legacy figure.
- Because the historical Figure 2 renderer is unavailable and the publication
  layout is deliberately different, no byte- or pixel-identity claim is made
  for this figure.

## Publication-asset checks

- Both PNG exports are 1200 dpi at 122 mm width.
- Both PDF exports are vector PDFs with embedded TrueType fonts.
- Both SVG exports are vector files and contain no trailing whitespace.
