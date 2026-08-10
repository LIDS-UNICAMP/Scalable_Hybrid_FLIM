# Reading the SVM probe curves

W&B shows one kappa series during autoencoder pretraining, `stage{N}/svm_kappa`
(`stage1/` or `stage2/`, depending on the stage), read against a single reference number:
the summary scalar `stage{N}/flim_ref_svm_kappa`, written once before training.

This note explains where each one comes from and why they differ.
All code references are to [`src/modules/autoencoder_flim_module.py`](../src/modules/autoencoder_flim_module.py).

## What the probe is

The autoencoder trains without labels, on reconstruction loss alone. To know whether the
embedding is getting *better* — not just reconstructing better — a linear SVM is fitted on
the labelled train split and scored on validation. That is the probe. It never enters the
loss; it only reports.

Both numbers come from the same function, `_svm_probe()`, which takes four
arguments: `X_tr, y_tr` (fit the SVM) and `X_val, y_val` (score it, producing kappa).

## Reference and curve are the same computation, run at different times

| | `stage{N}/flim_ref_svm_kappa` | `stage{N}/svm_kappa` |
|---|---|---|
| computed in | `on_fit_start()` — **once**, before any gradient step | `on_validation_epoch_end()` — **every epoch** |
| `X_tr` from | `_train_embeddings()` | `_train_embeddings()` |
| `X_val` from | `val_dataloader()` | `_val_emb` accumulated in `validation_step` |
| stored as | `self.baseline_metrics`, written to the W&B summary | logged as a per-epoch series |

`stage{N}/flim_ref_svm_kappa` is a single number, captured before training. It lives in the
W&B **summary**, not as a series: replayed every epoch it was bit-identical to the stage-1
kappa curve, two names for one number.

## Why the live curve moves

The validation side is fixed. `val_dataloader` has `shuffle=False` and, with `V_eval=1`,
the dataset uses the deterministic test transform — so `X_val` is identical every epoch.

The train side is not. `train_dataloader` has `shuffle=True`, so every call to
`_train_embeddings()` returns the same images and labels **in a different row order**.

This mattered because of how the probe *used* to be configured: `C=1e2`, no scaler,
`max_iter=10000`. When libsvm's SMO does not converge within that iteration cap, it stops
at an arbitrary point along the optimisation path, and that point depends on the order of
the training rows. Same matrix, permuted rows, different kappa.

The probe now runs **`max_iter=-1`** (`_svm_probe`), so libsvm runs to its own stopping
criterion and the fit is order-invariant. `C=1e2` and the absence of a scaler are unchanged.
The price is stated in `_svm_probe`'s docstring: this curve is not numerically comparable
with the CSVs written by the capped evaluators.

A synthetic check with the old, capped hyperparameters:

- solver hits `max_iter` (`fit_status_ = 1`): permuting the training order alone moved
  kappa across a range of 0.31
- solver converges (`fit_status_ = 0`): permuting changed nothing — identical predictions

So the movement is the probe, not the encoder.

## Stage 1 vs stage 2

**Stage 1 (`--freeze-encoder`).** The encoder is frozen: `requires_grad = False` on every
encoder parameter, and the optimizer filters on `requires_grad`. The encoder is
plain `Conv2d + ReLU + MaxPool` with no BatchNorm, so there is no running statistic
drifting behind the freeze either. The probe embeds with `src.utils.evaluate._encode_pooled`
— the official evaluator's own function — and nothing else.

The embedding is therefore bit-for-bit constant across epochs, and `stage1/svm_kappa`
*should* sit exactly on `stage1/flim_ref_svm_kappa`. Any oscillation you see in stage 1 is
pure probe noise — by construction, there is nothing the encoder could have learned.

This makes stage 1 the right place to measure how big that noise is.

**Stage 2 (encoder unfrozen).** Now `stage2/svm_kappa` moves for two reasons at once: probe
noise *and* real encoder learning. The two are summed into one curve and cannot be
separated by eye. Read it alongside `stage2/flim_drift` — a rising kappa with an exploding
drift does not mean the FLIM encoder adapted, it means it was overwritten.

## The kappa delta

`svm_kappa_delta_best` — best `stage{N}/svm_kappa` minus `stage{N}/flim_ref_svm_kappa` — is a
**summary scalar and a `run_metadata.json` key**, not a per-epoch curve: as a series it was
`svm_kappa` shifted by a constant. Two cautions:

1. **It inherits the baseline's noise.** The baseline is a single draw at one shuffle order,
   promoted to a fixed reference. A different draw would move every delta in the run.
2. **It is not comparable across runs.** The headroom depends entirely on where the run
   started: with `kappa_base = 0.82` the delta cannot exceed `+0.18`, and reaching it means
   zero errors; with `kappa_base = 0.01` a `+0.18` is nearly free. To compare across splits,
   use absolute `stage{N}/svm_kappa`, or normalise by the headroom, `delta / (1 - kappa_base)`.

Nothing consumes this metric — the checkpoint monitor is `stage1/val_recon_loss` in stage 1
and `stage2/svm_kappa` in stage 2, never the delta. The risk is one of reading, not of
optimisation.

## If you want the curves to agree in stage 1

Make the probe's training order deterministic, so both call sites see the same matrix:

```python
def _train_embeddings(self):
    ...
    X, y = self._embeddings(self._train_loader_cache)
    order = np.argsort(y, kind="stable")
    return X[order], y[order]
```

Before changing anything, read `stage{N}/svm_fit_status` and `stage{N}/svm_n_iter_max`,
which the probe already logs. If the solver converges on the real embeddings, order never
mattered and the oscillation has a different source.
