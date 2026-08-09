# Reading the SVM probe curves

W&B shows two kappa series during autoencoder pretraining:

- `baseline/svm_kappa` — a flat line
- `val/svm_kappa` — a line that moves

This note explains where each one comes from and why they differ.
All code references are to [`src/modules/autoencoder_flim_module.py`](../src/modules/autoencoder_flim_module.py).

## What the probe is

The autoencoder trains without labels, on reconstruction loss alone. To know whether the
embedding is getting *better* — not just reconstructing better — a linear SVM is fitted on
the labelled train split and scored on validation. That is the probe. It never enters the
loss; it only reports.

Both curves come from the same function, `_svm_probe()` (line 334), which takes four
arguments: `X_tr, y_tr` (fit the SVM) and `X_val, y_val` (score it, producing kappa).

## The two curves are the same computation, run at different times

| | `baseline/svm_kappa` | `val/svm_kappa` |
|---|---|---|
| computed in | `on_fit_start()` — **once**, before any gradient step | `on_validation_epoch_end()` — **every epoch** |
| `X_tr` from | line 314, `_train_embeddings()` | line 394, `_train_embeddings()` |
| `X_val` from | line 315, `val_dataloader()` | line 390, `_val_emb` accumulated in `validation_step` |
| stored as | `self.baseline_metrics`, replayed each epoch at line 413 | logged directly at line 405 |

`baseline/svm_kappa` is flat because it is not recomputed. It is one number, captured
before training, reprinted every epoch so the panel shows it next to the live curve.

## Why the live curve moves

The validation side is fixed. `val_dataloader` has `shuffle=False` and, with `V_eval=1`,
the dataset uses the deterministic test transform — so `X_val` is identical every epoch.

The train side is not. `train_dataloader` has `shuffle=True`, so every call to
`_train_embeddings()` returns the same images and labels **in a different row order**.

This matters because of how the probe is configured (line 346): `C=1e2`, no scaler,
`max_iter=10000`. When libsvm's SMO does not converge within that iteration cap, it stops
at an arbitrary point along the optimisation path, and that point depends on the order of
the training rows. Same matrix, permuted rows, different kappa.

A synthetic check with those exact hyperparameters:

- solver hits `max_iter` (`fit_status_ = 1`): permuting the training order alone moved
  kappa across a range of 0.31
- solver converges (`fit_status_ = 0`): permuting changed nothing — identical predictions

So the movement is the probe, not the encoder.

## Stage 1 vs stage 2

**Stage 1 (`--freeze-encoder`).** The encoder is frozen: `requires_grad = False` on every
encoder parameter, and the optimizer filters on `requires_grad` (line 478). The encoder is
plain `Conv2d + ReLU + MaxPool` with no BatchNorm, so there is no running statistic
drifting behind the freeze either. `embed()` is the pooled encoder output and nothing else.

The embedding is therefore bit-for-bit constant across epochs, and the two curves *should*
be a single line. Any oscillation you see in stage 1 is pure probe noise — by construction,
there is nothing the encoder could have learned.

This makes stage 1 the right place to measure how big that noise is.

**Stage 2 (encoder unfrozen).** Now `val/svm_kappa` moves for two reasons at once: probe
noise *and* real encoder learning. The two are summed into one curve and cannot be
separated by eye. Read it alongside `val/flim_drift` — a rising kappa with an exploding
drift does not mean the FLIM encoder adapted, it means it was overwritten.

## `val/svm_kappa_delta`

Logged at line 415 as `val/svm_kappa − baseline/svm_kappa`. It is redundant: both terms are
already on the panel. Two cautions:

1. **It inherits the baseline's noise.** The baseline is a single draw at one shuffle order,
   promoted to a fixed reference. A different draw would move every delta in the run.
2. **It is not comparable across runs.** The headroom depends entirely on where the run
   started: with `kappa_base = 0.82` the delta cannot exceed `+0.18`, and reaching it means
   zero errors; with `kappa_base = 0.01` a `+0.18` is nearly free. To compare across splits,
   use absolute `val/svm_kappa`, or normalise by the headroom, `delta / (1 - kappa_base)`.

Nothing consumes this metric — the checkpoint monitor is `val/recon_loss` in stage 1 and
`val/svm_kappa` in stage 2, never the delta. The risk is one of reading, not of optimisation.

## If you want the curves to agree in stage 1

Make the probe's training order deterministic, so both call sites see the same matrix:

```python
def _train_embeddings(self):
    ...
    X, y = self._embeddings(self._train_loader_cache)
    order = np.argsort(y, kind="stable")
    return X[order], y[order]
```

Before changing anything, confirm the cause by logging `clf.fit_status_` after the `fit`
at line 356. If the solver converges on the real embeddings, order never mattered and the
oscillation has a different source.
