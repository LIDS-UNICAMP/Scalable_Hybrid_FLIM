# SPiFiL Grafting — fitting layers one at a time, and splicing them into an existing network

Working notes from reading the SPiFiL sources at `/dados/home/moliveira/SPiFiL`.
Companion to [`scripts/spifil_resnet_graft.py`](../scripts/spifil_resnet_graft.py).

**Nothing on this page was executed.** Every number is what the code says it will
do, not measured output. `spifil` 1.0.0 *is* installed in the `scalable_FLIM` env
(editable, from `/dados/home/moliveira/SPiFiL`) — the script imports clean; it just
has not been run on data.

---

## 1. The premise: a filter is a piece of an image, not a trained tensor

SPiFiL builds convolutional encoders with **no backpropagation anywhere in the
fit**. A superpixel algorithm partitions each training image, the center of every
superpixel becomes a candidate point, patches around those points are ranked by
how well they separate classes, and the winning patches *become* the convolution
kernels — with the z-score folded into weight and bias so no normalization layer
is needed at inference.

There is no optimizer and no epoch count. The result is a plain PyTorch
`nn.Module`, which is exactly why it can be spliced into an existing network.

> Everything below follows from one fact: the kernel values are cut from real
> feature maps. Change what produces those feature maps and you change what
> SPiFiL learns.

---

## 2. Superpixels run once. Seeds are what travel.

This is the part that is easy to get wrong, and it governs every design decision
downstream.

**The superpixel segmentation happens exactly once, at layer 0, on the
color-transformed image** — the LAB map, not the raw RGB. It is never recomputed
for layers 2, 3 or 4. What propagates upward is the **seeds**: one coordinate per
superpixel, repositioned by integer division as the grid shrinks.

```python
# preparation.py — the whole of layer 0, in three lines
features = color(dataset.load_image(index))   # image -> LAB
labels   = superpixels(features, mask, n)     # superpixels ON the LAB map
coords   = seed_extractor(labels, features)   # one seed per region, at its medoid
```

The raw image appears only in the first line. From the second line on, nothing
looks at it again. The `SuperpixelAlgorithm` protocol names its parameter
`features` and its docstring says "*not* the raw RGB"; SLIC is called with
`convert2lab=False` precisely so skimage does not convert to Lab a second time.

### The coordinate transformation

When a layer pools and the map shrinks, the seeds follow via
`Seeds.project(stride)`:

- coordinates become `coords // stride`;
- the grid becomes `ceil(size / stride)`;
- seeds landing on the same new pixel **merge into one**, keeping the best
  (lowest positive) rank of the group;
- `stride <= 1` is a no-op copy.

`projection_stride(spec)` supplies the stride: `1` when `pool_type="none"`,
otherwise `pool_stride`. Projection runs **per image**, and every seed of an
image already carries that image's class, so a merge can never mix classes.

`extract_patches` enforces the consequence: it raises unless
`seeds.grid == features.shape[1:]` exactly. There is no automatic resampling —
if the seeds are on the wrong grid, the fit stops rather than silently cutting
patches from the wrong places.

---

## 3. Layer-by-layer is the native granularity

`Learner.fit()` is nothing but `prepare()` followed by a loop over a public
per-layer method. Driving that loop yourself costs nothing and buys inspection
between layers.

```python
learn.prepare()                    # layer 0 + first scoring pass
for layer in range(1, learn.n_layers + 1):
    learn.fit_layer(layer)
    bank = learn.state[layer].bank
    print(len(bank), bank.labels.tolist())

learn.refit_from(1)                # drop layers >= 1 and redo them
learn.export("runs/my-model")
```

### Layer numbering

Layer 0 is the `ColorTransform`'s output. Conv layers are **1-based**, so
`arch.layers[L-1]` is layer `L`'s spec.

`n_layers` is `len(arch.layers)` — what you *asked for*, not what has been
fitted. For actual progress use `max(learn.state)`, or check whether
`learn.state[L].bank is None`.

### Two ranges that do not match

| Operation | Valid range | Idiom |
|---|---|---|
| Fit, export, specs | `1 … n_layers` | `range(1, n_layers + 1)` |
| Scoring | `0 … n_layers - 1` | `range(0, n_layers)` |

Scoring layer `L` uses layer `L+1`'s kernel window, because the patches being
scored are the ones `L+1` will convolve. The last layer has no successor, so
`score_layer(n_layers)` raises.

Order is also forced: `fit_layer(L)` raises if `state[L-1]` is missing.

---

## 4. Grafting into the middle of an existing network

### The tempting shortcut, and why it is wrong

`ColorTransform` is just a callable `(H,W,3) uint8 -> (C,H,W) float32` with an
`out_channels` property. Nothing in the protocol requires it to be a color
conversion, so it is tempting to hand it a frozen ResNet trunk and let SPiFiL fit
on top.

That runs. It is also **a different method**: superpixels would then be computed
on the ResNet's 64-channel feature map instead of on the LAB image. Section 2
says they belong on LAB.

### The correct seam: build `state[0]` yourself

`Learner.prepare()` skips its entire preparation step when `state[0]` already
exists, and still runs `score_layer(0)` when no seed carries a rank yet. That is
a supported hook — the `Checkpoint` callback resumes a fit exactly this way.

So layer 0 gets assembled by hand:

1. Run `preparation.prepare(..., color=LabNorm())` to get LAB features,
   superpixels and seeds on the **224×224** grid.
2. Run the ResNet trunk to get features on the **56×56** grid.
3. Project the seeds by `224 // 56 = 4`.
4. Put the ResNet features and the projected seeds into `state[0]`.

```python
prepared = list(preparation.prepare(
    data, superpixels=SLIC(), seed_extractor=Medoids(),
    n_superpixels=n_superpixels, color=LabNorm(),   # superpixels run HERE
))

stride, rest = divmod(lab_grid, trunk.grid)         # 224 // 56 = 4
if rest:
    raise ValueError("projection needs an exact divisor")

learn.state[0] = LayerState(
    features=[torch.from_numpy(trunk(data.load_image(i))).to(device)
              for i in range(len(prepared))],
    seeds=[p.seeds.project(stride) for p in prepared],
)
learn.prepare()   # sees state[0], skips preparation, only scores
```

`LayerState` requires only `features` and `seeds`; every other field has a
default. `superpixel_labels` is inspection-only — nothing in the fit reads it.

### Still pass the trunk as `color=`

Not because it will be called — with `state[0]` pre-populated it never is — but
because `Learner` reads `color.out_channels` to size the first block, and
`export()` writes `describe_color(color)` into `architecture.json`. Leaving the
default `LabNorm` there would record 3-channel LAB metadata for a model fitted on
64-channel ResNet features, and `load_encoder` would trust it.

### Resize before the LAB, not after

The images must be resized to 224 **before** the color transform, so the seed
coordinates are born on a grid that divides evenly into the ResNet's. Override
`load_image` and `load_mask` on a `SpifilDataset` subclass — masks with `order=0`
nearest-neighbor, since a mask is a label map and interpolating it invents labels
that do not exist. `from_folders` uses `cls(samples)`, so a subclass comes back
correctly as long as no required `__init__` parameter is added.

---

## 5. Recipe: ResNet-18, `layer2` replaced

The spatial arithmetic lines up exactly: `layer2` took 56×56 -> 28×28, and
SPiFiL's pooled size is `floor((56-1)/2)+1 = 28`.

```python
before_graft = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                             resnet.maxpool, resnet.layer1)      # -> (64, 56, 56)
after_graft  = nn.Sequential(resnet.layer3, resnet.layer4,
                             resnet.avgpool, nn.Flatten(), resnet.fc)
need = int(resnet.layer3[0].conv1.in_channels)                   # 128

arch = ArchSpec(layers=[
    LayerSpec(kernel_size=3, out_channels=64,   pool_type="none"),               # 56
    LayerSpec(kernel_size=3, out_channels=need, pool_type="max", pool_stride=2), # 28
])

model = nn.Sequential(before_graft, learn.model, adapter, after_graft)
```

Derive `out_channels`, the feature-grid size and the projection stride from a
probe forward through `before_graft` rather than writing 64, 56 and 4 by hand —
then moving the graft point needs no other edit.

The full grid trace for this configuration:

| Stage | Grid | What happens to the seeds |
|---|---|---|
| LAB + superpixels | 224×224 | seeds created here, ~100 per image |
| ResNet trunk | 56×56 | `project(4)` — coordinates divided by 4, collisions merge |
| SPiFiL layer 1 (`pool_type="none"`) | 56×56 | `project(1)` — no-op, seeds stay put |
| SPiFiL layer 2 (`pool_stride=2`) | 28×28 | `project(2)` — but nothing consumes it, it is the last layer |

---

## 6. How many filters you actually get

The count passes through two ceilings, in order.

### Ceiling 1 — integer division per class

`UniformAllocator` gives every class `out_channels // n_classes` filters and
drops the remainder.

| Classes | Layer 1 asks 64 | Layer 2 asks 128 | Adapter needed? |
|---|---|---|---|
| 2 — larvae | 32 × 2 = **64** | 64 × 2 = **128** | no |
| 8 — eggs | 8 × 8 = **64** | 16 × 8 = **128** | no |
| 6 — cysts | 10 × 6 = **60** | 21 × 6 = **126** | **yes** |

### Ceiling 2 — available seeds

`n_pick = min(per_class, pool_size)`. A class with fewer ranked seeds than its
budget produces fewer filters.

This ceiling matters more here than in stock SPiFiL, because the projection from
224 to 56 **merges colliding seeds**. Ask for 100 superpixels and you may reach
layer 1 with fewer — the script prints the surviving count so the loss is
visible.

Print `len(bank)` every layer too. It is the real filter number, and it is the
same sanity check the SPiFiL README recommends: `eggs` builds 16/32/48 while
`cysts` builds 12/30/48.

---

## 7. How a filter is actually produced

1. **Seeds.** Superpixels partition the **LAB** features inside the mask. Each
   region reduces to one seed at its medoid, inheriting the image's class. Once,
   at layer 0 — see section 2.
2. **Patches.** A window is cut around every seed using the *next* layer's kernel
   size — 3×3 × 64 channels = 576 numbers in the ResNet-18 recipe. All images
   merge into one global dataset. Patches are channel-major, matching
   `F.unfold`'s layout so a flattened `Conv2d` weight multiplies them directly.
3. **Scoring.** A Fisher score under a Mahalanobis metric measures how well each
   patch separates classes, then becomes a per-class rank where 1 is best.
4. **Budget.** The allocator splits `out_channels` across classes.
5. **Diversity-aware greedy pick.** Per class, take the `budget × 3` best-ranked
   as a pool, start from rank 1, then repeatedly take whoever maximizes
   `alpha * rank + (1 - alpha) * distance to what is already picked`. With
   `alpha=0.5`: half "be discriminative", half "do not duplicate a filter I
   already have".
6. **Fold into weights.**

   ```
   K[k][f] = unit_norm(zscore(patch_k))[f] / stdev[f]
   bias[k] = -sum_f mean[f] * K[k][f]
   ```

That last step is what makes `K·x + bias` identical to
`unit_norm(zscore(patch_k)) · zscore(x)`. The convolution measures similarity
between the input and the patch that became the filter, in z-scored space, for
free at inference time.

---

## 8. The 1×1 adapter

A channel-count converter that exists only when SPiFiL produces fewer filters
than the downstream stage demands. `layer3` was built expecting exactly 128 input
channels; with 6 classes SPiFiL delivers 126, and `layer3` rejects it.

```python
got = int(learn.model.block(learn.n_layers).out_channels)
adapter = nn.Identity() if got == need else nn.Conv2d(got, need, kernel_size=1)
```

A 1×1 window mixes channels without looking at neighboring pixels, so the spatial
size is untouched. Think of a plug adapter: same current, different pin count.

**Cost.** The adapter is the *only* part of the final model with random weights —
SPiFiL did not learn it and it did not come from pretraining. In practice this
costs nothing extra, because `after_graft` needs fine-tuning regardless.

With 6 classes there is no filter count that avoids it, since 128 is not
divisible by 6. The clean alternative is a custom `FilterAllocator` that
distributes the remainder instead of dropping it.

---

## 9. It does not start from random init

Reading the code it looks like the kernels begin as noise, because
`SpifilNet.__init__` creates every block immediately with `nn.Conv2d(...)` and
torch fills it with its default initialization. The shape is right and the values
are meaningless.

Then `fit_layer` calls `set_filters`, which does `copy_` over the top. Those
random values live between the line that constructs the model and the line that
fits the layer. They are never a starting point for anything — there is no
`.backward()` in the fit at all.

Layer 2 works the same way one level up: it cuts patches from layer 1's *output*,
which is already "ResNet features convolved with layer 1's filters". Real data
all the way down.

---

## 10. Why I-JEPA is the wrong backbone for this

A ViT has exactly one convolution — the patch embedding,
`nn.Conv2d(3, 1280, 14, stride=14)`. The other 32 layers are attention and MLP,
with no spatial kernels for SPiFiL to learn. Even the patch embed is out of
reach: SPiFiL spans are always odd (`2*(k//2)+1`, so 14 -> 15) and its blocks are
shape-preserving stride-1 convolutions, not stride-14.

You *can* feed it through the same `state[0]` seam, but three constraints make it
a poor trade against ResNet:

- **The grid is locked at 16×16.** `position_embeddings` is a fixed
  `(1, 256, 1280)` parameter, so input must be exactly 224×224 and 224/14 = 16.
  Projecting 224 seeds onto a 16×16 grid means a stride of 14 — most seeds would
  merge away.
- **Pooling becomes impossible.** Default stride-2 pooling takes you
  16 -> 8 -> 4; two layers is the ceiling.
- **1280 channels is fatal without reduction.** Layer-1 patches would be
  1280 × 3 × 3 = 11520-dimensional, and Mahalanobis accumulates a D×D float64
  covariance — a 1 GB matrix and an O(D³) Cholesky. PCA down to 64 channels is
  mandatory.

ResNet at `layer1` gives 56×56 and 64 channels: a stride-4 projection that keeps
most seeds, and 576-dimensional patches with no PCA needed.

---

## 11. Gotchas

**Freeze the SPiFiL layers after fitting.** This one is easy to miss and it
silently undoes the whole method. `set_filters` copies weights under
`torch.no_grad()` but never touches `requires_grad`, so the fitted blocks come
out of `fit_layer` still trainable — torch's default. The moment someone builds
`Adam(model.parameters())` to fine-tune `after_graft`, gradients start rewriting
filters that were supposed to be cut patches, with no error and no shape
mismatch. The official notebook always calls `SpifilEncoder.freeze()`;
`SpifilNet` has no such method, so do it by hand:

```python
for param in learn.model.parameters():
    param.requires_grad_(False)
```

**The exported bundle is only the middle.** `export()` writes the `Trunk`
metadata and the two SPiFiL blocks — not the 1×1 adapter, not `after_graft`.
Reloading it does not reconstruct the grafted model. When `got != need` the
adapter is load-bearing and lives nowhere in the bundle, so
`nn.Sequential(load_encoder(...), after_graft)` fails on channel count. Save the
assembled `nn.Sequential` separately if you want the whole thing back.

**Watch `N` against `D` when grafting deeper.** A layer-1 patch has
`in_channels × kernel²` dimensions — 576 at `layer1` (64 channels). Mahalanobis
estimates a `D×D` covariance and wants `N > D`, where `N` is the total ranked
seed count. Eight images at ~100 seeds gives 800 against 576: it works, but the
margin is 1.4×, and Ledoit-Wolf shrinkage quietly drifts toward a scaled
identity near that threshold — no warning fires. Cutting after `layer2` (128
channels, `D = 1152`) would need more images.

**Superpixels belong on the LAB image, not on the trunk's features.** Passing the
trunk as `ColorTransform` runs, but silently moves the segmentation onto the
64-channel ResNet map. Build `state[0]` by hand instead — section 4.

**Resize before the color transform.** If the image reaches LAB at its original
size, seed coordinates land on a grid that may not divide evenly into the trunk's,
and the projection stops being an exact integer division.

**Freeze the trunk and keep it in `.eval()`.** Every filter is a patch cut from
those exact activations. BatchNorm drifting in train mode, or any fine-tuning of
the trunk after the fit, silently invalidates the whole bank.

**Seed count shrinks at the projection.** Collisions merge. Ask for 100
superpixels and layer 1 may see fewer; print the surviving count.

**Record the right color transform in the bundle.** `export()` writes
`describe_color(self.color)`. If `state[0]` was injected but `color` was left at
the default, the bundle claims LAB features for a ResNet-fitted model, and
`resolve_color` will trust it.

**The downstream half will not work off the shelf.** SPiFiL filters emit z-scored
*similarity* responses, not ResNet activations. `layer3` was trained against
`layer2`'s distribution. Expect poor accuracy until you fine-tune `after_graft`,
or at least the head.

**Residual connections are gone in the grafted span.** `SpifilConvBlock` is
conv -> ReLU -> pool, no skip. This is an architectural change, not just new
weights.

**`load_encoder` needs the transform handed back.** `resolve_color` refuses to
rebuild a `ColorTransform` that takes constructor arguments, and a trunk wrapper
does. Always pass `color=` on reload.

**A partial fit exports silently wrong weights.** `export()` writes all
`n_layers` into `architecture.json`, but unfitted blocks carry torch's default
init. Correct shapes, meaningless values, no warning.

**Even kernel sizes widen by one.** Span is `2*(kernel_size//2)+1`, so
`kernel_size=4` builds a 5×5 kernel.

**Fix the seed.** `spifil-fit` calls `torch.manual_seed(cfg.seed)` with a
default of 42 and that is its only determinism knob — no numpy seed, no cuDNN
flags. A hand-written driver that skips it is less reproducible than the CLI for
no reason.

### Inherited from the library, not introduced by the graft

`SLIC(compactness=10.0)` runs on `LabNorm` output, whose three bands are squeezed
into roughly `[0, 1]`, while the spatial term still spans 224 pixels. The
distance trade-off therefore leans heavily spatial, and superpixels tend toward a
regular grid rather than following appearance. That is SPiFiL's own default pair,
not something the graft changed — and the published results use DISF, not SLIC,
so SLIC is already off-paper. Worth knowing before blaming the graft for bland
seed placement.

---

## 12. Source map

| What | File |
|---|---|
| Orchestrator, `prepare`, `fit_layer`, `refit_from`, `export` | `spifil/learner.py` |
| `LayerState`, the injection point | `spifil/learner.py` |
| Layer 0: LAB, superpixels, seeds | `spifil/preparation.py` |
| `Seeds.project` — the coordinate transformation | `spifil/types.py` |
| `coords // stride`, `ceil(size/stride)` | `spifil/coords.py` |
| Patch extraction, grid validation | `spifil/patches.py` |
| `ColorTransform` protocol, `LabNorm` | `spifil/color.py` |
| Per-class budget, integer division | `spifil/allocation.py` |
| Diversity-aware greedy pick | `spifil/selection.py` |
| Patches -> conv weights, the z-score fold | `spifil/nn/builder.py` |
| The conv block, `set_filters`, pooling math | `spifil/nn/blocks.py` |
| Bundle loading, `resolve_color` | `spifil/nn/encoder.py` |
| This graft, end to end | [`scripts/spifil_resnet_graft.py`](../scripts/spifil_resnet_graft.py) |

The DISF superpixel backend needed to reproduce the paper is a separate prebuilt
wheel under `vendor/`; SLIC is the dependency-free default and partitions
differently, so it does not reproduce the published numbers.
