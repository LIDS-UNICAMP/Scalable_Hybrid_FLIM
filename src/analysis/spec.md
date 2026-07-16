# SDD: Generate t-SNE Visualizations for Available FLIM Models on Test Splits

## 1. Overview

### Feature Name
FLIM Test Embedding t-SNE Analysis

### Objective
Implement a Python script that, for each available FLIM model and for each specific problem dataset, loads embeddings from the **test** split defined in the available JSON descriptors, applies **t-SNE**, and generates **Matplotlib** scatter plots with **contrastive class colors**.

There is **no training** in this workflow. The script must only:
1. Discover available FLIM models.
2. Read dataset split descriptors.
3. Load the **test** samples.
4. Extract output embeddings from the model.
5. Apply t-SNE to those embeddings.
6. Save the resulting plots to disk in an organized output structure.

---

## 2. Context

The datasets follow this structure pattern:

- `splits_incremental/eggs/split1`
- `splits_incremental/larvae/split1`
- `splits_incremental/protozoan/split1`

Each split directory contains JSON files such as:

- `data_descriptor_perc1.json`
- `data_descriptor_perc5.json`
- `data_descriptor_perc25.json`
- `data_descriptor_perc50.json`
- `data_descriptor_perc75.json`
- `data_descriptor_perc100.json`

The relevant field inside each JSON is:

- `test`

That field must be used as the source of samples for embedding extraction and plotting.

---

## 3. Required Behavior

### 3.1 Core Functional Behavior
The implementation must:

- Iterate over **every available FLIM model**.
- Iterate over **every problem folder**:
  - `eggs`
  - `larvae`
  - `protozoan`
- Iterate over every available split JSON matching the repository pattern.
- Read only the **test** subset from each JSON descriptor.
- Generate embeddings using the corresponding FLIM model.
- Apply **t-SNE** to the extracted embeddings.
- Create one saved Matplotlib figure per:
  - model
  - problem
  - split JSON / percentage descriptor

### 3.2 Plot Behavior
Each t-SNE figure must:

- Be created with a size that is readable and not visually cramped.
- Use **contrastive colors** between classes.
- Avoid `plt.show()`.
- Avoid `plt.imshow()`.
- Avoid `plt.legend()`.
- Instead of legend, place the **class name above or near the corresponding cloud of points**.
- Save directly to disk without blocking execution.

### 3.3 Progress Tracking
The script must use **tqdm** so execution progress is visible across the main loops.

### 3.4 Platform Compatibility
The script must support the user’s current environment on **macOS**.

If the repository currently contains Ubuntu-oriented inference logic, add the minimum compatibility handling needed to ensure the script can run on Mac. This includes device selection logic such as:

- Apple Silicon / MPS when available
- CPU fallback when MPS is unavailable
- No CUDA-only assumptions

---

## 4. Non-Goals

This implementation must **not**:

- Perform training
- Modify training pipelines
- Introduce new dataset annotation formats
- Depend on interactive plotting
- Require GUI rendering
- Rebuild model-loading logic from scratch if the repository already contains reusable utilities

---

## 5. Reuse-First Policy

Before implementing anything new, inspect whether the repository already has reusable components for:

- FLIM model discovery
- model checkpoint loading
- dataset reading
- transform pipelines
- embedding extraction
- label parsing
- path management
- experiment naming conventions

### Mandatory rule
Do **not** rewrite an entire loading/inference flow if a compatible internal implementation already exists.

Prefer:
- reusing existing loaders
- reusing existing dataset abstractions
- reusing current experiment/config naming patterns
- extending an existing analysis or evaluation module if that location already matches this responsibility

---

## 6. Suggested Placement in Repository

Place the implementation in the most suitable analysis/evaluation area of the repository.

### Preferred placement principle
This script should live in a location associated with:
- evaluation
- embedding inspection
- model analysis
- visualization utilities

### Recommended structure
If the repository already has an evaluation or analysis module, place it there.

If not, use a structure similar to:

- `src/.../analysis/`
- `src/.../evaluation/`
- `scripts/.../analysis/`

### Constraint
Choose a location that is consistent with the current project structure and naming style already used in the repository.

---

## 7. Output Directory Structure

Create and save outputs under:

- `tsne_analisys/`

Mirror the dataset organization pattern in the output directory.

### Expected organization principle
The output tree must preserve the same logical sequence as the datasets:
- problem
- split
- descriptor / percentage
- model-specific result

### Example structural intent
Outputs should be grouped in a way that makes it easy to navigate by:
1. problem dataset
2. split
3. percentage descriptor
4. FLIM model

Do not flatten everything into a single folder.

---

## 8. Data Contract

### Input
JSON descriptor files in the split directories.

### Required field
- `test`

### Expected usage
The implementation must read only the test samples from that field and use them as inference input for embedding extraction.

### Validation
The script must validate:
- JSON exists
- `test` field exists
- `test` field is not empty
- referenced samples are loadable
- model is available before attempting inference

If any of these fail, log the issue and continue safely to the next valid unit of work whenever possible.

---

## 9. Model Discovery Requirements

The script must automatically work for **each available FLIM model**.

### Discovery strategy
Use the repository’s existing convention for available FLIM models if one already exists.

Examples of acceptable reuse:
- known experiment folders
- checkpoint directories
- config registries
- existing model index dictionaries
- current evaluation scripts that already enumerate FLIM runs

### Constraint
Do not hardcode a brittle list if the repository already has a structured source of truth.

---

## 10. Embedding Extraction Requirements

The script must extract the **output embedding** for each sample.

### Rules
- Use the correct inference mode (`eval`)
- Disable gradients during inference
- Batch processing is preferred if it already exists in the codebase
- Preserve sample-to-label association for plotting
- Ensure embeddings are converted into a format t-SNE can consume safely

### Important
The embedding source must correspond to the FLIM output representation intended for downstream analysis, not logits from a separate training head unless that is the repository’s established embedding interface.

---

## 11. t-SNE Requirements

### Functional Requirements
- Apply t-SNE to the collected test embeddings
- Use a deterministic/random-state-aware setup for reproducibility
- Only compute t-SNE after all embeddings for the current analysis unit are available

### Quality Expectations
- The projection should be readable
- Point overlap should be handled as reasonably as possible
- Plot size and point styling should favor interpretation

---

## 12. Plot Annotation Rules

### Color
Use visually distinct colors for each class.

### Labels
Do not use Matplotlib legend.

Instead:
- identify the spatial cloud corresponding to each class
- place the class name near or above that cloud
- ensure text placement remains readable

### Readability
The final plot should be understandable without additional manual interaction.

---

## 13. Runtime and Device Handling

### Device Selection
The implementation must include environment-aware device selection:
- use MPS on supported Mac environments when available
- otherwise use CPU
- do not assume CUDA

### Stability
If the existing code has Ubuntu-specific branches, adapt only the minimum necessary for compatibility.

### Logging
The script must log:
- selected device
- model being processed
- dataset/problem being processed
- split/descriptor being processed
- output file path
- skipped items and reasons

---

## 14. Progress Bar Requirements

Use `tqdm` to track progress across the outer workflow.

Recommended visibility:
- model loop
- dataset/problem loop
- descriptor loop

Avoid overly noisy nested progress output if it harms readability.

---

## 15. Error Handling

The script must fail gracefully.

### Handle at least:
- missing JSON
- malformed JSON
- missing `test` field
- empty test set
- missing checkpoint/model artifact
- unsupported sample path
- inference failure for one model/problem/split
- t-SNE failure due to insufficient data

### Policy
Prefer **skip + log** over crashing the entire run when one unit fails.

---

## 16. Files to Create or Change

## New File(s)
Create a dedicated script for this analysis in the most appropriate analysis/evaluation location of the repository.

Potential supporting additions are allowed only if needed:
- a small helper module for visualization
- a utility for model discovery
- a utility for device selection

## Existing File(s)
Only modify existing files if needed for:
- reuse hooks
- shared model loading
- path/config exposure
- lightweight compatibility fixes

### Restriction
Do not introduce broad refactors unrelated to this feature.

---

## 17. Implementation Notes

### Must Follow Existing Project Style
The implementation must:
- follow the current repository’s code style
- follow the current repository’s architectural conventions
- use type hints if the project already uses them
- preserve existing naming patterns
- keep responsibilities separated

### Avoid Irrelevant File Access
Do not read or inspect irrelevant files such as:
- `__pycache__`
- compiled artifacts
- temporary cache files
- binary intermediates unrelated to model discovery or dataset descriptors

If this rule is relevant to implementation assumptions, document it clearly in code comments or internal docs.

---

## 18. Acceptance Criteria

The task is accepted only if all items below are satisfied:

1. A Python script exists in the most appropriate repository location for analysis/evaluation.
2. The script processes **all available FLIM models** discoverable in the repository.
3. The script processes the problem folders:
   - eggs
   - larvae
   - protozoan
4. The script reads the JSON descriptors under the split directories.
5. The script uses only the `test` field for sample selection.
6. The script performs inference only, with **no training**.
7. The script extracts embeddings and applies t-SNE.
8. The script generates Matplotlib plots and saves them to disk.
9. No `plt.show()` is used.
10. No `plt.imshow()` is used.
11. No `plt.legend()` is used.
12. Class names are written near the point clouds instead of using a legend.
13. The output directory `tsne_analisys` is created.
14. The saved output structure mirrors the dataset organization.
15. The script uses `tqdm`.
16. The script runs with Mac-aware device handling.
17. The implementation reuses existing repository code where appropriate instead of rewriting major components.
18. Failures in one model/problem/split do not unnecessarily abort the entire execution.

---

## 19. Evaluation / Scoring Criteria

### A. Functional Completeness — 40 points
- 40: Covers all models, all problem folders, all relevant JSONs, and saves all expected plots
- 25: Covers most cases but misses one important dimension
- 10: Partial pipeline only
- 0: Does not satisfy the requested workflow

### B. Repository Integration — 20 points
- 20: Fits existing repository organization and reuses current utilities cleanly
- 10: Works but is poorly placed or duplicates logic
- 0: Ignores project structure

### C. Visualization Quality — 15 points
- 15: Readable figures, good sizing, good label placement, strong class color separation
- 8: Usable but cluttered
- 0: Hard to interpret

### D. Robustness — 15 points
- 15: Graceful handling of missing data/models and stable continuation
- 8: Some partial handling
- 0: Crashes easily

### E. Platform Compatibility — 10 points
- 10: Mac-compatible, avoids CUDA-only assumptions, uses appropriate device fallback
- 5: Partial compatibility
- 0: Ubuntu/CUDA-only behavior

### Passing Threshold
- **Minimum acceptable score: 85/100**

---

## 20. Final Deliverable

The deliverable is a repository-integrated Python analysis script that:

- discovers available FLIM models,
- loads test samples from the JSON descriptors,
- extracts embeddings,
- applies t-SNE,
- saves one non-interactive Matplotlib plot per analysis unit,
- organizes outputs under `tsne_analisys` following the dataset structure,
- and runs safely on macOS with progress tracking.

---