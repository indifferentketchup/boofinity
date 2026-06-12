# dependency-stack

## ADDED Requirements

### Requirement: huggingface_hub 1.x compatibility

The package SHALL resolve Hugging Face auth tokens without the `HfFolder` API removed
in huggingface_hub 1.0, matching upstream fix 2ecb218.

#### Scenario: Token resolution under huggingface_hub >= 1.0

- GIVEN a venv with huggingface_hub >= 1.0 installed
- WHEN `infinity_emb.transformer.utils_optimum` is imported and `_list_all_repo_files` resolves a token
- THEN no ImportError or AttributeError is raised and the token comes from `huggingface_hub.get_token()`

### Requirement: No BetterTransformer dependency

The torch code path SHALL NOT import or invoke `optimum.bettertransformer`. The
`bettertransformer` engine argument SHALL remain accepted for backward compatibility,
default to false, and be ignored with a single deprecation warning when set true.

#### Scenario: Import with optimum 2.x installed

- GIVEN a venv with optimum >= 2.0 installed alongside the package
- WHEN `import infinity_emb` runs and a torch-path model class is loaded
- THEN no ImportError is raised from `optimum.bettertransformer`

#### Scenario: Legacy flag passed

- GIVEN a deployed CLI config passing `--bettertransformer true`
- WHEN the engine starts
- THEN startup succeeds, one deprecation warning is logged, and inference behaves as with the flag false

#### Scenario: TF32 enablement survives

- GIVEN a CUDA-capable torch build
- WHEN any torch model module is imported
- THEN `torch.backends.cuda.matmul.allow_tf32` is true (side effect preserved from acceleration.py)

### Requirement: Relaxed uvicorn and transformers pins

The package metadata SHALL allow uvicorn >= 0.32 without a minor-version cap and
transformers >= 4.47,< 6.

#### Scenario: Fresh dependency resolution

- GIVEN a fresh venv built from the updated pyproject with the CPU torch index
- WHEN dependencies resolve
- THEN uvicorn resolves to the current release (>= 0.49) and transformers may resolve to 5.x

### Requirement: Embedding parity preserved

Any dependency or code change in this change SHALL preserve bge-m3 embedding parity
against the frozen batch-1 baseline.

#### Scenario: Parity gate

- GIVEN the batch-1 baseline `tests/parity/fixtures/baseline_bge-m3_cpu.npz`
- WHEN `tests/parity/check_parity.py` runs from the new venv
- THEN every input reaches cosine >= 0.9999 and the script exits 0
