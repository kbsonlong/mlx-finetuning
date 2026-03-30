# MLX-LoRA Autoresearch Program

## Scope

Use the frozen dataset in `data/mlx_lm/`. The validation and test splits are immutable once generated.

## Editable surface

Only change the configuration block at the top of `train.py` unless the user explicitly expands scope.

## Experiment rules

1. Read `outputs/results.tsv` before proposing a new run.
2. Prefer small, explainable parameter changes.
3. Do not overwrite frozen validation or test data.
4. Record every run, including failures.
5. Promote a configuration only when `val_loss` improves under the same dataset metadata.

## Constraints

- Main metric: `val_loss`
- Resource cap: `max_memory_gb` from `config/base_config.yaml`
- Keep the run output parse-friendly, one metric per line

