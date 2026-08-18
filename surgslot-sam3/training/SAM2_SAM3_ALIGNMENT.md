# SAM2 / SAM3 training alignment

This file records the deliberate differences between the public SurgSLOT-SAM2
and SurgSLOT-SAM3 training implementations. A difference should remain only
when it is required by the model interface or by the paper's SAM3 recipe.

## 1. Dataset and sampling pipeline

### Aligned

The following SAM3 files now use the same implementation as SAM2:

- `dataset/vos_dataset.py`
- `dataset/vos_raw_dataset.py`
- `dataset/vos_sampler.py`
- `dataset/vos_long_mem_sampler.py`
- `dataset/vos_segment_loader.py`
- `utils/data_utils.py`

This alignment includes:

- annotation-driven image/mask frame matching;
- validated object-ID filtering for palette masks;
- shared category-ID lookup and `-1` fallback for datasets without metadata;
- short-video sampling with replacement;
- explicit normal/long-memory batch metadata;
- category/object alignment checks during collation;
- validation that a batch does not mix normal and long-memory clips.

### Dataset configuration alignment

The SAM3 dataset YAML now follows SAM2 for dataset composition and sampling:

- 6 normal video samplers;
- 6 long-memory video samplers;
- 4 single-frame image samplers;
- the same batch size for all three branches;
- mixed-dataset probabilities of `0.4 / 0.4 / 0.2`;
- `step_length: 1` for all video datasets.

The former SAM3-only GraSP train extension and ART-Net entries were removed.
The old `7x` batch-size multiplier for the image branch was also removed.

`batch_sizes` does not add three batches together. `MixedDataLoader` selects
one of the three dataloaders on each step according to `dataset_prob`, and the
corresponding entry controls the selected dataloader's batch size.

Two model-input settings remain deliberately different in the parent configs:

- resolution: SAM2 uses 512 and SAM3 uses 672;
- normalization: SAM2 uses ImageNet mean/std, while SAM3 uses mean/std 0.5.

These are not dataset-composition differences. SAM3's own training/evaluation
configs consistently use mean/std 0.5, so changing them to the SAM2 values
would change the input distribution expected by the SAM3 backbone/checkpoint.
`num_train_workers` also differs, but it only affects data-loading throughput,
not the samples or sampling probabilities.

### SAM3-only transform retained for now

`dataset/transforms.py` contains `RandomResizedCropAPI`, which is absent from
SAM2 and is not referenced by the current public SAM3 YAML. It is temporarily
retained because it came from the original SAM3 training project. It can be
removed later if the final reproduction configuration does not use it.

### Pending model-level decision

The aligned sampler/collator now carries `num_long_mem_frames`, matching SAM2.
SAM2 validates this value and deterministically initializes the configured
long-memory frames. The current SAM3 model wrapper instead uses its own
`num_init_long_mem_frames` setting and contains a SAM3-specific probabilistic
branch. This behavior is not changed in the dataset-alignment step; it must be
reviewed together with `model/sam3_train.py` in the next comparison stage.

### SAM3 ViT MLP training path

SAM3's fused `addmm_act` kernel is inference-only and detaches the linear
weights. The original SurgicalSAM3 project therefore uses it only when
autograd is disabled; gradient-enabled training uses `fc1` followed by the
configured activation. The public SAM3 copy initially lost this fallback and
raised `ValueError: Expected grad to be disabled` while fine-tuning ViT blocks
24--31. The fallback has been restored in `sam3/model/vitdet.py`. The intended
partial-freeze policy remains unchanged: blocks 0--23 are frozen and blocks
24--31 remain trainable.

## Remaining comparison stages

1. `model/sam3_train.py` versus SAM2 model wrappers and prompt/memory behavior.
2. `loss_fns_slot.py` and `loss_fns.py`.
3. `trainer.py`, `optimizer.py`, and checkpoint handling.
4. Launch/config naming, logging, and public training commands.

## Final-recipe removals

### Entropy-weighted focal loss

`use_entropy_weighted_loss`, its boundary/alpha parameters, and the associated
pixel-weighting implementation have been removed from both SurgSLOT-SAM2 and
SurgSLOT-SAM3. This was an experimental loss that is not part of the final
training recipe. The mask term now always uses the standard sigmoid focal loss
path in both versions.
