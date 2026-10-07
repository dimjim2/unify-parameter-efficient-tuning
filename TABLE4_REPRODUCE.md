# Table 4: four attention variants on English-to-Romanian translation

This is a subset reproduction: Table 4 has eight rows, including **five**
small-budget rows. This launcher runs prefix (35.2 BLEU), ungated prefix (34.9),
attention PA (33.7), and attention MH PA (35.3), all at length/rank 30.
PA-FFN and the three rank/length-200 rows are omitted.

## Methodology audit

- Appendix A/Table 8: mBART-cc25, WMT16 en-ro, 50,000 updates, learning rate
  5e-5, polynomial decay, no warmup, label smoothing 0.1, weight decay 0.01,
  gradient norm 1.0. Each GPU runs an independent method: 4,096-token dynamic
  batches and four accumulation steps target 16,384 tokens per update.
- Training sequences are truncated to 150 tokens. All 1,999 validation examples
  are used, matching the paper rather than the original script's 1,600 cap.
- Prefix uses joint-softmax concatenation (Eq. 7); ungated prefix uses separate
  prefix softmax plus addition (Section 4.4). Both use the released MLP prefix
  parameterization. MH PA uses the released ReLU prefix path; attention PA uses
  the released parallel adapter path.
- Five beams, minimum length 1, maximum generation length 200. The translation
  entry point now uses the explicit generation length rather than silently
  overriding it with the label-truncation length.
- Final BLEU uses the repository's Romanian normalization, diacritic removal,
  Moses tokenization and SacreBLEU with tokenization disabled. The Trainer's
  raw SacreBLEU is not the same evaluation pipeline.

## Unresolved replication limitations

- Table 4 labels rank 30 as 0.1%, but Appendix B gives
  `2 * 30 * 1024 * 3 * 12 = 2,211,840` attention parameters: approximately
  0.36% of mBART-large (about 610M). Rank 200 similarly gives approximately
  2.42%, not the table's 3.6%. Preserve the stated ranks; report this discrepancy.
  Prefix MLP training parameters are larger still than the stored prefix tensors.
- Checkpoint selection follows the released Trainer's evaluation loss. With
  label smoothing this is not literally unsmoothed perplexity, which is the
  metric named in Appendix A. The exact table-specific choice is unconfirmed.
- The released evaluation decodes truncated/tokenized reference labels, not
  untouched test references. Verify its impact before claiming exact BLEU parity.
- The released script comments suggest zero smoothing/weight decay for prefix,
  whereas Table 8 gives 0.1/0.01. We follow Table 8 for all four methods.
- Dataset availability, split sizes, B200/PyTorch compatibility, gradients,
  checkpoint reloading and end-to-end BLEU have not been validated locally.
  The existing setup script does not provision mBART, WMT16, Moses or SacreBLEU.
- One seed (42) is an initial reproduction, not an uncertainty estimate. No
  configuration review can guarantee identical published scores.

## Running

Use the repository's modified Transformers in a B200-compatible environment.
Install SacreBLEU and provision `mosesdecoder/`, including its WMT16 scripts.
Confirm model/data downloads and expected split sizes (610,320/1,999/1,999).

```bash
# Short end-to-end probe, not a paper result. Still preprocesses the full data.
GPU_IDS=0,1,2,3 MAX_STEPS=10 RUN_TAG=table4_probe bash run_table4_small.sh

# Only after all four probes train, reload a checkpoint, generate and score:
GPU_IDS=0,1,2,3 RUN_TAG=table4_seed42 bash run_table4_small.sh
```

An existing RUN_TAG is refused. Each method stores its runner, Git revision,
tracked diff, environment, training log and outputs under
`checkpoints/wmt16_table4_small/<RUN_TAG>/<METHOD>/`.
Inspect failures in `logs/table4_small/<RUN_TAG>/` before scheduling full runs.
