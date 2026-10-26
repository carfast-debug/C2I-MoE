# Third-party provenance

This code package contains the encoder and fusion adaptations used in C2I-MoE.
It does not distribute pretrained weights, patient imaging, cohort manifests,
or ClinicalBERT assets.

- `encoders/model_3dvit.py` implements the structural backbone used with 3DINO
  weights. The supplied upstream 3DINO license is retained verbatim as
  `3DINO-LICENSE.txt`: CC BY-NC-ND 4.0. No alternative license is granted for
  upstream material or pretrained weights.
- `encoders/brain_gfm.py`, `graph_prompt.py`, and `disease_names.py` come from
  the authors' local BrainGFM adaptation. No upstream license was found for
  these adapted files in the supplied project. No project-wide relicensing
  has been applied.
- The interaction expert and query-rotation implementations derive from the
  local I2MoE and RollingQ adaptations. Their method attribution must be
  retained; this package does not claim ownership of their original work.

The authors have not specified a repository-wide reuse license.
Public visibility alone does not replace the original third-party terms.
