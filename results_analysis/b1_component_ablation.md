# B1 Component Ablation Chain

Chain:

- PatchOnly: Backbone only
- RefineOnly: Backbone + refinement (no fusion)
- FusionOnly: Backbone + fusion (no deeper refinement)
- Full: Full model

## ETTm2 (H=192)

- Backbone only: params=7,823,808, mse=0.2724 ± 0.0000, mae=0.3131 ± 0.0000
- Backbone + refinement (no fusion): params=2,224,576, mse=0.2272 ± 0.0000, mae=0.2953 ± 0.0000
- Backbone + fusion (no deeper refinement): params=3,256,768, mse=0.2347 ± 0.0000, mae=0.2997 ± 0.0000
- Full model: params=3,851,840, mse=0.2216 ± 0.0000, mae=0.2922 ± 0.0000

## ETTh2 (H=192)

- Backbone only: params=7,823,808, mse=0.4515 ± 0.0000, mae=0.4594 ± 0.0000
- Backbone + refinement (no fusion): params=2,224,576, mse=0.3821 ± 0.0000, mae=0.4002 ± 0.0000
- Backbone + fusion (no deeper refinement): params=3,256,768, mse=0.3720 ± 0.0000, mae=0.4004 ± 0.0000
- Full model: params=3,851,840, mse=0.3733 ± 0.0000, mae=0.3976 ± 0.0000

## Weather (H=192)

- Backbone only: params=22,277,184, mse=0.1963 ± 0.0000, mae=0.2580 ± 0.0000
- Backbone + refinement (no fusion): params=2,224,576, mse=0.1941 ± 0.0000, mae=0.2397 ± 0.0000
- Backbone + fusion (no deeper refinement): params=3,256,768, mse=0.1936 ± 0.0000, mae=0.2387 ± 0.0000
- Full model: params=3,851,840, mse=0.1978 ± 0.0000, mae=0.2426 ± 0.0000
