<div align="center">

# Latent Interface Training

### Breaking the Vision–Action Shortcut for Generalizable Robot Foundation Models

[Jianman Lin](https://github.com/jianmanlincjx)<sup>1\*</sup> · Shailesh Shailesh<sup>2\*</sup> · Zhongyi Luo<sup>3</sup> · Jiafei Duan<sup>2†</sup>

<sup>1</sup>South China University of Technology · <sup>2</sup>National University of Singapore · <sup>3</sup>Nanyang Technological University

[![Project page](https://img.shields.io/badge/Project-Page-1F4E9C)](https://jianmanlincjx.github.io/LIT/)
[![Paper](https://img.shields.io/badge/Paper-PDF-B23A32)](https://jianmanlincjx.github.io/LIT/static/paper/LIT.pdf)
[![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoints-linjianman%2FLIT-f7c843)](https://huggingface.co/linjianman/LIT)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Real--robot%20data-yam__bimanual__manipulation-f7c843)](https://huggingface.co/datasets/chinchinati/yam_bimanual_manipulation)
[![License](https://img.shields.io/badge/License-Apache%202.0-2ea44f)](./LICENSE)

</div>

---

<div align="center">

<img src="docs/static/images/framework.png" width="92%" alt="Latent Interface Training: Stage 1 learns an action prior without images; Stage 2 routes vision through a pose-supervised latent interface.">

</div>

LIT keeps a robot foundation model's backbone and action expert as they are and changes only the interface
between them. Direct conditioning on visual tokens is removed; a small set of learnable latent tokens —
supervised to reconstruct the terminal SE(3) end-effector pose — becomes the action expert's **only** route
to vision. **Stage 1** learns an action prior from language, state and that pose with no images.
**Stage 2** trains the interface. The same recipe, with the same interface settings, is applied to four
architectures.

<div align="center">

<img src="docs/static/images/results_strip.png" width="88%" alt="Base vs LIT: LIBERO-Plus Overall on π0.5 68.97→79.67, MolmoAct2 63.62→71.92, FAST-WAM 51.44→60.63, ImageWAM 83.02→86.89; real robot (MolmoAct2) lighting 53.3→70.0, camera 30.0→46.7, distractors 50.0→63.3.">

<sub>LIBERO-Plus: zero-shot on all 10,030 tasks, one episode each; Overall is the mean over the seven perturbation axes.
Real robot: three tasks on the MolmoAct2 backbone, 10 rollouts per task per OOD condition. In-distribution LIBERO success is preserved or improved on every architecture.
Per-axis numbers are on the <a href="https://jianmanlincjx.github.io/LIT/#plus">project page</a>; <code>results/plot_results_strip.py</code> redraws this figure.</sub>

</div>

**Contents** &nbsp; [Integrating LIT into a VLA or WAM](#integrating-lit-into-a-vla-or-wam) · [Worked example: MolmoAct2](#worked-example-molmoact2-train-then-evaluate) · [Two things that change the numbers](#two-things-that-change-the-numbers) · [Checkpoints](#checkpoints) · [Code](#code) · [Citation](#citation)

---

## Integrating LIT into a VLA or WAM

LIT is a **plug-in interface**, not a new model. It leaves the backbone, the action expert, the action
representation, the chunk length and the generation objective of the host exactly as they are, and changes only
*how vision reaches the action expert*. The recipe below is what we applied, unchanged, to two VLAs (π0.5,
MolmoAct2) and two WAMs (FAST-WAM, ImageWAM); each step names the reference implementation in the forks.

### What the host must have

| Requirement | Why |
| --- | --- |
| A backbone that produces token features for the image (and, ideally, language/state) | the latents read them |
| An action expert conditioned on those features at an identifiable **coupling point** — cross-attention KV, a joint attention mask, or feature injection into a DiT | that is the one place LIT edits |
| Action chunks of length `H` and an end-effector pose in the dataset state (position + rotation + gripper) | the pose target is the chunk's terminal pose |

### Step 1 — Find the coupling point and close it (the firewall)

Locate where visual features enter the action expert and shut that path, so that vision has **no direct route**.
Language and robot state may keep their original path (MolmoAct2) or be routed through the interface too
(π0.5) — the invariant is *no image tokens into the action expert*.

| Host | Coupling point | What "closing it" means | Reference |
| --- | --- | --- | --- |
| MolmoAct2 (VLA) | layer-wise cross-attention from the action expert to the VLM token sequence | mask image tokens out of every action-expert attention | `mask_image_from_action_expert` in `lerobot/src/lerobot/policies/molmoact2/modeling_molmoact2.py` |
| π0.5 (VLA) | one joint self-attention shared by the VLM and the action expert | attention mask: action-expert rows cannot attend image (or language/state) columns | mask in `src/pi05_goal_prior/modeling_pi05_goal_prior.py` |
| FAST-WAM (WAM) | video-DiT block features injected into the ActionDiT | drop the direct feature injection; only the interface remains | `goal_prior_stage: stage2` in `configs/model/fastwam_goal_prior_stage2.yaml`, wired in `fastwam_joint.py` / `mot.py` |
| ImageWAM (WAM) | FLUX.2 DiT block features → action head | same: the direct path is closed | `backbones/imagewam.py`, `configs/model/imagewam_flux2_klein_4b_goal_prior_stage2.yaml` |

### Step 2 — Add the latent interface

`N = 100` learnable tokens (`latent_dim = 768`) become the action expert's **only** visual input.

- **Reading side.** The latents attend to the backbone's visual *and* semantic features. Two ways that both
  worked: append them to the backbone sequence so the backbone contextualises them layer by layer (MolmoAct2,
  π0.5), or run a small aggregator — self-attention over the latents, then cross-attention to language/state, then
  to image features — per backbone layer group (FAST-WAM, ImageWAM: `inner_dim = 512`, 8 heads, 6 layer groups).
- **Writing side.** Hand the latents to the action expert at exactly the point closed in Step 1 — as the KV of its
  cross-attention, as the only columns its rows may attend, or as the injected features — layer-wise if the host
  conditions layer-wise.

Reference: `semantic_visual_recurrent` (MolmoAct2); `goal_prior.py` (Pi05, fastwam); `goal_pose_prior.py` (ImageWAM).

### Step 3 — Supervise 8 of the latents with the chunk's terminal pose

Reserve `num_pose_tokens = 8` latents. A 3-layer MLP (`GoalPoseDecoder`, `inner_dim = 512`) decodes them to the
**end-effector pose at the end of the current action chunk**: position (3) + axis-angle rotation (3) + gripper,
read from the dataset's state at `t + H` (`H = 10` in our runs) and quantile-normalised to `[-1, 1]` with the
dataset statistics. Loss: MSE, weight `λ = 0.3`, added to the host's native action loss.

Reference: `GoalPoseDecoder` in every fork; the target is built in the data processor
(`target_pose_delta_index` in MolmoAct2's `processor_molmoact2.py`).

### Step 4 — Add the Stage-1 SE(3) encoder (training-time only)

A 3-layer MLP that turns the same normalised pose into a few conditioning tokens for the action expert
(`SE3Encoder` / `GoalPoseEncoder`). It exists only so the action expert can be pretrained **without images**; it
is deleted after Stage 1 and never used at inference — the latents predict the pose from vision instead.

### Step 5 — Train in two stages

```text
Stage 1   backbone frozen · no images · action expert from scratch (or re-initialised)
          condition on  language + state + SE(3)-encoded terminal pose  →  native action loss      (10K steps)
Stage 2   init the action expert from Stage 1 · drop the SE(3) encoder · enable latents + pose decoder
          fine-tune everything:  L = L_action + 0.3 · L_pose                                          (30K steps)
```

Learning rates that mattered: the new modules (latents, aggregator) and the action expert at `1e-4` with a `5K`-step
warm-up; the backbone at `1e-5`. Interface settings were **not** tuned per architecture:
`num_latents = 100`, `num_pose_tokens = 8`, `latent_dim = 768`, `inner_dim = 512`, `lambda_pose = 0.3`.
Nothing else in the host's training recipe changes.

### Step 6 — Three checks before trusting a run

Each caught a silent failure for us at least once:

- **Is the interface used at all?** Measure the action expert's attention mass on the latent tokens vs. everything
  else (`tools/probe_latent_attention.py` in Pi05, `scripts/audit_goal_prior_v2.py` in ImageWAM). Near zero means
  the firewall leaks or a gate never opened.
- **Do the latents carry the pose?** Normalised pose loss should reach ~`1e-3` on training data; decode it and draw it
  back onto the frame (`scripts/render_pose_videos.sh`) — the predicted marker should lead the gripper, not trail it.
- **New parameters actually train and save.** Under bf16 autocast, small gates and freshly added modules can freeze
  or be dropped from the checkpoint; check the parameter count in `train_config.json` and that the new keys load.

## Worked example: MolmoAct2, train then evaluate

The MolmoAct2 fork carries the full recipe; the other three forks follow the same steps in their own READMEs.

```bash
git clone --recursive -b feat/libero-goal-prior-v4 https://github.com/jianmanlincjx/Molmoact2.git
git clone https://github.com/MAGICLAB-NUS/LIT.git && cd LIT

export LIT_MOLMOACT2=/path/to/Molmoact2
export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus            # github.com/sylvestf/LIBERO-plus
export DATASET_ROOT=/path/to/libero_lerobot_format      # LIBERO in LeRobot format, all four suites
bash scripts/preflight.sh                                # submodule commit, LIBERO-Plus, GPUs
```

**Train** (from the pretrained backbone; the `_v3` launchers' defaults equal the released `train_config.json`):

```bash
cd "$LIT_MOLMOACT2"
bash scripts/libero_goal_prior_v3/train_stage1.sh           # Stage 1: no images, 10K steps, batch 128/GPU
bash scripts/libero_goal_prior_v3/train_stage2.sh           # Stage 2: 100 latents (8 pose-supervised), 30K steps, from Stage 1
OPTIMIZER_ACTION_EXPERT_LR=1e-4 SCHEDULER_ACTION_EXPERT_WARMUP_STEPS=5000 \
  bash scripts/libero_goal_prior/train_baseline.sh          # (optional) matched baseline, 30K steps, batch 32/GPU
```

Batch sizes are per GPU (the paper used 7 GPUs). Stage 2 reports the Stage-1 SE(3) encoder as unexpected keys
when it loads — expected; the encoder is training-time only.

**Evaluate** `outputs/…/checkpoints/030000/pretrained_model` (or a released checkpoint from [Checkpoints](#checkpoints)):

```bash
cd LIT
bash scripts/eval_libero_plus.sh lit <checkpoint-dir>   # LIBERO-Plus: 10,030 tasks, one episode each, seed 1000
bash scripts/eval_libero.sh      lit <checkpoint-dir>   # LIBERO: 4 suites × 50 episodes
```

Both are resumable across GPUs and print the per-axis / per-suite result when they finish.
`scripts/aggregate.py <results-root>` re-aggregates later; `--pair <a> <b>` compares two runs on the tasks both finished.

## Two things that change the numbers

- **`LIBERO_PLUS_FIX_LANG=1`** — set by the scripts. Without it the LIBERO-Plus harness feeds the
  perturbation parameters to the policy as its instruction, and every axis drops by several points.
- **Overall = arithmetic mean over the seven perturbation axes** (Camera 1,599 · Noise 1,601 · Lighting 1,142 ·
  Background 1,076 · Robot 1,550 · Layout 1,525 · Language 1,537 tasks). The task-weighted rate is about two
  points lower; `aggregate.py` prints both and labels the reported one.

## Checkpoints

Stage-2 models (the ones in the tables) and the Stage-1 action priors they start from are on Hugging Face:
**[linjianman/LIT](https://huggingface.co/linjianman/LIT)** (public, no login needed; 43 GB for the four Stage-2 models, 84 GB with Stage 1).

```bash
hf download linjianman/LIT --include "*/lit_stage2/*" --local-dir LIT_ckpt        # the four reported models
hf download linjianman/LIT --include "molmoact2/*" --local-dir LIT_ckpt           # one framework, both stages
```

| Directory | Format | How to use it |
| --- | --- | --- |
| `molmoact2/lit_stage2`, `pi05/lit_stage2` | LeRobot policy directory (`config.json` + `model.safetensors` + normalisers) | `--policy.path <dir>` |
| `fastwam/lit_stage2`, `imagewam/lit_stage2` | `model.pt` + `config.yaml` + `dataset_stats.json` | `ckpt=<dir>/model.pt` `dataset_stats_path=<dir>/dataset_stats.json` |
| `*/lit_stage1` | same layout as the Stage 2 of that framework | start Stage 2 from it and skip Stage 1 — each fork's README ② gives the variable (`POLICY_PATH`, `STAGE1`, `resume=`, `STAGE1_CHECKPOINT`) |

The MolmoAct2 checkpoints used for evals, plus the MolmoAct2 baseline checkpoint, are public:
[shailes-h/Molmoact2-LIT](https://huggingface.co/shailes-h/Molmoact2-LIT)
(`hf download shailes-h/Molmoact2-LIT`).

**Real-robot data.** The demonstrations behind the real-robot results are public:
[shailes-h/yam_bimanual_manipulation](https://huggingface.co/datasets/shailes-h/yam_bimanual_manipulation)
— YAM dual-arm platform, three tasks (blocks → box, dust-pan wipe, egg transfer), 292 successful episodes,
three cameras, LeRobot v3.0, 4.8 GB (`hf download shailes-h/yam_bimanual_manipulation --repo-type dataset`).

## Code

One fork per framework. Every README has the same three parts — **① evaluate the released checkpoint**,
**② train, then evaluate** (from the pretrained base), **③ how LIT is integrated in that framework** —
and keeps the upstream README as `README_upstream.md`. Use the pinned branch and commit.

| Framework | Repository | Branch | Commit | Verified on a fresh machine |
| --- | --- | --- | --- | --- |
| MolmoAct2 | [Molmoact2](https://github.com/jianmanlincjx/Molmoact2) + [lerobot](https://github.com/jianmanlincjx/lerobot) submodule | `feat/libero-goal-prior-v4` | `bf8ca94` / `c802e8a6` | ✅ weights load; LIBERO and LIBERO-Plus rollouts |
| π0.5 | [Pi05](https://github.com/jianmanlincjx/Pi05) | `main` | `b227a65` | ✅ baseline and LIT weights load; rollouts |
| FAST-WAM | [fastwam](https://github.com/jianmanlincjx/fastwam) | `feat/goal-pose-prior` | `64dcd74` | ✅ LIT weights load; rollouts (environment recipe in its README) |
| ImageWAM | [ImageWAM](https://github.com/jianmanlincjx/ImageWAM) | `feat/goal-prior-bottleneck-fix` | `1792d10` | ✅ environment builds from `uv.lock` (torch 2.7.1+cu118, CUDA visible); evaluation not re-run here — needs the gated FLUX.2-dev autoencoder |

## Citation

```bibtex
@article{lin2026lit,
  title   = {Breaking the Vision--Action Shortcut: Latent Interface Training
             for Generalizable Robot Foundation Models},
  author  = {Lin, Jianman and Shailesh, Shailesh and Luo, Zhongyi and Duan, Jiafei},
  journal = {Under review},
  year    = {2026}
}
```
