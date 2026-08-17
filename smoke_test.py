from __future__ import annotations

"""Sanity check: load the checkpoints, run a forward pass with dummy data,
and confirm the Glue alignment works between two independently trained nets.

Usage: python smoke_test.py
"""

import torch

from composed_nets import fit_sj_glue, load_checkpoint


def check_vision(path: str) -> torch.Tensor:
    model = load_checkpoint(path)
    cfg = model.cfg
    video = torch.rand(2, cfg.video_frames_obs, 3, cfg.video_size, cfg.video_size)
    out = model(video)
    assert torch.isfinite(out["program"]).all()
    print(f"[vision]  program={tuple(out['program'].shape)} codes={tuple(out['codes'].shape)} "
          f"future_rgb={tuple(out['future_rgb'].shape)}")
    return out["codes"]


def check_physics(path: str) -> torch.Tensor:
    model = load_checkpoint(path)
    cfg = model.cfg
    state = torch.randn(2, cfg.video_frames_obs, cfg.max_objects, cfg.state_dim)
    mask = torch.ones(2, cfg.max_objects)
    out = model.encode_state(state, mask)
    rollout = model.rollout_from_codes(out["codes"])
    assert torch.isfinite(out["program"]).all()
    assert torch.isfinite(rollout["future_state"]).all()
    print(f"[physics] program={tuple(out['program'].shape)} codes={tuple(out['codes'].shape)} "
          f"future_state={tuple(rollout['future_state'].shape)}")
    return out["codes"]


def check_decoder(path: str) -> torch.Tensor:
    model = load_checkpoint(path)
    cfg = model.cfg
    future_state = torch.randn(2, cfg.video_frames_future, cfg.max_objects, cfg.state_dim)
    mask = torch.ones(2, cfg.max_objects)
    out = model(future_state, mask)
    assert torch.isfinite(out["outcome_pred"]).all()
    print(f"[decoder] outcome_pred={tuple(out['outcome_pred'].shape)} codes={tuple(out['codes'].shape)}")
    return out["codes"]


def check_glue(vision_path: str, physics_path: str) -> None:
    vision = load_checkpoint(vision_path)
    physics = load_checkpoint(physics_path)

    torch.manual_seed(0)
    video = torch.rand(64, vision.cfg.video_frames_obs, 3, vision.cfg.video_size, vision.cfg.video_size)
    state = torch.randn(64, physics.cfg.video_frames_obs, physics.cfg.max_objects, physics.cfg.state_dim)
    mask = torch.ones(64, physics.cfg.max_objects)

    codes_a = vision.codes(video).numpy()
    codes_b = physics.encode_state(state, mask)["codes"].numpy()

    glue = fit_sj_glue(codes_a, codes_b, vision.sj.masks, physics.sj.masks, card=vision.cfg.card)
    print(f"[glue]    compatible={glue.compatible} group_map={glue.group_map} "
          f"resolved_slots={glue.resolved_slots} objective={glue.objective:.4f} margin={glue.margin:.4f}")


if __name__ == "__main__":
    check_vision("movi_vision_sj_seed11.pkl")
    check_physics("movi_physics_sj_seed11.pkl")
    check_decoder("movi_decoder_sj_seed11.pkl")
    check_glue("movi_vision_sj_seed11.pkl", "movi_physics_sj_seed11.pkl")
    print("OK: all checkpoints loaded and ran successfully.")
