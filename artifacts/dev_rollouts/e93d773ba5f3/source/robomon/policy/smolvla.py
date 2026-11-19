"""SmolVLA wrapper: one query returns K sampled action chunks and an observation embedding.

Uses exactly the processors `lerobot-eval` builds (env pre-processor for LIBERO, the checkpoint's own
pre/post-processors with the camera rename map). The K samples share one observation and differ only
in the flow-matching start noise, which is drawn from a caller-supplied generator (reproducible).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
CHECKPOINT_REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
EMBEDDING_DEFINITION = "smolvla.prefix_input_masked_mean.batch1.v1"


@dataclass
class Query:
    chunks_norm: (
        np.ndarray
    )  # (K, chunk, action_dim) in the policy's normalized action space
    exec_actions: np.ndarray  # (chunk, action_dim) env-space actions of sample 0
    embedding: np.ndarray  # (d,) masked mean of the VLM prefix input embeddings


class SmolVLARunner:
    def __init__(
        self,
        path: str = "lerobot/smolvla_libero",
        threads: int = 2,
        revision: str | None = None,
        device: str = "cpu",
    ):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from huggingface_hub import constants as hub_constants, snapshot_download

        torch.set_num_threads(threads)
        if device not in ("cpu", "mps"):
            raise ValueError("Supported execution devices are cpu and mps")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        revision = revision or (
            CHECKPOINT_REVISION if path == "lerobot/smolvla_libero" else None
        )
        self.path, self.revision = path, revision
        cfg = PreTrainedConfig.from_pretrained(path, revision=revision)
        cfg.device = device
        if cfg.vlm_model_name != VLM_REPO:
            raise ValueError(
                "Pin the base VLM and tokenizer revision before using another model"
            )
        # LeRobot's VLM constructor does not expose a revision parameter. A pinned local snapshot
        # fixes both the architecture/tokenizer files and initial VLM weights it loads internally.
        vlm_path = snapshot_download(
            VLM_REPO,
            revision=VLM_REVISION,
            allow_patterns=["*.json", "*.txt", "*.jinja", "*.safetensors", "*.model"],
            local_files_only=hub_constants.HF_HUB_OFFLINE,
        )
        cfg.vlm_model_name = vlm_path
        self.policy = SmolVLAPolicy.from_pretrained(
            path, config=cfg, revision=revision
        ).eval()
        self.cfg = self.policy.config
        if (
            self.cfg.n_obs_steps != 1
            or self.cfg.adapt_to_pi_aloha
            or self.cfg.rtc_config is not None
        ):
            raise ValueError(
                "This audit requires a stateless, one-observation LIBERO policy without RTC"
            )
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.cfg,
            pretrained_path=path,
            pretrained_revision=revision,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "tokenizer_processor": {"tokenizer_name": vlm_path},
                "rename_observations_processor": {"rename_map": RENAME_MAP},
            },
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(
            env_cfg=LiberoEnvConfig(), policy_cfg=self.cfg
        )
        self.chunk_size = int(self.cfg.chunk_size)
        self.action_dim = int(self.cfg.action_feature.shape[0])
        # capture the prefix embeddings (image tokens + language + state) of each query
        self._captured: dict = {}
        orig = self.policy.model.embed_prefix

        def wrapped(*a, **kw):
            out = orig(*a, **kw)
            self._captured["embs"], self._captured["pad"] = out[0], out[1]
            return out

        self.policy.model.embed_prefix = wrapped

    def _prepare_env_observation(self, obs: dict, task: str) -> dict:
        from lerobot.envs.utils import prepare_canonical_observation
        from lerobot.sim.integrations.libero import canonical_observation

        canon = canonical_observation(obs)
        batch = prepare_canonical_observation(
            {name: np.asarray(v)[None] for name, v in canon.items()}
        )
        batch["task"] = [task]
        return self.pre(self.env_pre(batch))

    def _prepare_demo_frame(self, frame: dict) -> dict:
        """LeRobotDataset already returns canonical CHW [0,1] images: DO NOT rotate them again."""
        task = frame.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("A dataset frame needs a nonempty task instruction")
        batch = {"task": [task]}
        for key in (*RENAME_MAP, "observation.state"):
            value = torch.as_tensor(frame[key])
            expected = (8,) if key == "observation.state" else (3, 256, 256)
            if (
                tuple(value.shape) != expected
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"Invalid canonical dataset feature: {key}, {value.shape}, {value.dtype}"
                )
            if key in RENAME_MAP and (value.min() < 0 or value.max() > 1):
                raise ValueError("Dataset images must be floating point in [0,1]")
            batch[key] = value.to(dtype=torch.float32).clone().unsqueeze(0)
        return self.pre(self.env_pre(batch))

    @torch.no_grad()
    def _embed_prepared(self, batch: dict) -> np.ndarray:
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        batch = self.policy._prepare_batch(dict(batch))
        images, masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        embs, pad, _ = self.policy.model.embed_prefix(
            images,
            masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state=state,
        )
        return embs[0].float()[pad[0].bool()].mean(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_demo_frame(self, frame: dict) -> np.ndarray:
        """Same single-observation prefix as online support, without sampling or executing actions."""
        self.policy.reset()
        return self._embed_prepared(self._prepare_demo_frame(frame))

    def _sample_shared_prefix(self, o: dict, noise: torch.Tensor) -> torch.Tensor:
        """K samples for ONE observation: encode images + instruction + state once, copy the key/value
        cache K times, and run only the action expert for the K noise seeds. Mirrors
        `VLAFlowMatching.sample_actions`; tested to match the repeat-the-batch reference path."""
        from lerobot.policies.smolvla import modeling_smolvla as M
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        pol, m, k = self.policy, self.policy.model, noise.shape[0]
        batch = pol._prepare_batch(dict(o))
        images, img_masks = pol.prepare_images(batch)
        state = pol.prepare_state(batch)
        embs, pad, att = m.embed_prefix(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state=state,
        )
        att2d = M.make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        _, pkv = m.vlm_with_expert.forward(
            attention_mask=att2d,
            position_ids=pos,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=m.config.use_cache,
        )
        pkv.batch_repeat_interleave(k)
        pad_k = pad.expand(k, -1)
        x = M.euler_integrate(
            lambda x_t, ts: m.denoise_step(
                prefix_pad_masks=pad_k, past_key_values=pkv, x_t=x_t, timestep=ts
            ),
            noise.to(device=self.cfg.device, dtype=torch.float32),
            m.config.num_steps,
        )
        return x[:, :, : self.action_dim]

    @torch.no_grad()
    def query(
        self,
        obs: dict,
        task: str,
        k: int,
        generator: torch.Generator,
        shared_prefix: bool = False,
    ) -> Query:
        from lerobot.utils.constants import ACTION

        if k < 1:
            raise ValueError("k must be positive")
        # n_obs_steps=1: clearing queues makes queries independent of earlier shadow/batched queries.
        self.policy.reset()
        self._captured.clear()
        # same path as lerobot-eval: LIBERO obs -> LeRobot keys (images rotated 180 deg, 8-D state) -> tensors
        o = self._prepare_env_observation(obs, task)
        noise = torch.randn(
            (k, self.chunk_size, int(self.cfg.max_action_dim)), generator=generator
        )
        if shared_prefix:
            chunks = self._sample_shared_prefix(o, noise)
        else:  # reference path: repeat the whole observation k times through lerobot's own method
            batch = {
                key: (
                    v.repeat_interleave(k, dim=0)
                    if isinstance(v, torch.Tensor) and v.shape[0] == 1
                    else v
                )
                for key, v in o.items()
            }
            chunks = self.policy.predict_action_chunk(
                batch, noise=noise
            )  # (k, chunk, action_dim), normalized
        embs, pad = self._captured["embs"][0].float(), self._captured["pad"][0].bool()
        # Reference K-sample vision batches can have backend-dependent low-precision numerics.
        # Support always uses the SAME batch-one definition as the demonstration bank.
        embedding = (
            self._embed_prepared(o)
            if not shared_prefix and k > 1
            else embs[pad].mean(0).cpu().numpy().astype(np.float32)
        )
        act = self.post(
            chunks[0].clone()
        )  # unnormalize sample 0 -> (chunk, action_dim)
        act = self.env_post({ACTION: act})[ACTION]
        return Query(
            chunks_norm=chunks.float().cpu().numpy(),
            exec_actions=act.float().cpu().numpy(),
            embedding=embedding,
        )
