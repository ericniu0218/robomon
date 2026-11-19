"""SmolVLA wrapper: one query returns K sampled action chunks and an observation embedding.

Uses exactly the processors `lerobot-eval` builds (env pre-processor for LIBERO, the checkpoint's own
pre/post-processors with the camera rename map). The K samples share one observation and differ only
in the flow-matching start noise, which is drawn from a caller-supplied generator (reproducible).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

RENAME_MAP = {"observation.images.image": "observation.images.camera1",
              "observation.images.image2": "observation.images.camera2"}


@dataclass
class Query:
    chunks_norm: np.ndarray   # (K, chunk, action_dim) in the policy's normalized action space
    exec_actions: np.ndarray  # (chunk, action_dim) env-space actions of sample 0
    embedding: np.ndarray     # (d,) masked mean of the VLM prefix input embeddings


class SmolVLARunner:
    def __init__(self, path: str = "lerobot/smolvla_libero", threads: int = 2):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        torch.set_num_threads(threads)
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = "cpu"
        self.policy = SmolVLAPolicy.from_pretrained(path, config=cfg).eval()
        self.cfg = self.policy.config
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.cfg, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": "cpu"},
                                    "rename_observations_processor": {"rename_map": RENAME_MAP}})
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=LiberoEnvConfig(), policy_cfg=self.cfg)
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

    def _sample_shared_prefix(self, o: dict, noise: torch.Tensor) -> torch.Tensor:
        """K samples for ONE observation: encode images + instruction + state once, copy the key/value
        cache K times, and run only the action expert for the K noise seeds. Mirrors
        `VLAFlowMatching.sample_actions`; tested to match the repeat-the-batch reference path."""
        from lerobot.policies.smolvla import modeling_smolvla as M
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        pol, m, k = self.policy, self.policy.model, noise.shape[0]
        batch = pol._prepare_batch(dict(o))
        images, img_masks = pol.prepare_images(batch)
        state = pol.prepare_state(batch)
        embs, pad, att = m.embed_prefix(images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                                        batch[OBS_LANGUAGE_ATTENTION_MASK], state=state)
        att2d = M.make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        _, pkv = m.vlm_with_expert.forward(attention_mask=att2d, position_ids=pos, past_key_values=None,
                                           inputs_embeds=[embs, None], use_cache=m.config.use_cache)
        pkv.batch_repeat_interleave(k)
        pad_k = pad.expand(k, -1)
        x = M.euler_integrate(
            lambda x_t, ts: m.denoise_step(prefix_pad_masks=pad_k, past_key_values=pkv, x_t=x_t, timestep=ts),
            noise.to(torch.float32), m.config.num_steps)
        return x[:, :, : self.action_dim]

    @torch.no_grad()
    def query(self, obs: dict, task: str, k: int, generator: torch.Generator,
              shared_prefix: bool = True) -> Query:
        from lerobot.envs.utils import prepare_canonical_observation
        from lerobot.sim.integrations.libero import canonical_observation
        from lerobot.utils.constants import ACTION

        # same path as lerobot-eval: LIBERO obs -> LeRobot keys (images rotated 180 deg, 8-D state) -> tensors
        canon = canonical_observation(obs)
        o = prepare_canonical_observation({name: np.asarray(v)[None] for name, v in canon.items()})
        o["task"] = [task]
        o = self.pre(self.env_pre(o))
        noise = torch.randn((k, self.chunk_size, int(self.cfg.max_action_dim)), generator=generator)
        if shared_prefix:
            chunks = self._sample_shared_prefix(o, noise)
        else:  # reference path: repeat the whole observation k times through lerobot's own method
            batch = {key: (v.repeat_interleave(k, dim=0) if isinstance(v, torch.Tensor) and v.shape[0] == 1 else v)
                     for key, v in o.items()}
            chunks = self.policy.predict_action_chunk(batch, noise=noise)  # (k, chunk, action_dim), normalized
        embs, pad = self._captured["embs"][0].float(), self._captured["pad"][0].bool()
        embedding = embs[pad].mean(0)
        act = self.post(chunks[0].clone())                      # unnormalize sample 0 -> (chunk, action_dim)
        act = self.env_post({ACTION: act})[ACTION]
        return Query(chunks_norm=chunks.float().numpy(), exec_actions=act.float().numpy(),
                     embedding=embedding.numpy().astype(np.float32))
