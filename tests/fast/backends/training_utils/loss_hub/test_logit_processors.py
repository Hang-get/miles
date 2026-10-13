from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.loss.hub import logit_processors


class TestGetLogProbsAndEntropy:
    def test_stored_log_probs_forward_the_debug_grad_switch_from_training_args(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fused_cross_entropy_calls: list[tuple[bool, bool]],
    ) -> None:
        """Stored log probabilities forward the debug grad contract from training arguments."""
        parallel_state = SimpleNamespace(
            tp=SimpleNamespace(rank=0, group=None),
            cp=SimpleNamespace(rank=0, size=1),
        )
        monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
        monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: parallel_state)
        args = Namespace(
            log_probs_backend="torch",
            qkv_format="thd",
            rollout_temperature=1.0,
            true_on_policy_mode=False,
            log_probs_chunk_size=-1,
            allgather_cp=False,
            debug_unified_grad_fused_logprob=True,
        )

        with torch.no_grad():
            result = logit_processors.get_log_probs_and_entropy(
                torch.randn(1, 4, 8),
                args=args,
                unconcat_tokens=[torch.tensor([0, 1, 2, 3])],
                total_lengths=[4],
                response_lengths=[3],
            )

        assert fused_cross_entropy_calls == [(True, True)]
        assert not result["log_probs"][0].requires_grad


class TestTokenLogProbs:
    def test_response_slices_match_the_logit_slicing_of_real_logits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A backend's per-token log-probs, sliced by TokenLogProbs, equal what miles derives from full logits."""
        parallel_state = SimpleNamespace(tp=SimpleNamespace(rank=0, group=None), cp=SimpleNamespace(rank=0, size=1))
        monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
        args = Namespace(qkv_format="thd", rollout_temperature=1.0, true_on_policy_mode=False, allgather_cp=False)
        torch.manual_seed(0)
        samples = [torch.randint(0, 11, (n,)) for n in (5, 7, 4)]
        response_lengths = [3, 6, 2]
        tokens = torch.cat(samples)
        logits = torch.randn(1, tokens.numel(), 11)
        full = torch.log_softmax(logits[0], dim=-1)
        next_tokens = torch.roll(tokens, -1)
        per_token = full.gather(1, next_tokens[:, None]).squeeze(1)
        entropy = -(full.exp() * full).sum(-1)

        result = logit_processors.TokenLogProbs(per_token[None], entropy[None]).response_slices(
            args=args,
            total_lengths=[len(s) for s in samples],
            response_lengths=response_lengths,
            with_entropy=True,
            entropy_requires_grad=False,
        )

        chunks = logit_processors._iter_response_chunks(
            logits,
            args=args,
            unconcat_tokens=samples,
            total_lengths=[len(s) for s in samples],
            response_lengths=response_lengths,
            include_response_indices=False,
        )
        for got_lp, got_ent, (logits_chunk, tokens_chunk, _) in zip(
            result["log_probs"], result["entropy"], chunks, strict=True
        ):
            want = torch.log_softmax(logits_chunk, dim=-1)
            torch.testing.assert_close(got_lp, want.gather(1, tokens_chunk[:, None]).squeeze(1))
            torch.testing.assert_close(got_ent, -(want.exp() * want).sum(-1))
