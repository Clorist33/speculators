"""Native token masks must supervise the next-token DSpark anchor targets."""
import runpy
from pathlib import Path

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dspark import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel

prediction_mask = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / 'scripts/prepare_ascendc_dspark_formal.py')
)['prediction_mask']


def test_native_reply_boundary_matches_dspark_prediction_positions():
    torch.manual_seed(7)
    cfg = Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                     vocab_size=64, _attn_implementation='eager')
    model = DSparkDraftModel(DSparkSpeculatorConfig(
        transformer_layer_config=cfg, draft_vocab_size=64, block_size=8,
        aux_hidden_state_layer_ids=[0, 1], mask_token_id=0,
        sample_from_anchor=True, markov_rank=4,
    )).eval()
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    native = [0]*12 + [1]*8 + [0]*8 + [1]*12 + [0]*8
    shifted = prediction_mask(native)
    assert shifted[11] == 1 and native[11] == 0  # predicts first reply token
    assert shifted[19] == 0 and native[19] == 1  # predicts following context
    assert not any(shifted[-8:])
    n = len(native)
    with torch.no_grad():
        _, _, targets, aligned, positions = model._backbone_forward(
            torch.randn(1,n,32), torch.arange(n).unsqueeze(0),
            torch.tensor([shifted]), torch.randn(1,n,16),
            torch.zeros(1,n,dtype=torch.long), max_anchors=4,
        )
    expected = torch.tensor(native)[positions + 1].unsqueeze(0)
    assert aligned.sum() > 0
    assert torch.all((aligned == 0) | (expected == 1))
    assert targets.shape[:2] == aligned.shape
