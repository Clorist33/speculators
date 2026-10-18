"""Raw trajectories stay in RAM and enter the existing teacher-extraction path."""
import json
from unittest.mock import Mock

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from speculators.data_generation.trajectory import (
    inspect_trajectory_inputs,
    prepare_in_memory_trajectories,
)
from speculators.data_generation.trajectory_cache import load_trajectory_cache
from speculators.train.data import ArrowDataset
from speculators.train.dataloader import create_train_val_loaders
from speculators.train.config import TrainConfig


@pytest.fixture
def tokenizer():
    backend = Tokenizer(WordLevel({
        '[UNK]': 0, 'user': 1, 'assistant': 2, 'task': 3,
        'first': 4, 'answer': 5, 'continue': 6, 'final': 7,
    }, unk_token='[UNK]'))
    backend.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]')
    tok.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})
    tok.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}"
        "{% endfor %}"
    )
    return tok


@pytest.fixture
def source(tmp_path):
    raw = tmp_path / 'raw.jsonl'
    split = tmp_path / 'split.jsonl'
    example = {'messages': [
        {'role': 'user', 'content': 'task'},
        {'role': 'assistant', 'content': 'first answer first answer'},
        {'role': 'user', 'content': 'continue'},
        {'role': 'assistant', 'content': 'final answer final answer'},
    ]}
    raw.write_text(''.join(json.dumps(example) + '\n' for _ in range(4)))
    split.write_text(''.join(json.dumps(dict(
        shard=raw.name, line=i, split='validation' if i in (1, 3) else 'train',
        group_id='val' if i in (1, 3) else 'train',
    )) + '\n' for i in range(1, 5)))
    return raw, split


def test_memory_storage_and_explicit_split(source, tokenizer, monkeypatch):
    from datasets import Dataset
    monkeypatch.setattr(Dataset, 'save_to_disk', Mock(side_effect=AssertionError('disk write')))
    raw, split = source
    data, ratio, manifest = prepare_in_memory_trajectories(
        raw, split, 'unused', 128, 2, tokenizer=tokenizer,
    )
    assert len(data) == 4 and not data.cache_files
    assert int(len(data) * ratio) == 2
    assert manifest['counts']['train'] == manifest['counts']['validation'] == 2
    assert [x['source_line'] for x in manifest['trajectories']] == [1, 2, 3, 4]
    Dataset.save_to_disk.assert_not_called()
    assert sorted(p.name for p in raw.parent.iterdir()) == ['raw.jsonl', 'split.jsonl']


def test_split_leakage_and_missing_lines_fail(source, tokenizer):
    raw, split = source
    metas = [json.loads(s) for s in split.read_text().splitlines()]
    metas[0]['group_id'] = 'train'
    split.write_text(''.join(json.dumps(x)+'\n' for x in metas))
    with pytest.raises(ValueError, match='leakage'):
        prepare_in_memory_trajectories(raw, split, 'unused', 128, 2, tokenizer=tokenizer)
    split.write_text(''.join(json.dumps(x)+'\n' for x in metas[1:]))
    with pytest.raises(ValueError, match='no trajectory split'):
        prepare_in_memory_trajectories(raw, split, 'unused', 128, 2, tokenizer=tokenizer)


def test_multiple_shards_use_global_split_and_explicit_quarantine(tmp_path, tokenizer):
    example = {'messages': [
        {'role': 'user', 'content': 'task'},
        {'role': 'assistant', 'content': 'first answer first answer'},
    ]}
    shards = ['train-00000.jsonl', 'train-00001.jsonl']
    for name in shards:
        (tmp_path / name).write_text((json.dumps(example) + '\n') * 2)
    split = tmp_path / 'split.jsonl'
    rows = [
        dict(shard=shards[0], line=1, split='train', group_id='train'),
        dict(shard=shards[0], line=2, split='train', group_id='train'),
        dict(shard=shards[1], line=1, split='validation', group_id='validation'),
    ]
    split.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    quarantine = tmp_path / 'quarantine.json'
    quarantine.write_text(json.dumps([{'id': '00001:2', 'error': 'bad template'}]))

    sources, assignments, excluded = inspect_trajectory_inputs(
        tmp_path, split, shards, quarantine, check_lines=True,
    )
    assert len(sources) == 2
    assert sum(len(rows) for rows in assignments.values()) == 3
    assert excluded == {(shards[1], 2)}
    data, ratio, manifest = prepare_in_memory_trajectories(
        tmp_path, split, 'unused', 128, 2, tokenizer=tokenizer,
        shard_names=shards, quarantine_path=quarantine,
    )
    assert len(data) == 3 and int(len(data) * ratio) == 2
    assert manifest['counts']['quarantined'] == 1
    assert [row['source_shard'] for row in manifest['trajectories']] == [
        shards[0], shards[0], shards[1],
    ]

    quarantine.write_text('[]')
    with pytest.raises(ValueError, match='no trajectory split'):
        inspect_trajectory_inputs(tmp_path, split, shards, quarantine, check_lines=True)
    rows[-1]['group_id'] = 'train'
    split.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='leakage'):
        inspect_trajectory_inputs(tmp_path, split, shards, quarantine)


def test_memory_row_reaches_teacher_and_dspark(source, tokenizer, monkeypatch, tmp_path):
    import speculators.train.data as module
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from speculators.models.dspark import DSparkSpeculatorConfig
    from speculators.models.dspark.core import DSparkDraftModel
    from speculators.losses import resolve_loss_config
    from speculators.losses.eager import tv_loss
    raw, split = source
    data, ratio, _ = prepare_in_memory_trajectories(
        raw, split, 'unused', 128, 2, tokenizer=tokenizer,
    )
    monkeypatch.setattr(module, 'load_from_disk', Mock(side_effect=AssertionError('disk read')))
    transfer = Mock()
    transfer.get_cached.return_value = None
    def setup(self):
        self.client = object()
    monkeypatch.setattr(ArrowDataset, '_setup_client', setup)
    requests = []
    def teacher(client, model, item, **kwargs):
        requests.append(item['input_ids'])
        n = len(item['input_ids'])
        transfer.get_generated.return_value = {
            'token_ids': torch.tensor(item['input_ids']),
            'hidden_states': torch.randn(n, 3, 16),
        }
        return 'temporary-hidden-states-handle'
    monkeypatch.setattr(module, 'generate_hidden_states', teacher)
    train, val = create_train_val_loaders(
        data_path=str(raw), in_memory_data=data, total_seq_len=128,
        hidden_states_dtype=torch.float32, noise_std=0, transfer=transfer,
        vllm_endpoint='http://unused/v1', on_missing='generate', on_generate='delete',
        verifier_name_or_path='unused', request_timeout=1, max_retries=0,
        generation_validation_retries=0, max_consecutive_generation_failures=2,
        hidden_size=16, num_target_layers=2, num_workers=0, prefetch_factor=2,
        preprocess=None, train_data_ratio=ratio,
    )
    assert len(train.dataset) == len(val.dataset) == 2
    assert train.dataset._map_to_file_idx(0) == 0
    assert val.dataset._map_to_file_idx(0) == 2
    batch = next(iter(train))
    assert requests and all(isinstance(x, int) for x in requests[0])
    module.load_from_disk.assert_not_called()
    assert transfer.delete.called
    cfg = Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                      vocab_size=64, _attn_implementation='eager')
    model = DSparkDraftModel(DSparkSpeculatorConfig(
        transformer_layer_config=cfg, draft_vocab_size=64, block_size=2,
        aux_hidden_state_layer_ids=[0,1], mask_token_id=0,
        sample_from_anchor=True, markov_rank=4,
    ))
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    with torch.no_grad():
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                torch.nn.init.normal_(parameter, std=0.02)
    _, loss, _ = model(**batch, max_anchors=2, tv_loss_fn=tv_loss,
                       loss_config=resolve_loss_config('{"ce":0.1,"tv":0.9}', 'eager'))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


def test_config_accepts_memory_flags():
    cfg = TrainConfig.resolve([
        '--verifier-name-or-path', '/unused', '--speculator-type', 'dspark',
        '--data-format', 'trajectory-jsonl', '--data-path', '/raw.jsonl',
        '--trajectory-split-path', '/split.jsonl', '--trajectory-overlength', 'skip',
        '--trajectory-shards', 'train-00000.jsonl', 'train-00001.jsonl',
        '--trajectory-quarantine-path', '/quarantine.json',
    ])
    assert cfg.data.data_format == 'trajectory-jsonl'
    assert cfg.data.trajectory_split_path == '/split.jsonl'
    assert cfg.data.trajectory_overlength == 'skip'
    assert cfg.data.trajectory_shards == ['train-00000.jsonl', 'train-00001.jsonl']
    assert cfg.data.trajectory_quarantine_path == '/quarantine.json'


def test_saved_trajectory_cache_reuses_rows_and_rejects_mismatches(
    source, tokenizer, tmp_path,
):
    raw, split = source
    original, ratio, manifest = prepare_in_memory_trajectories(
        raw, split, 'unused', 128, 2, tokenizer=tokenizer,
    )
    cache = tmp_path / 'cache'
    original.save_to_disk(str(cache))
    manifest['train_ratio'] = ratio
    manifest['saved_dataset_path'] = str(cache)
    (cache / 'preprocess_manifest.json').write_text(json.dumps(manifest))

    loaded, cached_ratio, loaded_manifest = load_trajectory_cache(
        cache, model_path='unused', max_length=128, block_size=2,
        overlength='error', shard_names=[raw.name], split_path=str(split),
        tokenizer=tokenizer,
    )
    assert loaded.cache_files
    assert cached_ratio == ratio
    assert loaded_manifest['train_rows'] == 2
    assert loaded['input_ids'] == original['input_ids']

    transfer = Mock()
    train, val = create_train_val_loaders(
        data_path=str(cache), in_memory_data=loaded, total_seq_len=128,
        hidden_states_dtype=torch.float32, noise_std=0, transfer=transfer,
        vllm_endpoint='http://unused/v1', on_missing='generate', on_generate='delete',
        verifier_name_or_path='unused', request_timeout=1, max_retries=0,
        generation_validation_retries=0, max_consecutive_generation_failures=2,
        hidden_size=16, num_target_layers=2, num_workers=0, prefetch_factor=2,
        preprocess=None, train_data_ratio=cached_ratio,
    )
    assert len(train.dataset) == len(val.dataset) == 2
    assert train.dataset._map_to_file_idx(0) == 0
    assert val.dataset._map_to_file_idx(0) == 2

    with pytest.raises(ValueError, match='block_size mismatch'):
        load_trajectory_cache(
            cache, model_path='unused', max_length=128, block_size=3,
            overlength='error', tokenizer=tokenizer,
        )
    split.write_text(split.read_text() + '\n')
    with pytest.raises(ValueError, match='split_sha256 mismatch'):
        load_trajectory_cache(
            cache, model_path='unused', max_length=128, block_size=2,
            overlength='error', split_path=str(split), tokenizer=tokenizer,
        )
