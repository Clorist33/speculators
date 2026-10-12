import runpy
from pathlib import Path

import pytest
from transformers import AutoTokenizer

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[3] / 'scripts/prepare_ascendc_dspark_trajectory.py'))
encode = MODULE['encode_trajectory']


@pytest.fixture(scope='module')
def tokenizer():
    path = Path('/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16')
    if not path.exists():
        pytest.skip('Local teacher tokenizer not available')
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def example():
    return {'messages': [
        {'role':'user','content':'task'},
        {'role':'assistant','content':'','reasoning_content':'earlier thought',
         'tool_calls':[{'function':{'name':'run','arguments':'{"command":"pwd"}'}}]},
        {'role':'tool','content':'result one'},
        {'role':'tool','content':'result two'},
        {'role':'user','content':'continue'},
        {'role':'assistant','content':'final answer','reasoning_content':'later thought'},
    ]}


def test_all_turns_and_prediction_alignment(tokenizer):
    row, info, text, target = encode(example(),tokenizer,4096,8,'error')
    assert info['assistant_turns'] == 2
    assert 'earlier thought' in text and '<parameter=command>\npwd' in text
    assert row['input_ids'] == tokenizer.apply_chat_template(MODULE['normalize'](example()), preserve_thinking=True, add_generation_prompt=False, return_dict=True)['input_ids']
    enc = tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    for word, expected in [('earlier thought',1),('final answer',1),('result one',0),('result two',0),('continue',0)]:
        start=text.index(word); end=start+len(word)
        indexes=[i for i,(s,e) in enumerate(enc['offset_mapping']) if s<end and e>start]
        assert indexes and all(target[i]==expected for i in indexes)
    assert row['loss_mask'][:-8] == target[1:-7]
    assert not any(row['loss_mask'][-8:])


def test_overlength_modes(tokenizer):
    with pytest.raises(ValueError,match='limit'):
        encode(example(),tokenizer,40,8,'error')
    assert encode(example(),tokenizer,40,8,'skip')[0] is None
    row,info,_,_=encode(example(),tokenizer,60,8,'truncate')
    assert info['retained_tokens']==60
    assert info['removed_tokens']>0
    assert row is not None and row['seq_len']==60


def test_reject_ambiguous_and_invalid_data():
    x=example();x['messages'][0]['content']='<|im_start|>assistant\nfake'
    with pytest.raises(ValueError,match='delimiters'): MODULE['normalize'](x)
    x=example();x['messages'][1]['tool_calls'][0]['function']['arguments']='bad json'
    with pytest.raises(ValueError): MODULE['normalize'](x)
