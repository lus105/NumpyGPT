import numpy as np
import pytest

from src.gpt2 import generate


N_VOCAB = 50
N_EMBD = 16
N_HEAD = 2
N_LAYER = 2


def _make_params():
    rng = np.random.default_rng(42)

    def r(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    def block():
        return {
            'attn': {
                'c_attn': {'w': r(N_EMBD, 3 * N_EMBD), 'b': r(3 * N_EMBD)},
                'c_proj': {'w': r(N_EMBD, N_EMBD), 'b': r(N_EMBD)},
            },
            'mlp': {
                'c_fc': {'w': r(N_EMBD, 4 * N_EMBD), 'b': r(4 * N_EMBD)},
                'c_proj': {'w': r(4 * N_EMBD, N_EMBD), 'b': r(N_EMBD)},
            },
            'ln_1': {
                'g': np.ones(N_EMBD, dtype=np.float32),
                'b': np.zeros(N_EMBD, dtype=np.float32),
            },
            'ln_2': {
                'g': np.ones(N_EMBD, dtype=np.float32),
                'b': np.zeros(N_EMBD, dtype=np.float32),
            },
        }

    return {
        'wte': r(N_VOCAB, N_EMBD),
        'wpe': r(128, N_EMBD),
        'blocks': [block() for _ in range(N_LAYER)],
        'ln_f': {'g': np.ones(N_EMBD, dtype=np.float32), 'b': np.zeros(N_EMBD, dtype=np.float32)},
    }


@pytest.mark.slow
def test_generate_end_to_end():
    params = _make_params()
    inputs = [1, 2, 3]
    n_generate = 10

    output = generate(inputs[:], params, N_HEAD, n_generate)

    assert len(output) == n_generate, 'should return exactly n_tokens_to_generate tokens'
    assert all(isinstance(t, int) for t in output), 'tokens should be integers'
    assert all(0 <= t < N_VOCAB for t in output), 'all tokens should be valid vocab ids'

    # greedy decoding is deterministic — same inputs must yield same output
    output2 = generate(inputs[:], params, N_HEAD, n_generate)
    assert output == output2, 'generate must be deterministic'
