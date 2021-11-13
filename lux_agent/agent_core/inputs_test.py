import sys
import pytest
import json
import numpy as np

from .inputs import game_state_to_global_input, game_state_to_input
from ..lux.game import Game

np.set_printoptions(threshold=sys.maxsize)

@pytest.fixture
def game_replay():
    with open('lux_agent/mocks/30468699.json') as json_file:
        replay = json.load(json_file)
        observation = replay['steps'][0][0]['observation']
        game_state = Game()
        game_state._initialize(observation['updates'])

        return game_state, replay

    assert False, 'Cannot to open game replay'


def test_game_state_to_global_input(game_replay, snapshot):
    game_state, replay = game_replay
    observation = replay['steps'][10][0]['observation']
    game_state._update(observation["updates"])
    output = game_state_to_global_input(game_state, 0)
    snapshot.snapshot_dir = 'snapshots'
    snapshot.assert_match(
        np.array2string(output, precision=2, separator=',', suppress_small=True),
        'game_state_to_global_input.txt'
    )


def test_game_state_to_input(game_replay, snapshot):
    game_state, replay = game_replay
    observation = replay['steps'][10][0]['observation']
    game_state._update(observation["updates"])
    output = game_state_to_input(game_state, 0, 'u_1')
    snapshot.snapshot_dir = 'snapshots'
    snapshot.assert_match(
        np.array2string(output, precision=2, separator=',', suppress_small=True),
        'game_state_to_input.txt'
    )
