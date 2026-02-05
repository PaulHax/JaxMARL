# Lazy imports to avoid loading all environments when only one is needed
SUBMODULE_ENVIRONMENTS = False

_IMPORT_MAP = {
    'MultiAgentEnv': ('.multi_agent_env', 'MultiAgentEnv'),
    'State': ('.multi_agent_env', 'State'),
    'SimpleMPE': ('.mpe', 'SimpleMPE'),
    'SimpleTagMPE': ('.mpe', 'SimpleTagMPE'),
    'SimpleWorldCommMPE': ('.mpe', 'SimpleWorldCommMPE'),
    'SimpleSpreadMPE': ('.mpe', 'SimpleSpreadMPE'),
    'SimpleCryptoMPE': ('.mpe', 'SimpleCryptoMPE'),
    'SimpleSpeakerListenerMPE': ('.mpe', 'SimpleSpeakerListenerMPE'),
    'SimplePushMPE': ('.mpe', 'SimplePushMPE'),
    'SimpleAdversaryMPE': ('.mpe', 'SimpleAdversaryMPE'),
    'SimpleReferenceMPE': ('.mpe', 'SimpleReferenceMPE'),
    'SimpleFacmacMPE': ('.mpe', 'SimpleFacmacMPE'),
    'SimpleFacmacMPE3a': ('.mpe', 'SimpleFacmacMPE3a'),
    'SimpleFacmacMPE6a': ('.mpe', 'SimpleFacmacMPE6a'),
    'SimpleFacmacMPE9a': ('.mpe', 'SimpleFacmacMPE9a'),
    'SMAX': ('.smax', 'SMAX'),
    'HeuristicEnemySMAX': ('.smax', 'HeuristicEnemySMAX'),
    'LearnedPolicyEnemySMAX': ('.smax', 'LearnedPolicyEnemySMAX'),
    'SwitchRiddle': ('.switch_riddle', 'SwitchRiddle'),
    'Overcooked': ('.overcooked', 'Overcooked'),
    'overcooked_layouts': ('.overcooked', 'overcooked_layouts'),
    'OvercookedV2': ('.overcooked_v2', 'OvercookedV2'),
    'overcooked_v2_layouts': ('.overcooked_v2', 'overcooked_v2_layouts'),
    'Ant': ('.mabrax', 'Ant'),
    'Humanoid': ('.mabrax', 'Humanoid'),
    'Hopper': ('.mabrax', 'Hopper'),
    'Walker2d': ('.mabrax', 'Walker2d'),
    'HalfCheetah': ('.mabrax', 'HalfCheetah'),
    'Hanabi': ('.hanabi', 'Hanabi'),
    'InTheGrid': ('.storm', 'InTheGrid'),
    'InTheGrid_2p': ('.storm', 'InTheGrid_2p'),
    'InTheMatrix': ('.storm', 'InTheMatrix'),
    'CoinGame': ('.coin_game', 'CoinGame'),
    'JaxNav': ('.jaxnav', 'JaxNav'),
    'CageEnv': ('.cage', 'CageEnv'),
    'HeuristicRedCAGE': ('.cage', 'HeuristicRedCAGE'),
    'HeuristicBLineCAGE': ('.cage', 'HeuristicBLineCAGE'),
    'HeuristicMeanderCAGE': ('.cage', 'HeuristicMeanderCAGE'),
}

def __getattr__(name):
    if name in _IMPORT_MAP:
        module_name, attr_name = _IMPORT_MAP[name]
        import importlib
        module = importlib.import_module(module_name, __package__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
