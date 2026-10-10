"""
Configs of CA-QMIX and the baselines on the SMACv2 10-vs-10 scenarios.

    main_config, create_config = get_config(race='protoss', method='caqmix', seed=0)
"""
from easydict import EasyDict

RACES = ('protoss', 'terran', 'zerg')
METHODS = ('caqmix', 'qmix', 'vdn', 'qtran', 'qplex', 'qatten', 'collaq')
ABLATIONS = ('alpha_none', 'binary_gate', 'no_hard_mask', 'no_vis_input', 'shuffled_vis', 'delta_q_none')

N_AGENTS = 10
N_ACTIONS = 16

# Per-agent observation layout [move | enemy | ally | own]: start of the ally block, features per
# ally (without last actions) and own features; size of the global state; Qatten's own-state size.
RACE_INFO = {
    'protoss': dict(map_name='10gen_protoss', ally_start=94, ally_feats=9, own_feats=5, state_shape=310, own_state=8),
    'terran': dict(map_name='10gen_terran', ally_start=84, ally_feats=8, own_feats=4, state_shape=290, own_state=7),
    'zerg': dict(map_name='10gen_zerg', ally_start=84, ally_feats=8, own_feats=4, state_shape=290, own_state=7),
}


def get_config(
        race: str = 'protoss',
        method: str = 'caqmix',
        seed: int = 0,
        ablation: str = None,
        obs_last_action: bool = None,
        center_coop: bool = False,
        cuda: bool = True,
        collector_env_num: int = 8,
        evaluator_env_num: int = 8,
):
    """
    Arguments:
        - race: one of ``RACES``.
        - method: one of ``METHODS``.
        - ablation: CA-QMIX ablation, one of ``ABLATIONS``.
        - obs_last_action: include allies' last actions in the observation
          (default as in the paper: True for CA-QMIX, False for the baselines and the CA-QMIX ablations).
        - center_coop: subtract the action mean from Q_coop (CA-QMIX only).
    """
    assert race in RACES and method in METHODS, (race, method)
    assert ablation is None or (method == 'caqmix' and ablation in ABLATIONS), ablation
    info = RACE_INFO[race]
    if obs_last_action is None:
        obs_last_action = method == 'caqmix' and ablation is None
    n_allies = N_AGENTS - 1
    ally_feats = info['ally_feats'] + (N_ACTIONS if obs_last_action else 0)
    ally_end = info['ally_start'] + n_allies * ally_feats
    obs_shape = ally_end + info['own_feats']

    model = dict(
        agent_num=N_AGENTS,
        obs_shape=obs_shape,
        global_obs_shape=info['state_shape'],
        action_shape=N_ACTIONS,
        hidden_size_list=[64],
        lstm_type='gru',
        dueling=False,
    )
    learn = dict(
        update_per_collect=20,
        batch_size=32,
        learning_rate=0.0005,
        clip_value=5,
        double_q=True,
        target_update_theta=0.01,
        discount_factor=0.99,
        learner=dict(hook=dict(save_ckpt_after_iter=999999999)),  # keep only the best checkpoint (saved on eval)
    )
    if method == 'caqmix':
        model.update(
            self_feature_range=[ally_end, obs_shape],
            ally_feature_range=[info['ally_start'], ally_end],
            attention_size=32,
            mixer=True,
            dueling=True,
        )
        learn.update(lambda_delta_q=1.0)
        if center_coop:
            model.update(center_coop=True)
        if ablation == 'alpha_none':
            model.update(alpha=False)
        elif ablation == 'delta_q_none':
            learn.update(lambda_delta_q=0.0)
        elif ablation is not None:
            model.update(gate_variant={'binary_gate': 'binary'}.get(ablation, ablation))
    elif method in ('qmix', 'vdn'):
        model.update(mixer=method == 'qmix')
    elif method == 'qtran':
        model.update(embedding_size=64)
        learn.update(td_weight=1, opt_weight=0.01, nopt_min_weight=0.0001)
    elif method == 'qplex':
        model.update(mixer=True, mixing_embed_dim=32, num_kernel=4)
    elif method == 'qatten':
        model.update(mixer=True, agent_own_state_size=info['own_state'], n_attention_head=4, weighted_head=True)
    elif method == 'collaq':
        model.update(
            alone_obs_shape=obs_shape - n_allies * ally_feats,
            self_feature_range=[ally_end, obs_shape],
            ally_feature_range=[info['ally_start'], ally_end],
            attention=True,
            attention_size=32,
            # As run in the experiments: no mixing network on protoss, QMIX mixer on terran/zerg.
            mixer=race != 'protoss',
        )

    exp_name = f'smac2_{race}_{method}' + (f'_{ablation}' if ablation else '') + f'_seed{seed}'
    main_config = dict(
        exp_name=exp_name,
        env=dict(
            map_name=info['map_name'],
            difficulty='7',
            agent_num=N_AGENTS,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=32,
            obs_alone=method == 'collaq',
            obs_last_action=obs_last_action,
            manager=dict(shared_memory=False, reset_timeout=6000, retry_type='renew', max_retry=5),
        ),
        policy=dict(
            cuda=cuda,
            on_policy=False,
            multi_agent=True,
            model=model,
            learn=learn,
            collect=dict(n_episode=32, unroll_len=10, env_num=collector_env_num),
            eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=50000, eval_mode='envstep')),
            other=dict(
                eps=dict(type='linear', start=1.0, end=0.05, decay=50000),
                replay_buffer=dict(type='naive', replay_buffer_size=15000),
            ),
        ),
    )
    create_config = dict(
        env=dict(type='smac2', import_names=['dizoo.smac2.envs.smac2_env']),
        env_manager=dict(type='subprocess'),
        policy=dict(type='qmix' if method == 'vdn' else method),  # VDN = QMIX policy without mixer
        collector=dict(type='episode', get_train_sample=True),
        replay_buffer=dict(type='naive'),
    )
    return EasyDict(main_config), EasyDict(create_config)
