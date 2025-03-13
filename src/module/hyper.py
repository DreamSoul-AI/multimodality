from config import cfg


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']

    cfg['batch_size'] = 512
    cfg['num_chunks'] = 512
    cfg['seq_len'] = 64
    cfg['vocab_size'] = 256
    cfg['step_period'] = 1
    cfg['num_steps'] = 8000
    cfg['eval_period'] = 200
    cfg['eval'] = {}
    cfg['eval']['num_steps'] = -1
    cfg['num_epochs'] = 400
    cfg['collate_mode'] = 'dict'

    cfg['model'] = {}
    cfg['model']['model_name'] = cfg['model_name']
    cfg['model']['bootstrap'] = {'vocab_size': 256, 'emb_size': 16, 'length': 64, 'jump': 16, 'hdim1': 128,
                                 'hdim2': 256, 'n_layers': 2, 'bidirectional': True}
    cfg['model']['trace'] = {'vocab_size': 256, 'vocab_dim': 64, 'hidden_dim': 256, 'n_layers': 1, 'ffn_dim': 4096,
                             'n_heads': 1, 'feature_type': 'sqr', 'compute_type': 'iter'}

    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'SGD'
    cfg[tag]['optimizer']['lr'] = 1e-1
    cfg[tag]['optimizer']['momentum'] = 0.9
    cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
    cfg[tag]['optimizer']['weight_decay'] = 5e-4
    cfg[tag]['optimizer']['nesterov'] = True
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = 'CosineAnnealingLR'
    return
