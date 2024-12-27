from config import cfg

def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']

    cfg['device'] = 'cuda'
    cfg['batch_size'] = 8
    cfg['step_period'] = 1
    cfg['num_steps'] = 100
    cfg['print_step'] = 100
    cfg['eval_period'] = 200
    # cfg['num_epochs'] = 1
    cfg['collate_mode'] = 'dict'

    cfg['compressor'] = {}
    cfg['compressor']['seq_len'] = 8

    cfg['model'] = {}
    cfg['model']['model_name'] = cfg['model_name']
    cfg['model']['linear'] = {}
    cfg['model']['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
    cfg['model']['cnn'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['model']['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}
    cfg['model']['gpt1'] = {'n_head': 4, 'n_layer': 4, 'drop_out': 0, 'vocab_size': 257, 'n_embd': 64,
                            'block_size': 828, 'device': 'mps'}
    cfg['model']['trace'] = {'vocab_size': 256, 'vocab_dim': 64, 'hidden_dim': 256, 'n_layers': 1, 'ffn_dim': 4096,
                            'n_heads': 1, 'feature_type': 'sqr', 'compute_type': 'iter'}

    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'Adam'
    cfg[tag]['optimizer']['lr'] = 1e-3
    cfg[tag]['optimizer']['momentum'] = 0.9
    cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
    cfg[tag]['optimizer']['weight_decay'] = 0.0
    cfg[tag]['optimizer']['nesterov'] = True
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = 'CosineAnnealingLR'
    return
