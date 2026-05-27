import os

# 始终以本文件所在目录为 adaptive_bitrate_streaming 包根（与 cwd、run/ 下执行无关）
_ABR_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))


class Config:
    _base_dir = _ABR_PACKAGE_ROOT + os.sep
    baseline_model_paths = {
        'genet': _base_dir + 'data/all_models/genet/nn_model_ep_9900.ckpt',
        'udr_1': _base_dir + 'data/all_models/udr_1/nn_model_ep_57600.ckpt',
        'udr_2': _base_dir + 'data/all_models/udr_2/nn_model_ep_52400.ckpt',
        'udr_3': _base_dir + 'data/all_models/udr_3/nn_model_ep_58000.ckpt',
        'udr_real': _base_dir + 'data/all_models/udr_real/nn_model_ep_49000.ckpt',
    }
    
    trace_dirs = {
        'fcc-train': _base_dir + 'data/traces/train/fcc-train/',
        'fcc-valid': _base_dir + 'data/traces/valid/fcc-valid/',
        'fcc-test': _base_dir + 'data/traces/test/fcc-test/',
        'hsr-test': _base_dir + 'data/traces/test/hsr-test/',
        'Norway3G-test': _base_dir + 'data/traces/test/Norway3G-test/',
        'SolisWiFi-test': _base_dir + 'data/traces/test/SolisWiFi-test/',
        'SolisWiFi-train': _base_dir + 'data/traces/train/SolisWiFi-train/',
        'Lab-test': _base_dir + 'data/traces/test/Lab-test/',
        'Ghent-test': _base_dir + 'data/traces/test/Ghent-test/',
        'fcc_hsdpa_cooked_test_traces': _base_dir + 'data/traces/test/fcc_hsdpa_cooked_test_traces/',
        'fcc_hsdpa_cooked_traces': _base_dir + 'data/traces/train/fcc_hsdpa_cooked_traces/',
        'fcc_hsdpa_test_traces': _base_dir + 'data/traces/test/fcc_hsdpa_test_traces/',
        'fcc16-test': _base_dir + 'data/traces/test/fcc16-test/',
        'fcc16-train': _base_dir + 'data/traces/train/fcc16-train/',
        'fcc18-test': _base_dir + 'data/traces/test/fcc18-test/',
        'fcc18-train': _base_dir + 'data/traces/train/fcc18-train/',
        'Puffer22-test': _base_dir + 'data/traces/test/Puffer22-test/',
        'Puffer22-train': _base_dir + 'data/traces/train/Puffer22-train/',
        'Puffer21-test': _base_dir + 'data/traces/test/Puffer21-test/',
        'Puffer21-train': _base_dir + 'data/traces/train/Puffer21-train/',
        'Oboe-test': _base_dir + 'data/traces/test/Oboe-test/',
        'Oboe-train': _base_dir + 'data/traces/train/Oboe-train/',
    }

    video_size_dirs = {
        'video1': _base_dir + 'data/videos/video1_sizes/',
        'video2': _base_dir + 'data/videos/video2_sizes/',
        'video3': _base_dir + 'data/videos/video3_sizes/',
    }

    artifacts_dir = _base_dir + 'artifacts/'
    results_dir = artifacts_dir + 'results/'
    exp_pools_dir = artifacts_dir + 'exp_pools/'

    # plm special
    plm_types = ['gpt2', 'llama', 'llava', 't5-lm', 'opt', 'mistral', 'qwen']
    plm_sizes = ['xxs', 'xs', 'small', 'base', 'large', 'xl', 'xxl', '2_7b']  # note that the actual size of plm is dependent on the type of plm. 
                                                         # for example, for llama, 'base' is 7b, while for gpt2, 'base' is 340M. you can specify it yourself.
    plm_dir = os.path.join(os.path.dirname(_ABR_PACKAGE_ROOT), 'downloaded_plms')
    plm_ft_dir = _base_dir + 'data/ft_plms'
    plm_embed_sizes = {
        'gpt2': {
            'base': 1024,
            'small': 768,
            'large': 1280,
            'xl': 1600,
        },
        'llama': {
            'base': 2048,
            'large': 3072,
            'xl': 4096,
            '2_7b': 4096
        },
        't5-lm': {
            'base': 768,
            'small': 512,
            'large': 4096,
            'xl': 2048,
        },
        'llava': {
            'base': 4096,
        },
        'mistral': {
            'base': 4096,
        },
        'opt': {
            'large': 5120,
            'base': 4096,
            'small': 2560,
            'xs': 2048,
            'xxs': 512,
        },
        'qwen': {
            'large': 2560,
        },
    }
    plm_layer_sizes = {
        'gpt2': {
            'base': 24,
            'small': 12,
            'large': 36,
            'xl': 48
        },
        'llama': {
            'base': 32,
            'large': 32,
            'xl': 32,
            '2_7b': 32
        },
        't5-lm': { 
            'base': 12,
            'small': 6,
            'large': 24,
            'xl': 24
        },
        'llava': {
            'base': 32,
        },
        'mistral': {
            'base': 32,
        },
        'opt': {
            'large': 40,
            'base': 32,
            'small': 32,
            'xs': 32,
            'xxs': 16,
        },
        'qwen': {
            'large': 32,
        },
    }


cfg = Config()
