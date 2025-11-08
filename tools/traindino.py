import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import argparse
import logging
# import sys
#
# sys.path.append("/data2/gaoyupeng/LESPS-master/ChangeDINO/mmseg")
import os.path as osp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.analysis import get_model_complexity_info
from mmseg.registry import RUNNERS
# import pdb;
# pdb.set_trace()

# from GDINO.train import a
# a()
import torch
import torch.nn as nn


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


from mmengine.analysis import get_model_complexity_info

class LossWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, data_samples):
        return self.model.loss(inputs, data_samples)


def estimate_loss_complexity(model, input_shape):
    loss_wrapper = LossWrapper(model)

    # 创建模拟输入
    dummy_input = torch.rand(1, *input_shape)
    batch_img_metas = [{
        'img_shape': input_shape[1:],
        'ori_shape': input_shape[1:],
        'pad_shape': input_shape[1:],
        'scale_factor': 1.0,
    }]
    data_samples = [type('', (), {'metainfo': m})() for m in batch_img_metas]

    # 为数据样本添加必要的属性
    for sample in data_samples:
        sample.gt_sem_seg = torch.randint(0, model.num_classes, (1, *input_shape[2:]))

    # 准备输入字典
    inputs = dict(inputs=dummy_input, data_samples=data_samples)

    try:
        analysis_results = get_model_complexity_info(
            loss_wrapper,
            input_shape=None,
            inputs=inputs,
            show_table=False,
            show_arch=True
        )

        params = analysis_results['params']
        flops = analysis_results['flops']

        print(f'Loss function Parameters: {params}')
        print(f'Loss function FLOPs: {flops}')
    except Exception as e:
        print(f"Error in loss complexity analysis: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and scratchformermodels')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args
def main():
    args = parse_args()

    # 加载配置
    cfg = Config.fromfile(args.config)

    cfg.launcher = None
    if 'launcher' in cfg:
        cfg.pop('launcher')
    if 'dist_params' in cfg:
        cfg.pop('dist_params')
    cfg.gpu_ids = [0]
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 设置工作目录
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # 处理AMP训练
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # 设置恢复训练
    cfg.resume = args.resume

    # 构建runner
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()

if __name__ == '__main__':
    main()