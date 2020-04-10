import os
# import torch
# import torchvision
from collections import OrderedDict


def get_initial_checkpoint(config):
    checkpoint_dir = os.path.join(config.TRAIN_DIR+config.RECIPE, 'checkpoints')
    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir)
                   if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
    if checkpoints:
        return os.path.join(checkpoint_dir, list(sorted(checkpoints))[-1])
    return None


def load_checkpoint(config, model, checkpoint, device_name='cuda'):
    print('load checkpoint from', checkpoint)
    # checkpoint = torch.load(checkpoint)
    checkpoint = torch.load(checkpoint, map_location=torch.device(device_name))
    # state_dict = checkpoint['state_dict']

    # added
    if config.PARALLEL:
        state_dict_old = checkpoint['state_dict']
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            if key.startswith('module.'):
                state_dict[key] = state_dict_old[key]
            else:
                state_dict['module.' + key] = state_dict_old[key]

    else:
        state_dict_old = checkpoint['state_dict']
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            if key.startswith('module.'):
                state_dict[key[7:]] = state_dict_old[key]
            else:
                state_dict[key] = state_dict_old[key]

    model.load_state_dict(state_dict)

    last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else -1
    score = checkpoint['score'] if 'score' in checkpoint else -1
    loss = checkpoint['loss'] if 'loss' in checkpoint else float('inf')

    return last_epoch, score, loss


def save_checkpoint(config, model, epoch, score, loss, weights_dict=None):
    checkpoint_dir = os.path.join(config.TRAIN_DIR+config.RECIPE, 'checkpoints')

    checkpoint_path = os.path.join(checkpoint_dir, 'epoch_%04d_score%.4f_loss%.4f.pth' % (epoch, score, loss))

    if weights_dict is None:
        weights_dict = {
            'state_dict': model.state_dict(),
            'epoch' : epoch,
            'score': score,
            'loss': loss
        }
    torch.save(weights_dict, checkpoint_path)


def load_checkpoint_legacy(config, model, checkpoint):
    print('load checkpoint from', checkpoint)
    model = torch.load(checkpoint)


def save_checkpoint_legacy(config, model, epoch, score, loss, weights_dict=None):
    checkpoint_dir = os.path.join(config.TRAIN_DIR+config.RECIPE, 'checkpoints')

    checkpoint_path = os.path.join(checkpoint_dir, 'epoch_%04d_score%.4f_loss%.4f.pth' % (epoch, score, loss))
    checkpoint_pt_path = os.path.join(checkpoint_dir, 'epoch_%04d_score%.4f_loss%.4f.pt' % (epoch, score, loss))

    torch.save(model, checkpoint_path)

    model = model.to('cpu')
    example = torch.rand(4, 3, config.DATA.IMG_W, config.DATA.IMG_H).to('cpu')
    traced_script_module = torch.jit.trace(model, example)
    # traced_script_module.save(checkpoint_pt_path)
    torch.jit.save(traced_script_module, checkpoint_pt_path)
    model = model.to('cuda')