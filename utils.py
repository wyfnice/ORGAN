import argparse
import os
import yaml
import shutil
import torch
import random
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from dataset import FSGDataset

def _denorm_to_unit_range(img_tensor):
    """Map tensors from [-1, 1] to [0, 1] for visualization/saving.
    Expects shape [..., C, H, W].
    """
    return (img_tensor.clamp(-1, 1) + 1) / 2

def make_result_folders(output_directory,remove_first=True):
    if remove_first:
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)

    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    log_directory = os.path.join(output_directory, 'logs')
    if not os.path.exists(log_directory):
        print("Creating directory: {}".format(log_directory))
        os.makedirs(log_directory)
    return checkpoint_directory, image_directory, log_directory


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if ((not callable(getattr(trainer, attr))
                    and not attr.startswith("__"))
                   and ('loss' in attr
                        or 'grad' in attr
                        or 'nwd' in attr
                        or 'accuracy' in attr
                        or 'obsv' in attr))]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def write_image(iterations, dir, im_ins, im_outs, format='jpeg', save_per_class=False, class_labels=None):
    B, K1, C, H, W = im_ins.size()
    B, K2, C, H, W = im_outs.size()
    file_name = os.path.join(dir, '%08d' % (iterations + 1) + '.' + format)
    image_tensor = torch.cat([im_ins, im_outs], dim=1)
    image_tensor = image_tensor.view(B*(K1+K2), C, H, W)
    image_grid = vutils.make_grid(image_tensor.data, nrow=K1+K2, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1, format=format)

    if save_per_class and class_labels is not None:
        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels)
        class_labels = class_labels.cpu().numpy()

        per_class_dir = os.path.join(dir, "per_class")
        os.makedirs(per_class_dir, exist_ok=True)

        for idx in range(B):
            label = class_labels[idx]
            class_dir = os.path.join(per_class_dir, f"merged_{label}")
            os.makedirs(class_dir, exist_ok=True)

            for k in range(K2):
                img_tensor = im_outs[idx, k]  # [C, H, W]
                img_tensor = _denorm_to_unit_range(img_tensor)
                img_file = os.path.join(class_dir, f"{iterations + 1:08d}_sample{idx}_k{k}.{format}")
                vutils.save_image(img_tensor, img_file)


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def get_loader(dataset, root, mode, n_sample, num_for_seen, batch_size, num_workers, shuffle, drop_last,
               new_size=None, height=28, width=28, crop=False, center_crop=False):

    assert dataset in ['flower', 'vggface', 'animal', 'our', 'colon', 'intestinal']

    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if center_crop:
        transform_list = [transforms.CenterCrop((height, width))] + \
                         transform_list if crop else transform_list
    else:
        transform_list = [transforms.RandomCrop((height, width))] + \
                         transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list \
        if new_size is not None else transform_list

    transform = transforms.Compose(transform_list)
    dataset = FSGDataset(root, mode, num_for_seen, n_sample, transform)

    loader = DataLoader(dataset,
                        batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last,
                        num_workers=num_workers)
    return loader


def get_loaders(conf):
    dataset = conf['dataset']
    root = conf['data_root']
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    train_loader = get_loader(
        dataset=dataset,
        root=root,
        mode='train',
        n_sample=conf['n_sample_train'],
        num_for_seen=conf['dis']['num_classes'],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True
    )
    test_loader = get_loader(
        dataset=dataset,
        root=root,
        mode='test',
        n_sample=conf['n_sample_test'],
        num_for_seen=conf['dis']['num_classes'],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False
    )
    return train_loader, test_loader


def unloader(img):
    img = (img + 1) / 2
    tf = transforms.Compose([
        transforms.ToPILImage()
    ])
    return tf(img)


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and
                  key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views)
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def batched_scatter(input, dim, index, src):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views)
    index = index.expand(expanse)
    return torch.scatter(input, dim, index, src)


def cal_para(model):
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params: %.3fM' % (trainable_num / 1e6))


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')