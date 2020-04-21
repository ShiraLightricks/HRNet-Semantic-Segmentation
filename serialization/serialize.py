import argparse
import os
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from serialization.pytorch_converter import convert
from serialization.utils import create_preprocess_dict, compress_and_save

from lib.config import config
from lib.config import update_config
from lib.models.seg_hrnet import get_seg_model
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def create_model(yaml_path):
    args = parse_args(yaml_path)

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'serialization')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = get_seg_model(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth')
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def _convert_upsample(builder, node, graph, err):
    if 'scales' in node.attrs:
        scales = node.attrs['scales']
    elif len(node.input_tensors):
        scales = node.input_tensors[node.inputs[1]]
    else:
        # HACK: Manual scales
        # PROVIDE MANUAL SCALE HERE
        scales = [1, 1, 0.5, 0.5]

    scale_h = scales[2]
    scale_w = scales[3]
    input_shape = graph.shape_dict[node.inputs[0]]
    target_height = int(input_shape[-2] * scale_h)
    target_width = int(input_shape[-1] * scale_w)

    builder.add_resize_bilinear(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        target_height=target_height,
        target_width=target_width,
        mode='UPSAMPLE_MODE'
    )


Y_PATH = "/experiments/cityscapes/seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml"


def serialize_hrnet(yaml_path=Y_PATH, models_path = "/cnvrg/output/"):
    im_size = (512, 1024)
    batch_size = 1
    num_channels = 3
    model = create_model(yaml_path)
    model_new_name = config.MODEL.NAME + "_for_coreML_" + str(im_size[0]) + "x" + str(im_size[1])
    sample_input_tensor = torch.rand(batch_size, num_channels, im_size[0], im_size[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_input_tensor = sample_input_tensor.to(device)

    onnx_export_path = models_path + model_new_name

    torch.onnx.export(model, sample_input_tensor, onnx_export_path, opset_version=11)

    classes_list = 19 * ["label"]
    mlmodel = convert(onnx_export_path, minimum_ios_deployment_target='13')
    pd = create_preprocess_dict(divisible_by=1, resize_strategy=None, side_length=None, output_classes=classes_list)

    new_name = "seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484" #TODO:dynamic

    compress_and_save(mlmodel, save_path="/cnvrg/output/", model_name=new_name, preprocess_dict=pd)

