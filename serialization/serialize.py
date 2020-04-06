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
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger


# def parse_args():
#     parser = argparse.ArgumentParser(description='Train segmentation network')
#
#     parser.add_argument('--cfg',
#                         help='experiment configure file name',
#                         required=True,
#                         type=str)
#     parser.add_argument('opts',
#                         help="Modify config options using the command-line",
#                         default=None,
#                         nargs=argparse.REMAINDER)
#
#     args = parser.parse_args()
#     update_config(config, args)
#
#     return args

def parse_args(yaml_file):
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.cfg = yaml_file
    args = parser.parse_args()
    update_config(config, args)
    return args


def create_model(yaml_path):

    args = parse_args(yaml_path)

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'model serialization')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

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

    # gpus = list(config.GPUS)
    # model = nn.DataParallel(model, device_ids=gpus).cuda()

    return model

    # # prepare data
    # test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    # test_dataset = eval('datasets.' + config.DATASET.DATASET)(
    #     root=config.DATASET.ROOT,
    #     list_path=config.DATASET.TEST_SET,
    #     num_samples=None,
    #     num_classes=config.DATASET.NUM_CLASSES,
    #     multi_scale=False,
    #     flip=False,
    #     ignore_label=config.TRAIN.IGNORE_LABEL,
    #     base_size=config.TEST.BASE_SIZE,
    #     crop_size=test_size,
    #     downsample_rate=1)
    #
    # testloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=config.WORKERS,
    #     pin_memory=True)
    #
    # start = timeit.default_timer()
    # if 'val' in config.DATASET.TEST_SET:
    #     mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config,
    #                                                        test_dataset,
    #                                                        testloader,
    #                                                        model)
    #
    #     msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
    #         Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU,
    #                                                 pixel_acc, mean_acc)
    #     logging.info(msg)
    #     logging.info(IoU_array)
    # elif 'test' in config.DATASET.TEST_SET:
    #     test(config,
    #          test_dataset,
    #          testloader,
    #          model,
    #          sv_dir=final_output_dir)
    #
    # end = timeit.default_timer()
    # logger.info('Mins: %d' % np.int((end - start) / 60))
    # logger.info('Done')


if __name__ == '__main__':
    main()

# create onnox model
model_name = ""


# models_path = "/Users/shira/HRNet-Semantic-Segmentation/serialization/models/"
# model_new_name = model_name + "_for_coreML"
# model = torch.load(models_path + model_name)
# sample_input_tensor = ""

def serialize_hrnet(yaml_path, models_path):

    model, sample_input_tensor = create_model(yaml_path) #TODO returm sample_input_tensor
    model_new_name = model.name + "_for_coreML" #TODO get model name

    # coreML serialization
    torch.onnx.export(model, sample_input_tensor, models_path + model_new_name)
    mlmodel = convert(...)

    pd = create_preprocess_dict(...)
    compress_and_save(mlmodel, save_path="some path", model_name="my_model", preprocess_dict=pd)

# yaml_path = "/Users/shira/HRNet-Semantic-Segmentation/experiments/cityscapes/seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml"
# create_model(yaml_path)