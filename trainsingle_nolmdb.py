import os
import cv2
import torch
import torch.nn as nn
import yaml
import random
import argparse
import time
import numpy as np
import torch.backends.cudnn as cudnn

cudnn.benchmark = False
cudnn.deterministic = True
import torch.distributed as dist
from tqdm import tqdm
from dataload.newloader import CreateDataset, alignCollate
from utils.tools import loadchkeyfile, create_module, makedir, averager, save_checkpoint, randomChoice, tensor2img, \
    get_no_weight_decay_param, fix_param
from utils.logger import Logger
from models.loss.CTCloss import CTCLoss
from models.BuildModel import crnnModel, SVTRModel
from utils.label2tensor import strLabelConverter
from models.backbones.repvggblock import repvgg_model_convert
from models.optimizer.optimizer import adjust_learning_rate_step, adjust_learning_rate_poly, \
    adjust_learning_rate_warm, reset_base_lr, adjust_learning_rate_cos


### 设置随机种子
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def ValProgram(model, val_loader, criterion, config):
    model.eval()
    n_correct = 0
    data_num = 0
    ctc_loss_record = averager()
    bar = tqdm(total=len(val_loader))
    converter = strLabelConverter(config['train']['alphabet'])
    rightInferWord = []
    rightImg = []
    for batch_idx, (imgsCPU, intText, intLength, Text) in enumerate(val_loader):
        bar.update(1)
        imgs = imgsCPU.cuda()
        with torch.no_grad():
            preds_class, preds_map, stn_x = model(imgs)
        preds_size = torch.IntTensor([preds_class.size(0)] * preds_class.size(1))
        ctcloss = criterion(preds_class, intText, preds_size, intLength) / preds_class.size(1)
        ctc_loss_record.add(ctcloss)

        _, preds_class = preds_class.max(2)
        preds_class = preds_class.transpose(0, 1).contiguous().view(-1)
        sim_preds_char = converter.decode(preds_class.data, preds_size.data, raw=False)
        data_num += imgs.shape[0]
        if isinstance(sim_preds_char, str):
            sim_preds_char = [sim_preds_char]
        for pred, target, imgCPU in zip(sim_preds_char, Text, imgsCPU):
            if pred == target:
                n_correct += 1
                if n_correct < 1000:
                    rightInferWord.append([pred, target])
                    rightImg.append(tensor2img(imgCPU))

    wordAcc = n_correct / data_num
    val_loss = ctc_loss_record.val()
    return val_loss, wordAcc, rightInferWord, rightImg, n_correct, data_num


def Val(model, val_loader, criterion, config):
    acc_dict = {}
    acc_list = []
    val_loss, wordAcc, rightWords, rightImg, n_correct, data_num = ValProgram(model, val_loader, criterion, config)
    showTexts, showImgs = randomChoice(rightWords, rightImg)
    print('****************************show error recognize word*******************************')
    for i in range(len(showTexts)):
        print("===>pre:{}-------------gt:{}".format(showTexts[i][0], showTexts[i][1]))
        cv2.imwrite('./show/' + "pre_{}_gt_{}.jpg".format(showTexts[i][0], showTexts[i][1]), showImgs[i])
    print('****************************show error recognize word*******************************')
    print('#######################val_loss:{}#############'.format(val_loss))
    print('n_correct:{} data_num:{}  wordAcc:{}'.format(n_correct, data_num,wordAcc))

    acc_dict['avg_acc'] = wordAcc
    acc_list.append(wordAcc)
    return acc_dict, acc_list


def trainProgram(opt):
    #####
    config = yaml.load(open(opt.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    # 创建文件夹
    makedir('./show')
    makedir(config['train']['save_dir'])
    makedir(os.path.join(config['train']['save_dir'], 'log'))
    makedir(os.path.join(config['train']['save_dir'], 'modelFile'))

    config['train']['alphabet'] = loadchkeyfile(config['train']['key_file'])
    config['train']['nclass'] = len(config['train']['alphabet']) + 1

    train_dataset = CreateDataset(config, 'train')
    if config['train']['isVal']:
        val_dataset = CreateDataset(config, 'val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        collate_fn=alignCollate(config, 'train'),
        drop_last=True,
        pin_memory=True)
    if config['train']['isVal']:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['train']['batch_size'] * 2,
            num_workers=config['train']['num_workers'],
            collate_fn=alignCollate(config, 'val'),
            shuffle=False,
            drop_last=False,
            pin_memory=True)
    assert config['train']['algorithm'] in ['CRNN', 'SVTR']
    if config['train']['algorithm'] == 'CRNN':
        model = crnnModel(config).cuda()
    elif config['train']['algorithm'] == 'SVTR':
        model = SVTRModel(config).cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    criterion = CTCLoss().cuda()
    optimizer = create_module(config['model']['optim_function'])(config, fix_param(model, config))

    # if dist.get_rank() == 0:
    #     print(model)

    start_epoch = 0
    # 加载预训练模型
    if config['train']['restore']:
        assert os.path.isfile(config['train']['resume_file']), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(config['train']['resume_file'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Resuming from checkpoint.')
        log_write = Logger(os.path.join(config['train']['save_dir'], 'log', 'log.txt'),
                           title=config['model']['backbone'], resume=True)
        log_write_val = Logger(os.path.join(config['train']['save_dir'], 'log', 'logval.txt'),
                               title=config['model']['backbone'], resume=True)
    else:

        print('Training from scratch.')
        log_write = Logger(os.path.join(config['train']['save_dir'], 'log', 'log.txt'),
                           title=config['model']['backbone'])
        log_write.set_names(['epoch', 'Total loss', 'CTC loss', 'L2 loss'])

        log_write_val = Logger(os.path.join(config['train']['save_dir'], 'log', 'logval.txt'),
                               title=config['model']['backbone'])
        log_write_val.set_names(['train_step'] + config['train']['val_lmdb_file'] + ['avg_acc'])


    # 自己添加log.txt 去除复杂的logtxt
    trainlog = os.path.join(config['train']['save_dir'],  'log.txt')
    vallog =  os.path.join(config['train']['save_dir'],  'logval.txt')
    n_epoch = config['train']['epochs']
    for epoch in range(start_epoch, n_epoch):
        if epoch < config['train']['warmepochs']:
            adjust_learning_rate_warm(config, optimizer, epoch)
        else:
            assert config['optimizer']['optim_type'] in ['poly', 'step', 'cos']
            if config['optimizer']['optim_type'] == 'poly':
                adjust_learning_rate_poly(config, optimizer, epoch)
            elif config['optimizer']['optim_type'] == 'step':
                adjust_learning_rate_step(config, optimizer, epoch)
            elif config['optimizer']['optim_type'] == 'cos':
                adjust_learning_rate_cos(config, optimizer, epoch)
            for lr_i in range(len(optimizer.param_groups)):
                if optimizer.param_groups[lr_i]['lr'] < config['optimizer']['min_lr']:
                    optimizer.param_groups[lr_i]['lr'] = config['optimizer']['min_lr']

        total_loss_record = averager()
        ctc_loss_record = averager()
        l2_loss_record = averager()
        batch_time_record = averager()

        batch_start = time.time()
        for batch_idx, (imgs, intText, intLength, _) in enumerate(train_loader):
            model.train()
            train_step = epoch * len(train_loader) + batch_idx
            imgs = imgs.cuda()
            preds_class, preds_map = model(imgs)
            preds_size = torch.IntTensor([preds_class.size(0)] * preds_class.size(1))

            ### cal ctcloss
            ctcloss = criterion(preds_class, intText, preds_size, intLength) / preds_class.size(1)
            ctcloss = ctcloss.cuda()
            ### cal l2loss
            if config['train']['usel2loss']:
                l2_loss = torch.tensor(0.).cuda()
                for module in model.modules():
                    if hasattr(module, 'get_custom_L2'):
                        l2_loss += 10e-4 * 0.5 * module.get_custom_L2()

                total_loss = ctcloss + l2_loss
            else:
                l2_loss = torch.tensor(0.)
                total_loss = ctcloss
            ctc_loss_record.add(ctcloss)
            l2_loss_record.add(l2_loss)
            total_loss_record.add(total_loss)
            model.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            batch_end = time.time()
            batch_time_record.add(torch.tensor(batch_end - batch_start))
            # 达到迭代步骤进行展示
            if train_step % config['train']['show_iter'] == 0:
                print('[%d/%d][step:%d]  Totalloss:%.3f CTCloss: %.3f L2loss: %.3f lr: %.6f batchTime: %.2f s' % (
                    epoch, config['train']['epochs'], train_step, total_loss_record.val(), ctc_loss_record.val(),
                    l2_loss_record.val(), optimizer.param_groups[0]['lr'], batch_time_record.val()))
            # 达到验证步骤进行验证并存储
            if config['train']['isVal'] and (
                    train_step % config['train']['val_step'] == 0 or train_step == n_epoch * len(
                train_loader) - 1) and train_step != 0:
                acc_dict, acc_list = Val(model, val_loader, criterion, config)
                with open(vallog, "a") as f:
                    f.write("epoch: {} trainstep: {} acc: {}\n".format(epoch, train_step, acc_list[0]))


                # log_write_val.append(train_step + acc_list)
                save_checkpoint({
                    'epoch': train_step, # 训练步长
                    'state_dict': model.state_dict(), # 模型参数
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(config['train']['save_dir'], 'modelFile'),
                    config['train']['algorithm'] + '-step-{}-wordAcc-{}-{}.pth'.format(train_step,
                                                                                       '%.4f' % acc_list[-1],
                                                                                       opt.tag))
                with open(trainlog, "a") as f:
                    f.write("epoch: {} trainstep: {} loss: {} CTCloss: {} l2loss: {}".format(
                        epoch, train_step, total_loss_record.val(), ctc_loss_record.val(), l2_loss_record.val()
                    ))
                #log_write.append([train_step, total_loss_record.val(), ctc_loss_record.val(), l2_loss_record.val()])
            batch_start = time.time()

        print('[%d/%d] epochTotalloss: %.3f epochCTCLoss: %.3f epochL2loss: %.3f lr: %.6f epochTime: %.2f h' % (
            epoch, config['train']['epochs'], total_loss_record.val(), ctc_loss_record.val(), l2_loss_record.val(),
            optimizer.param_groups[0]['lr'], batch_time_record.sum / 3600))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--config', type=str, default="/code/wujilong/code/OCR-TextRecog/config/mobilev3CRNN.yaml",
                        help='DDP parameter, do not modify')
    parser.add_argument('--tag', type=str, default="CHINESE", help='DDP parameter, do not modify')
    opt = parser.parse_args()
    trainProgram(opt)