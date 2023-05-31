import argparse
import json
import os.path
import platform

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataPreprocess import DataPreprocessor
from DataSet import DataSet
from models.model import Informer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=30)
    parser.add_argument('-c', '--cuda_device', type=str, default='0')
    parser.add_argument('-d', '--dataset', type=str, default='wht')
    parser.add_argument('-E', '--encoding_dimension', type=int, default=6)
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-G', '--use_cuda', action='store_true')
    parser.add_argument('-i', '--input_length', type=int, default=60)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-M', '--multiGPU', action='store_true')
    parser.add_argument('-n', '--normalize', type=str, default='std')
    parser.add_argument('-p', '--pred_length', type=int, default=24)
    parser.add_argument('-t', '--encoding_type', type=str, default='time_encoding')
    parser.add_argument('-S', '--stride', type=int, default=1)
    parser.add_argument('-s', '--save', type=str, default='save')
    parser.add_argument('-u', '--induce_length', type=int, default=10)
    parser.add_argument('--fudan',action='store_true')
    parser.add_argument('--nwpu',action='store_true')
    args = parser.parse_args()
    print(args)

    total_epoch = args.epoch
    input_length = args.input_length
    induce_length = args.induce_length
    pred_length = args.pred_length
    encoding_dimension = args.encoding_dimension
    encoding_type = args.encoding_type
    batch_size = args.batch_size
    use_cuda = args.use_cuda
    learning_rate = args.learning_rate
    normalize = args.normalize
    stride = args.stride
    save_path=args.save

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    arg_dict=vars(args)

    device = torch.device('cuda:' + str(args.cuda_device) if use_cuda else 'cpu')

    data_root = None
    if platform.system() == 'Windows':
        data_root = 'E:\\forecastdataset\\pkl'
    else:
        if args.fudan:
            data_root='/remote-home/liuwenbo/pycproj/forecastdata/pkl/'
        else:
            data_root = '/home/icpc/pycharmproj/forecast.dataset/pkl/'
    data_dir = None
    if args.dataset == 'wht':
        data_dir = os.path.join(data_root, 'wht.pkl')
    elif args.dataset == 'synthetic':
        data_dir = os.path.join(data_root, 'synthetic.pkl')
    else:
        print('invalid data')
        exit()

    data_preprocessor = DataPreprocessor(data_dir, input_length, pred_length, encoding_type=encoding_type,
                                         encoding_dimension=encoding_dimension, stride=stride)
    train_set = DataSet(data_preprocessor.load_train_set(), data_preprocessor.load_train_encoding_set(),
                        input_length, induce_length, pred_length)
    validate_set = DataSet(data_preprocessor.load_validate_set(), data_preprocessor.load_validate_encoding_set(),
                           input_length, induce_length, pred_length)
    test_set = DataSet(data_preprocessor.load_test_set(), data_preprocessor.load_test_encoding_set(),
                       input_length, induce_length, pred_length)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # for i, (enc_input, enc_encoding, dec_input, dec_encoding,ground_truth) in enumerate(train_loader):
    #     print(enc_input.shape,enc_encoding.shape,dec_input.shape,dec_encoding.shape,ground_truth.shape)
    #     exit()
    enc_in = data_preprocessor.load_enc_dimension()
    dec_in = data_preprocessor.load_dec_dimension()
    c_out = data_preprocessor.load_output_dimension()
    out_len = pred_length
    model = Informer(enc_in, dec_in, c_out, out_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    model = model.to(device)

    pbar_epoch = tqdm(total=total_epoch, ascii=True, dynamic_ncols=True)
    for epoch in range(total_epoch):
        model.train()
        total_iters = len(train_loader)
        pbar_iter = tqdm(total=total_iters, ascii=True, leave=False, dynamic_ncols=True)
        pbar_iter.set_description('training')
        for i, (enc_input, enc_encoding, dec_input, dec_encoding, ground_truth) in enumerate(train_loader):
            enc_input = enc_input.to(device)
            enc_encoding = enc_encoding.to(device)
            dec_input = dec_input.to(device)
            dec_encoding = dec_encoding.to(device)
            ground_truth = ground_truth.to(device)
            optimizer.zero_grad()
            pred = model(enc_input, enc_encoding, dec_input, dec_encoding)
            loss = criterion(pred, ground_truth)
            loss.backward()
            optimizer.step()
            pbar_iter.update(1)
            pbar_iter.set_postfix_str('loss: %.4f' % loss.item())
        pbar_iter.close()

        model.eval()
        total_iters = len(validate_loader)
        pbar_iter = tqdm(total=total_iters, ascii=True, leave=False, dynamic_ncols=True)
        pbar_iter.set_description('validating')
        prediction_list = []
        gt_list = []
        with torch.no_grad():
            for i, (enc_input, enc_encoding, dec_input, dec_encoding, ground_truth) in enumerate(validate_loader):
                # print(enc_input.shape,enc_encoding.shape,dec_input.shape,dec_encoding.shape)
                # exit()
                enc_input = enc_input.to(device)
                enc_encoding = enc_encoding.to(device)
                dec_input = dec_input.to(device)
                dec_encoding = dec_encoding.to(device)
                ground_truth = ground_truth.to(device)
                pred = model(enc_input, enc_encoding, dec_input, dec_encoding)
                prediction_list.append(pred)
                gt_list.append(ground_truth)
                pbar_iter.update(1)
            predictions = torch.cat(prediction_list, dim=0)
            ground_truths = torch.cat(gt_list, dim=0)
            validate_loss = criterion(predictions, ground_truths)
            pbar_epoch.set_postfix_str('loss: %.4f' % validate_loss.item())
            pbar_epoch.update()
        pbar_iter.close()
    pbar_epoch.close()

    model.eval()
    total_iters = len(test_loader)
    pbar_iter = tqdm(total=total_iters, ascii=True, dynamic_ncols=True)
    pbar_iter.set_description('testing')
    prediction_list = []
    gt_list = []
    with torch.no_grad():
        for i, (enc_input, enc_encoding, dec_input, dec_encoding, ground_truth) in enumerate(test_loader):
            enc_input = enc_input.to(device)
            enc_encoding = enc_encoding.to(device)
            dec_input = dec_input.to(device)
            dec_encoding = dec_encoding.to(device)
            ground_truth = ground_truth.to(device)
            pred = model(enc_input, enc_encoding, dec_input, dec_encoding)
            prediction_list.append(pred)
            gt_list.append(ground_truth)
            pbar_iter.update(1)
        pbar_iter.close()
        predictions = torch.cat(prediction_list, dim=0)
        ground_truths = torch.cat(gt_list, dim=0)
        test_loss = criterion(predictions, ground_truths)
        last_test_loss = criterion(predictions[:, -1, :], ground_truths[:, -1, :])
        print(predictions.shape, ground_truths.shape)
        print('\033[35mloss: %.4f\033[0m' % test_loss.item())
        print('\033[35mloss: %.4f\033[0m' % last_test_loss.item())
        result_dict=arg_dict
        result_dict['loss']=test_loss.item()
        f=open(os.path.join(save_path,'result.txt'),'w')
        print(json.dumps(result_dict,ensure_ascii=False),file=f)
        f.close()
