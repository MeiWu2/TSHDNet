import argparse
import time
import torch
import numpy as np
torch.set_num_threads(1)
import pickle
import yaml
import setproctitle
from save import set_config,load_model,EarlyStopping,data_reshaper
from utils import load_dataset,load_adj
from model import TSHDnet
from trainer import trainer

def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='PEMS03', help='Dataset name.')
    # parser.add_argument('--dataset', type=str, default='PEMS07', help='Dataset name.')
    parser.add_argument('--dataset', type=str, default='PEMS04', help='Dataset name.')
    # parser.add_argument('--dataset', type=str, default='PEMS08', help='Dataset name.')
    args = parser.parse_args(args = [])

    config_path = "/content/drive/MyDrive/configs/" + args.dataset + ".yaml"

    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    data_dir        = config['data_args']['data_dir']
    dataset_name    = config['data_args']['data_dir'].split("/")[-1]

    device          = torch.device(config['start_up']['device'])
    save_path       = '/content/drive/MyDrive/PEMS08_dump/' + config['start_up']['model_name'] + "_" + dataset_name + ".pt"             # the best model
    save_path_resume= '/content/drive/MyDrive/PEMS08_dump/' + config['start_up']['model_name'] + "_" + dataset_name + "_resume.pt"      # the resume model
    load_pkl        = config['start_up']['load_pkl']
    model_name      = config['start_up']['model_name']

    model_name      = config['start_up']['model_name']
    setproctitle.setproctitle("{0}.{1}@S22".format(model_name, dataset_name))

# ========================== load dataset, adjacent matrix, node embeddings ====================== #
    if load_pkl:
        t1   = time.time()
        dataloader  = pickle.load(open('/content/drive/MyDrive/PEMS08_dump/' + dataset_name + '.pkl', 'rb'))
        t2  = time.time()
        print("Load dataset: {:.2f}s...".format(t2-t1))
    else:
        t1   = time.time()
        batch_size  = 32
        dataloader  = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)
        pickle.dump(dataloader, open('/content/drive/MyDrive/PEMS08_dump/' + dataset_name + '.pkl', 'wb'))
        t2  = time.time()
        print("Load dataset: {:.2f}s...".format(t2-t1))
    scaler          = dataloader['scaler']

    _min = pickle.load(open("/content/drive/MyDrive/data/PEMS04/min.pkl".format(dataset_name), 'rb'))
    _max = pickle.load(open("/content/drive/MyDrive/data/PEMS04/max.pkl".format(dataset_name), 'rb'))

    t1   = time.time()
    adj_mx, adj_ori = load_adj('/content/drive/MyDrive/data/PEMS04/adj_PEMS04.pkl', config['data_args']['adj_type'])
    t2  = time.time()
    print("Load adjacent matrix: {:.2f}s...".format(t2-t1))


# ================================ Hyper Parameters ================================= #
    # model parameters
    model_args  = config['model_args']
    model_args['device']        = device
    model_args['num_nodes']     = adj_mx[0].shape[0]
    model_args['adjs']          = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori']      = torch.tensor(adj_ori).to(device)
    model_args['dataset']       = dataset_name
    model_args['alpha']         = torch.nn.Parameter(torch.tensor(0.5))
    model_args['dropout']       = 0.1
    # training strategy parametes
    optim_args                  = config['optim_args']
    optim_args['cl_steps']      = optim_args['cl_epochs'] * len(dataloader['train_loader'])
    optim_args['warm_steps']    = optim_args['warm_epochs'] * len(dataloader['train_loader'])
# ============================= Model and Trainer ============================= #
    # log

    # init the model
    model   = TSHDnet(**model_args).to(device)

    # get a trainer
    engine  = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    # begin training:
    train_time  = []    # training time
    val_time    = []    # validate time

    print("Whole trainining iteration is " + str(len(dataloader['train_loader'])))

    # training init: resume model & load parameters
    mode = config['start_up']['mode']
    assert mode in ['test', 'resume', 'scratch']
    resume_epoch = 0
    if mode == 'test':
        model = load_model(model, save_path)        # resume best
    else:
        if mode == 'resume':
            resume_epoch = config['start_up']['resume_epoch']
            model = load_model(model, save_path_resume)
        else:       # scratch
            resume_epoch = 0

    batch_num   = resume_epoch * len(dataloader['train_loader'])     # batch number (maybe used in schedule sampling)

    engine.set_resume_lr_and_cl(resume_epoch, batch_num)
# =============================================================== Training ================================================================= #
    if mode != 'test':
        for epoch in range(resume_epoch + 1, 5):
            # train a epoch
            time_train_start    = time.time()

            current_learning_rate = engine.optimizer.param_groups[0]['lr']
            train_loss = []
            train_mape = []
            train_rmse = []
            dataloader['train_loader'].shuffle()    # traing data shuffle when starting a new epoch.
            for itera, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx          = data_reshaper(x, device)
                trainy          = data_reshaper(y, device)
                mae, mape, rmse = engine.train(trainx, trainy, batch_num=batch_num, _max=_max, _min=_min)
                print("{0}: {1}".format(itera, mae), end='\r')
                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1
            time_train_end      = time.time()
            train_time.append(time_train_end - time_train_start)

            current_learning_rate = engine.optimizer.param_groups[0]['lr']

            if engine.if_lr_scheduler:
                engine.lr_scheduler.step()
            # record history loss
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)
# =============================================================== Validation ================================================================= #
            time_val_start      = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse, = engine.eval(device, dataloader, model_name, _max=_max, _min=_min)
            time_val_end        = time.time()
            val_time.append(time_val_end - time_val_start)

            curr_time   = str(time.strftime("%d-%H-%M", time.localtime()))
            log = 'Current Time: ' + curr_time + ' | Epoch: {:03d} | Train_Loss: {:.4f} | Train_MAPE: {:.4f} | Train_RMSE: {:.4f} | Valid_Loss: {:.4f} | Valid_RMSE: {:.4f} | Valid_MAPE: {:.4f} '
            print(log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_rmse, mvalid_mape, current_learning_rate,model_args['alpha'].item()))
            early_stopping(mvalid_loss, engine.model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
# =============================================================== Test ================================================================= #
            engine.test(model, save_path_resume, device, dataloader, scaler, model_name, _max=_max, _min=_min, loss=engine.loss, dataset_name=dataset_name)

        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs/epoch".format(np.mean(val_time)))
    else:
        engine.test(model, save_path_resume, device, dataloader, scaler, model_name, save=False, _max=_max, _min=_min, loss=engine.loss, dataset_name=dataset_name)

if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end   = time.time()
    print("Total time spent: {0}".format(t_end - t_start))