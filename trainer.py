import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import random
import os
import numpy as np
import gc
import json
import time
# from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown
import threading
def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)

# flag = True
# def monitor_gpu_energy():
#     """
#     监控 GPU 的总能耗。
    
#     :param duration: 监控的总时长（秒）。
#     :param interval: 每次采样的时间间隔（秒）。
#     :return: 总能耗（焦耳）。
#     """
#     global flag
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)  # 假设使用第一个 GPU

#     total_energy = 0.0  # 总能耗，单位为焦耳

#     try:
#         while flag:
#             power_usage = nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
#             total_energy += power_usage * 0.5  # 累积能耗
#             time.sleep(0.5)  # 等待下一个采样
#         logging.info(f"Total energy consumption: {total_energy / 50} J")
#     finally:
#         nvmlShutdown()



def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    # if torch.cuda.is_available():
    #     current_device = torch.cuda.current_device()
    #     device_name = torch.cuda.get_device_name(current_device)
    #     print(f"Current CUDA device: {current_device}")
    #     print(f"Device name: {device_name}")
    if args["model_name"] == "miro":
        if args["alg"] == "heuristic":
            logfilename = f"logs/{args['model_name']}/{args['dataset']}/{args['n_trials']}/{args['alg']}{args['rate_flag']}{args['dynamic']}/{args['budget']}{args['precent']}/{args['swap_policy']}/{args['seed']}/{args['dvs_config'][1]}/log_server"
        else:
            logfilename = f"logs/{args['model_name']}/{args['dataset']}/{args['n_trials']}/{args['alg']}{args['rate_flag']}{args['dynamic']}/{args['budget']}{args['strategy']}/{args['swap_policy']}/{args['seed']}/{args['dvs_config'][1]}/log_server"
        os.makedirs(logfilename, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[
                logging.FileHandler(filename=logfilename + ".log", mode='w'),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["prefix"],
            args["seed"],
            args["convnet_type"],
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[
                logging.FileHandler(filename=logfilename + ".log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
    # if args["edge"] == True:
    #     thread = threading.Thread(target=monitor_gpu_energy)
    #     thread.start()

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args["swap_ratio"],
        args["swap_policy"]
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_keys_sorted = sorted(nme_keys)
            nme_values = [nme_accy["grouped"][key] for key in nme_keys_sorted]
            nme_matrix.append(nme_values)


            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            if args["dataset"] != 'dvs128swap' and args["dataset"] != "urbansoundswap" and args["dataset"] != "dsadsswap":
                cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            if args["dataset"] != 'dvs128swap' and args["dataset"] != "urbansoundswap" and args["dataset"] != "dsadsswap":
                logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
    if (args["model_name"] == "miro"):
        with open(os.path.join(model.return_path, 'saved_config.json'), 'w') as f:
            json.dump(model.saved_config, f)
    global flag
    flag = False
    # if len(cnn_matrix)>0:
    #     np_acctable = np.zeros([task + 1, task + 1])
    #     for idxx, line in enumerate(cnn_matrix):
    #         idxy = len(line)
    #         np_acctable[idxx, :idxy] = np.array(line)
    #     np_acctable = np_acctable.T
    #     forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
    #     print('Accuracy Matrix (CNN):')
    #     print(np_acctable)
    #     print('Forgetting (CNN):', forgetting)
    #     logging.info('Forgetting (CNN): {}'.format(forgetting))
    # if len(nme_matrix)>0:
    #     np_acctable = np.zeros([task + 1, task + 1])
    #     for idxx, line in enumerate(nme_matrix):
    #         idxy = len(line)
    #         np_acctable[idxx, :idxy] = np.array(line)
    #     np_acctable = np_acctable.T
    #     forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
    #     print('Accuracy Matrix (NME):')
    #     print(np_acctable)
    #     print('Forgetting (NME):', forgetting)
    #     logging.info('Forgetting (NME):', forgetting)
    gc.collect()

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    # NumPy random seed
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
