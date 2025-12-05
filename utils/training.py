# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.gcl_dataset import GCLDataset
from models.utils.continual_model import ContinualModel

from utils import random_id
from utils.checkpoints import mammoth_load_checkpoint
from utils.loggers import *
from utils.status import ProgressBar
import time
import torchattacks
from robustbench.utils import load_model, clean_accuracy

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.

    Args:
        outputs: the output tensor
        dataset: the continual dataset
        k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
            dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


@torch.no_grad()
def evaluate_ensemble(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task
    """
    status = model.modelArr[0].training #model.net.training

    for c1 in range(np.shape(model.modelArr)[0]):
        #model.net.eval()
        model.modelArr[c1].eval()

    accs, accs_mask_classes = [], []
    n_classes = dataset.get_offsets()[1]
    expertIndex = 0
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > model.get_debug_iters():
                break
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs = model.modelArr[k](inputs) #model(inputs, k)
            else:
                outputs = model.modelArr[k](inputs) #model(inputs)

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        expertIndex = expertIndex +1

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    for c1 in range(np.shape(model.modelArr)[0]):
        #model.net.eval()
        model.modelArr[c1].train(status)
    #model.net.train(status)
    return accs, accs_mask_classes


@torch.no_grad()
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    n_classes = dataset.get_offsets()[1]
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > model.get_debug_iters():
                break
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                if model.NAME == 'siamesevit':
                    outputs = model.myPrediction(inputs, labels, k)
                else:
                    outputs = model(inputs)

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def evaluate_defense(model: ContinualModel, dataset: ContinualDataset, last=False,attackName="") -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []

    #atk = torchattacks.PGD(model.net, eps=8 / 255, alpha=2 / 255, steps=4)

    if attackName == 'FGSM':
        print("This is FGSM")
        atk = torchattacks.FGSM(model, eps=8 / 255)
    elif attackName == 'PGD':
        atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=False)
    elif attackName == 'PGDL2':
        atk = torchattacks.PGDL2(model, eps=128 / 255, alpha=15 / 255, steps=10, random_start=False)
        #atk = torchattacks.BIM(model, eps=8 / 255, alpha=2 / 255, steps=10)
    elif attackName == 'BIM':
        atk = torchattacks.BIM(model, eps=8 / 255, alpha=2 / 255, steps=10)
    elif attackName == 'CW':
        atk = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
    elif attackName == 'AutoAttack':
        atk = torchattacks.AutoAttack(model, eps=8 / 255)

    totalTest = []
    totalTestLabel = []
    n_classes = dataset.get_offsets()[1]
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > model.get_debug_iters():
                break
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            #Produce the adversarial samples
            adv_images = atk(inputs, labels)
            if np.shape(totalTest)[0] == 0:
                totalTest = adv_images
                totalTestLabel = labels
            else:
                totalTest = torch.cat([totalTest,adv_images],0)
                totalTestLabel = torch.cat([totalTestLabel,labels],0)

            '''
            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs = model(adv_images, k)
            else:
                if model.NAME == 'hybridmixture':
                    outputs = model.myPrediction(inputs,k)
                else:
                    outputs = model(inputs)

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
            '''

        '''
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
        '''

    acc = clean_accuracy(model, totalTest, totalTestLabel)
    model.net.train(status)
    return acc
    #return accs, accs_mask_classes


def evaluate_defense_ensemble(model: ContinualModel, dataset: ContinualDataset, last=False,attackName="") -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []

    #atk = torchattacks.PGD(model.net, eps=8 / 255, alpha=2 / 255, steps=4)

    if attackName == 'FGSM':
        print("This is FGSM")
        atk = torchattacks.FGSM(model, eps=8 / 255)
    elif attackName == 'PGD':
        atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=False)
    elif attackName == 'PGDL2':
        atk = torchattacks.PGDL2(model, eps=128 / 255, alpha=15 / 255, steps=10, random_start=False)
        #atk = torchattacks.BIM(model, eps=8 / 255, alpha=2 / 255, steps=10)
    elif attackName == 'BIM':
        atk = torchattacks.BIM(model, eps=8 / 255, alpha=2 / 255, steps=10)
    elif attackName == 'CW':
        atk = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
    elif attackName == 'AutoAttack':
        atk = torchattacks.AutoAttack(model, eps=8 / 255)

    accArr = []
    n_classes = dataset.get_offsets()[1]
    for k, test_loader in enumerate(dataset.test_loaders):
        totalTest = []
        totalTestLabel = []

        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > model.get_debug_iters():
                break
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            #Produce the adversarial samples
            adv_images = atk(inputs, labels)
            if np.shape(totalTest)[0] == 0:
                totalTest = adv_images
                totalTestLabel = labels
            else:
                totalTest = torch.cat([totalTest,adv_images],0)
                totalTestLabel = torch.cat([totalTestLabel,labels],0)

            '''
            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs = model(adv_images, k)
            else:
                if model.NAME == 'hybridmixture':
                    outputs = model.myPrediction(inputs,k)
                else:
                    outputs = model(inputs)

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
            '''
        model.SetCurrentExpertByIndex(k)
        acc1 = clean_accuracy(model, totalTest, totalTestLabel)
        accArr.append(acc1)
        '''
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
        '''

    acc = np.mean(accArr)
    model.net.train(status)
    return acc
    #return accs, accs_mask_classes


def initialize_wandb(args: Namespace) -> None:
    """
    Initializes wandb, if installed.

    Args:
        args: the arguments of the current execution
    """
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    run_name = args.wandb_name if args.wandb_name is not None else args.model

    run_id = random_id(5)
    name = f'{run_name}_{run_id}'
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=name)
    args.wandb_url = wandb.run.get_url()


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        initialize_wandb(args)

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    if args.start_from is not None:
        for i in range(args.start_from):
            train_loader, _ = dataset.get_data_loaders()
            model.meta_begin_task(dataset)
            model.meta_end_task(dataset)

    if args.loadcheck is not None:
        model, past_res = mammoth_load_checkpoint(args, model)

        if not args.disable_log and past_res is not None:
            (results, results_mask_classes, csvdump) = past_res
            logger.load(csvdump)

        print('Checkpoint Loaded!')

    progress_bar = ProgressBar(joint=args.joint, verbose=not args.non_verbose)

    if args.enable_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    start_task = 0 if args.start_from is None else args.start_from
    end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

    torch.cuda.empty_cache()

    start = time.time()
    for t in range(start_task, end_task):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        model.meta_begin_task(dataset)

        #model.end_task2(dataset,train_loader)
        #return

        if not args.inference_only:
            if t and args.enable_other_metrics:
                accs = evaluate(model, dataset, last=True)
                results[t - 1] = results[t - 1] + accs[0]
                if dataset.SETTING == 'class-il':
                    results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]

            scheduler = dataset.get_scheduler(model, args) if not hasattr(model, 'scheduler') else model.scheduler

            #Add data sample
            maxSize = model.args.buffer_size
            longTermSize = int(maxSize / 2)
            maxNofTasks = dataset.N_TASKS

            selectedMaxSize = int(longTermSize / maxNofTasks)
            dataX, dataY, dataLogit = [], [], []
            #end data

            isFirst = True
            for epoch in range(model.args.n_epochs):
                train_iter = iter(train_loader)
                data_len = None
                if not isinstance(dataset, GCLDataset):
                    data_len = len(train_loader)
                i = 0
                while True:
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        isFirst = False
                        break
                    if args.debug_mode and i > model.get_debug_iters():
                        break
                    if hasattr(dataset.train_loader.dataset, 'logits'):
                        inputs, labels, not_aug_inputs, logits = data
                        inputs = inputs.to(model.device)
                        labels = labels.to(model.device, dtype=torch.long)
                        not_aug_inputs = not_aug_inputs.to(model.device)
                        logits = logits.to(model.device)
                        loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, epoch=epoch)
                        #if epoch == 0:
                        #    model.TSFramework.AddSupervisedDataBatch(inputs, labels,logits)
                    else:
                        inputs, labels, not_aug_inputs = data
                        inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
                        not_aug_inputs = not_aug_inputs.to(model.device)
                        loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch)

                        #Add data
                        if epoch == 0:
                            with torch.no_grad():
                                outputs = model.net(inputs)

                            labels2 = labels
                            inputs2 = inputs
                            not_aug_inputs2 = not_aug_inputs
                            outputs2 = outputs

                            #inputs2 = inputs2.to(torch.device("cpu"))
                            not_aug_inputs2 = not_aug_inputs2.to(torch.device("cpu"))
                            outputs2 = outputs2.to(torch.device("cpu"))

                            if np.shape(dataX)[0] == 0:
                                dataX = not_aug_inputs2
                                dataY = labels2
                                dataLogit = outputs2
                            else:
                                # print(dataX.size())
                                dataX = torch.cat((dataX, not_aug_inputs2), 0)
                                dataY = torch.cat((dataY, labels2), 0)
                                dataLogit = torch.cat((dataLogit, outputs2), 0)
                        #End data
                    # print("The loss:", loss)
                    assert not math.isnan(loss)
                    #progress_bar.prog(i, data_len, epoch, t, loss)
                    i += 1
                    #return

                if scheduler is not None:
                    scheduler.step()

                if args.eval_epochs is not None and epoch % args.eval_epochs == 0 and epoch < model.args.n_epochs - 1:
                    epoch_accs = evaluate(model, dataset)

                    log_accs(args, logger, epoch_accs, t, dataset.SETTING, epoch=epoch)

            dataX = torch.reshape(dataX, (-1, 3, 32, 32))
            dataY = torch.reshape(dataY, (np.shape(dataY)[0], -1))
            dataLogit = torch.reshape(dataLogit, (np.shape(dataLogit)[0], -1))

            if model.NAME == 'pnn':
                model.processData(dataX,dataY,dataLogit,selectedMaxSize)

        #model.end_task2(dataset, train_loader,selectedMaxSize)
        model.meta_end_task(dataset)

        end = time.time()
        TrainingTime = end-start
        print("Total training time")
        print(TrainingTime)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        log_accs(args, logger, accs, t, dataset.SETTING)

        if args.savecheck:
            save_obj = {
                'model': model.state_dict(),
                'args': args,
                'results': [results, results_mask_classes, logger.dump()],
                'optimizer': model.opt.state_dict() if hasattr(model, 'opt') else None,
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
            }
            if 'buffer_size' in model.args:
                save_obj['buffer'] = deepcopy(model.buffer).to('cpu')

            # Saving model checkpoint
            checkpoint_name = f'checkpoints/{args.ckpt_name}_joint.pt' if args.joint else f'checkpoints/{args.ckpt_name}_{t}.pt'
            torch.save(save_obj, checkpoint_name)

    #Calculate the number of parameters
    total = sum([param.nelement() for param in model.net.parameters()])
    print("The total number of parameters for the model")
    print(total)

    if model.NAME == 'siamesevit':
        # Perform the adversarial evaluation
        myAccsArr = evaluate_defense_ensemble(model, dataset, attackName='FGSM')
        print("FGSM")
        print(myAccsArr)

        myAccsArr = evaluate_defense_ensemble(model, dataset, attackName='PGD')
        print("PGD")
        print(myAccsArr)

        myAccsArr = evaluate_defense_ensemble(model, dataset, attackName='PGDL2')
        print("PGDL2")
        print(myAccsArr)

        myAccsArr = evaluate_defense_ensemble(model, dataset, attackName='BIM')
        print("BIM")
        print(myAccsArr)

        myAccsArr = evaluate_defense_ensemble(model, dataset, attackName='CW')
        print("CW")
        print(myAccsArr)

        myAccsArr = evaluate_defense_ensemble(model, dataset, attackName='AutoAttack')
        print("AutoAttack")
        print(myAccsArr)
    else:
        #Perform the adversarial evaluation
        myAccsArr = evaluate_defense(model, dataset,attackName='FGSM')
        print("FGSM")
        print(myAccsArr)

        myAccsArr = evaluate_defense(model, dataset,attackName='PGD')
        print("PGD")
        print(myAccsArr)

        myAccsArr = evaluate_defense(model, dataset, attackName='PGDL2')
        print("PGDL2")
        print(myAccsArr)

        myAccsArr = evaluate_defense(model, dataset, attackName='BIM')
        print("BIM")
        print(myAccsArr)

        myAccsArr = evaluate_defense(model, dataset, attackName='CW')
        print("CW")
        print(myAccsArr)

        myAccsArr = evaluate_defense(model, dataset, attackName='AutoAttack')
        print("AutoAttack")
        print(myAccsArr)

    if args.validation:
        del dataset
        args.validation = None

        final_dataset = get_dataset(args)
        for _ in range(final_dataset.N_TASKS):
            final_dataset.get_data_loaders()
        accs = evaluate(model, final_dataset)
        log_accs(args, logger, accs, t, final_dataset.SETTING, prefix="FINAL")

    if not args.disable_log and args.enable_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                           results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
