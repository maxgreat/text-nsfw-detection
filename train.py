import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import models
from dataset import FastTextDataset, collate, BertTextDataset, collate_simple
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Train text classification")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("-n", "--name", required=True)
    parser.add_argument(
        "--tokenizer", default="nltk", help="Choice of tokenizer : nltk, bert."
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--lr", default=0.00001, type=float)
    parser.add_argument(
        "--lrd", default=[0.5, 2, 4, 6, 8], help="learning rate schedule"
    )
    parser.add_argument("--model", default="LSTM", help="LSTM or SUM or Bert")
    parser.add_argument("--epochs", default=10, help="Numbers of epochs")
    return parser.parse_args()


def eval_recall(l1, l2):
    return (np.argmax(l1, 1) == l2).sum() / l2.size


def train(loader, model, crit, optimizer, epoch, print_freq=100):
    """
        train model on data from loader for epoch epochs
    """
    model = model.train()
    loss_avg = 0
    last_guesses, last_correct = [], []
    for i, (inputs, lengths, labels) in enumerate(loader):
        print("%2.2f" % (i / len(loader) * 100), "\%", end="\r")
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        batch_time = time.time()
        out_class = model(inputs, lengths)
        last_guesses.extend(out_class.cpu().detach().numpy())
        last_correct.extend(labels.cpu().detach().numpy())
        loss = crit(out_class, labels)
        loss_avg += loss.item()
        loss.backward()
        optimizer.step()
        batch_time = time.time() - batch_time
        if i % print_freq == print_freq - 1 or i == (len(loader) - 1):
            score = eval_recall(np.asarray(last_guesses), np.asarray(last_correct))
            last_guesses, last_correct = [], []
            print(
                "[",
                epoch,
                "] ",
                i,
                "/",
                len(loader),
                "\tTime %2.2f" % batch_time,
                "\tLoss %2.2f" % loss.item(),
                "[%2.2f]" % (loss_avg / i),
                "\tScore : ",
                score,
            )
    return loss_avg / len(loader)


def eval(loader, model, crit):
    model = model.eval()
    loss_avg = 0
    guess, correct = [], []
    for i, (inputs, lengths, labels) in enumerate(loader):
        print("%2.2f" % (i / len(loader) * 100), "\%", end="\r")
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            out_class = model(inputs, lengths)
            guess.extend(out_class.cpu().detach().numpy())
            correct.extend(labels.cpu().detach().numpy())
            loss = crit(out_class, labels)
            loss_avg += loss.item()
    return loss_avg / i, eval_recall(np.array(guess), np.array(correct))


def main():
    args = parse_args()

    logger = SummaryWriter(os.path.join("./logs/", args.name))
    if args.model == "LSTM":
        model = nn.DataParallel(models.FastTextLSTM()).train().cuda()
    elif args.model == "SUM":
        model = nn.DataParallel(models.FastTextSum()).train().cuda()
    elif args.model == "Bert":
        model = nn.DataParallel(models.BertFinetuned().train().cuda())

    print("Creating dataset", end="", flush=True)
    t = time.time()

    if args.model == "Bert":
        train_dataset = BertTextDataset(args.dataset_dir + "train.txt")
        print("...", end="", flush=True)
        val_dataset = BertTextDataset(args.dataset_dir + "test.txt")
        print("Done in ", time.time() - t, "sec.", flush=True)
    else:
        train_dataset = FastTextDataset(
            args.dataset_dir + "train.txt", sum=(args.model == "SUM")
        )
        print("...", end="", flush=True)
        val_dataset = FastTextDataset(
            args.dataset_dir + "test.txt", sum=(args.model == "SUM")
        )
        print("Done in ", time.time() - t, "sec.", flush=True)

    if args.model == "SUM":
        coll = collate_simple
    else:
        coll = collate

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=coll,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=coll,
        drop_last=False,
    )

    criterium = nn.CrossEntropyLoss().cuda()
    opti = optim.Adam(model.parameters(), lr=args.lr)
    print("First evaluation :")
    loss_avg, best_rec = eval(val_loader, model, criterium)
    print("Avg loss : ", loss_avg)
    print("precision : ", best_rec)

    scheduler = optim.lr_scheduler.MultiStepLR(
        opti, milestones=args.lrd[1:], gamma=args.lrd[0]
    )

    for epoch in range(args.epochs):
        print("Train")
        train_loss = train(train_loader, model, criterium, opti, epoch)
        print("Validation")
        loss, score = eval(val_loader, model, criterium)
        print("Score : ", score)

        if score > best_rec:
            print("saving best model")
            best_rec = score
            torch.save(model, "data/best_" + args.name + ".pth")
        logger.add_scalar("Loss/loss", opti.param_groups[0]["lr"])
        logger.add_scalar("Loss/train", train_loss, epoch)
        logger.add_scalar("Loss/validation", loss, epoch)
        logger.add_scalar("Eval/validation", score, epoch)
        scheduler.step()


if __name__ == "__main__":
    main()
