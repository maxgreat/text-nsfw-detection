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

MARGIN = 0.0


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
    parser.add_argument("--bert_model", default="bert-base-multilingual-cased")
    parser.add_argument("--epochs", default=10, help="Numbers of epochs", type=int)
    parser.add_argument(
        "--extra_data",
        default=[
            "/data/datasets/porn/textual-data/wiki/wiki_Anatomie_Sexualité_et_sexologie_LGBT_WikiExtractor_text.jsonl"
        ],
    )

    return parser.parse_args()


def eval_recall(l1, l2):
    score = 0
    nb_porn = 0
    nb_other = 0
    avg_porn = 0
    avg_other = 0
    for i, guess in enumerate(l1):
        if l2[i] <= 0:
            nb_other += 1
            avg_other += guess
            if guess < MARGIN:
                score += 1
        else:
            nb_porn += 1
            avg_porn += guess
            if guess > MARGIN:
                score += 1
    return score / len(l1), (nb_porn, avg_porn / len(l1), nb_other, avg_other / len(l1))


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
            score, _ = eval_recall(np.asarray(last_guesses), np.asarray(last_correct))
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
        model = models.FastTextLSTM()
    elif args.model == "SUM":
        model = models.FastTextSum()
    elif args.model == "BertSum":
        model = models.BertSumer(args.bert_model)
    elif args.model == "Bert":
        model = models.BertPooler(args.bert_model)
    else:
        print("Unknown model to use :", args.model)

    model = nn.DataParallel(model.train().cuda())
    print("Creating dataset", end="", flush=True)
    t = time.time()

    if "Bert" in args.model:
        train_dataset = BertTextDataset(args.dataset_dir + "train.txt", args.bert_model)
        if len(args.extra_data) > 0:
            for arg in args.extra_data:
                train_dataset.add_data(arg)
        print("...", end="", flush=True)
        print("train set :", len(train_dataset))
        val_dataset = BertTextDataset(args.dataset_dir + "test.txt", args.bert_model)
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
    # criterium = nn.HingeEmbeddingLoss(margin=MARGIN).cuda()
    criterium = nn.SoftMarginLoss().cuda()
    if "Bert" in args.model:
        opti = optim.Adam(model.module.model.pooler.parameters(), lr=args.lr)
    else:
        opti = optim.Adam(model.parameters(), lr=args.lr)
    print("First evaluation :")
    loss_avg, (best_rec, nbs) = eval(val_loader, model, criterium)
    print("Avg loss : ", loss_avg)
    print("precision : ", best_rec)
    print("Numbers : ", nbs)
    best_rec = 0

    scheduler = optim.lr_scheduler.MultiStepLR(
        opti, milestones=args.lrd[1:], gamma=args.lrd[0]
    )
 
    for epoch in range(args.epochs):
        print("Train")
        train_loss = train(train_loader, model, criterium, opti, epoch)
        print("Validation")
        loss, (score, nbs) = eval(val_loader, model, criterium)
        print("Score : ", score)
        print("Numbers : ", nbs)

        if score > best_rec:
            print("saving best model")
            best_rec = score
            torch.save(model, "data/best_" + args.name + ".pth")

        if "Bert" in args.model:
            if epoch == 1:
                opti.add_param_group(
                    {"params": model.module.model.encoder.parameters(), "lr": args.lr}
                )
                scheduler = optim.lr_scheduler.MultiStepLR(
                    opti, milestones=args.lrd[1:], gamma=args.lrd[0]
                )

        logger.add_scalar("Loss/loss", opti.param_groups[0]["lr"], epoch)
        logger.add_scalar("Loss/train", train_loss, epoch)
        logger.add_scalar("Loss/validation", loss, epoch)
        logger.add_scalar("Eval/validation", score, epoch)
        scheduler.step()


if __name__ == "__main__":
    main()
