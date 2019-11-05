import models
import torch
import argparse
import fastText

from flask import Flask, escape, request

app = Flask(__name__)

model = torch.load("")
fasttext_model = fastText.load_model("/data/m.portaz/wiki.en.bin")


@app.route("/<search>")
def is_porn(search):
    inputs = torch.Tensor(
        [fasttext_model.get_word_vector(w) for w in args.text.split(" ")]
    ).unsqueeze(0)
    pred = model(inputs, torch.LongTensor([len(args.text.split(" "))]))
    return str(pred.detach().cpu().numpy())


def parse_args():
    parser = argparse.ArgumentParser(description="Classify text")
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--type", default="LSTM", help="LSTM or SUM")
    return parser.parse_args()


if __name__ == "__main__":
    app.run()
    args = parse_args()

    pred = model(inputs, torch.LongTensor([len(args.text.split(" "))]))
    print(pred)
