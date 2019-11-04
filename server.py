import models
import torch
import argparse
import fastText


def parse_args():
    parser = argparse.ArgumentParser(description="Classify text")
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = torch.load(args.model)
    fasttext_model = fastText.load_model("/data/m.portaz/wiki.en.bin")

    inputs = torch.Tensor(
        [fasttext_model.get_word_vector(w) for w in args.text.split(" ")]
    ).unsqueeze(0)

    pred = model(inputs, torch.Tensor([len(args.text.split(" "))]))
    print(pred)
