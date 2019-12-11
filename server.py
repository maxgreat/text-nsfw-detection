import torch
import fasttext

from transformers import BertTokenizer

from flask import Flask, request

app = Flask(__name__)

model = torch.load("data/best_fasttext_lstm_ext.pth")
fasttext_model = fasttext.load_model("/data/m.portaz/cc.fr.300.bin")

bert_model = torch.load("data/best_bert_extra_website.pth")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# fast_model = torch.load("data/best_fasttext_sum_queries_extended.pth")
# fasttext_model = fastText.load_model("/data/m.portaz/cc.fr.300.bin")


@app.route("/fasttext", methods=["GET", "POST"])
def is_porn():
    text = None
    if request.method == "GET":
        text = request.args.get("text")
        print("Text received:", text)
    elif request.method == "POST":
        data = request.form  # a multidict containing POST data
        text = data.get("text")
    if text is not None:
        if " " in text:
            inputs = torch.Tensor(
                [fasttext_model.get_word_vector(w) for w in text.split(" ")]
            ).unsqueeze(0)
        else:
            inputs = torch.Tensor([fasttext_model.get_word_vector(text)]).unsqueeze(0)
        pred = model(inputs, torch.LongTensor([len(text.split(" "))]))
        if pred <= 0:
            return "Safe " + str(pred.detach().cpu().numpy()) + "\n"
        else:
            return "NSFW " + str(pred.detach().cpu().numpy()) + "\n"


@app.route("/bert", methods=["GET", "POST"])
def bert_porn():
    if request.method == "GET":
        text = request.args.get("text")
        print("Text received:", text)
        if text is not None:
            inputs = torch.LongTensor([tokenizer.encode(text)]).cuda()
            pred = bert_model(inputs, 1)
            if pred <= 0:
                return "Safe " + str(pred.detach().cpu().numpy()) + "\n"
            else:
                return "NSFW " + str(pred.detach().cpu().numpy()) + "\n"
    elif request.method == "POST":
        data = request.form  # a multidict containing POST data
        text = data.get("text")
        if text is not None:
            inputs = torch.LongTensor([tokenizer.encode(text)]).cuda()
            pred = bert_model(inputs, 1)
            if pred <= 0:
                return "Safe " + str(pred.detach().cpu().numpy()) + "\n"
            else:
                return "NSFW " + str(pred.detach().cpu().numpy()) + "\n"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8010", debug=True)
