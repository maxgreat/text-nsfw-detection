import torch
import fastText

from transformers import BertTokenizer

from flask import Flask

app = Flask(__name__)

model = torch.load("data/best_fasttext_lstm_ext.pth")
fasttext_model = fastText.load_model("/data/m.portaz/cc.fr.300.bin")

bert_model = torch.load("data/best_bert_queries_extended.pth")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


@app.route("/nsfw/<text>")
def is_porn(text):
    print("Text received:", text)
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


@app.route("/bert/<text>")
def bert_porn(text):
    print("Text received:", text)
    inputs = torch.LongTensor([tokenizer.encode(text)])

    pred = bert_model(inputs, 1)
    if pred <= 0:
        return "Safe " + str(pred.detach().cpu().numpy()) + "\n"
    else:
        return "NSFW " + str(pred.detach().cpu().numpy()) + "\n"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8000", debug=True)
