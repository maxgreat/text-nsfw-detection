import random
import json


def split_train_text(queries):
    print("Number of queries :", len(queries))
    s = {}
    print("Reading queries")
    for i, q in enumerate(queries):
        if i % 1000 == 0:
            advance = (i / len(queries)) * 100
            print("%2.2f" % advance, end="\r")
        label, sentence = q.split(" ", 1)
        if sentence not in s:
            s[sentence] = 0
        if label == "__label__pron" or label == "__label__1":
            s[sentence] += 1
        else:
            s[sentence] -= 1

    print("Filtering queries")
    porn, other = [], []
    for i, sentence in enumerate(s):
        if i % 1000 == 0:
            advance = (i / len(s)) * 100
            print("%2.2f" % advance, end="\r")
        if s[sentence] > 0:
            porn.append("__label__pron " + sentence)
        else:
            other.append("__label__others " + sentence)

    print("Shuffling")
    random.shuffle(porn)
    random.shuffle(other)
    print("Selection")
    nb_fold = len(porn) // 5
    print("Nb train : ", nb_fold * 4)
    train = porn[: (nb_fold * 4)]
    train.extend(other[: (nb_fold * 4) * 3])
    print(len(train))
    print("Nb val :", nb_fold)
    val = porn[(nb_fold * 4) :]
    val.extend(other[(nb_fold * 4) : len(porn)])
    print(len(val))

    print("Adding important queries")
    important_dict = ["homme", "femme", "gay", "lesbienne"]
    added_queries = []
    for word in important_dict:
        added_queries.extend(
            [query for query in other if word in query and "__label__other" in query]
        )
    random.shuffle(added_queries)
    print("Adding", len(added_queries), " queries")
    train.extend(added_queries[:-100])
    val.extend(added_queries[-100:])

    print("Last shuffle")
    random.shuffle(train)
    random.shuffle(val)

    return train, val


def read_wiki_dump(wiki_file):
    data = [json.loads(line)["text"] for line in wiki_file]
    return data


def write_safe(data):
    for d in data:
        print("__label__other " + d.replace("\n", " "))


if __name__ == "__main__":
    """
    lines = open(
        "/data/datasets/porn/textual-data/queries/queries_with_femmeactuelle_senscritique_tetu.txt"
    ).readlines()

    train, val = split_train_text(lines)
    fout = open("/data/datasets/porn/textual-data/queries/ext_train.txt", "w")
    for e in train:
        fout.write(e)
    fout = open("/data/datasets/porn/textual-data/queries/ext_test.txt", "w")
    for e in val:
        fout.write(e)
    """
    data = read_wiki_dump(
        open(
            "/data/datasets/porn/textual-data/wiki/wiki_Anatomie_Sexualité_et_sexologie_LGBT_WikiExtractor_text.jsonl"
        )
    )
    write_safe(data)
