import random


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
    train.extend(other[: (nb_fold * 4)])
    print(len(train))
    print("Nb val :", nb_fold)
    val = porn[(nb_fold * 4) :]
    val.extend(other[(nb_fold * 4) : len(porn)])
    print(len(val))
    print("Last shuffle")
    random.shuffle(train)
    random.shuffle(val)

    return train, val
