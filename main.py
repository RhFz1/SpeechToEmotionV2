from model_endpoint import make_prediction

emo_string = "neutral calm happy sad angry fearful disgust surprised"
idx_emo = emo_string.split()
emo_dict = {key: value for key, value in enumerate(idx_emo)}

f1 = "./data/Actor_01/03-01-08-02-02-02-01.wav"

emotion, probs = make_prediction(f1)

temp = list(probs)
mapping = {key : value for key, value in enumerate(temp)}
mapping = dict(sorted(mapping.items(), key= lambda x : -x[1]))
for key in mapping:
    print(f"{emo_dict[key].rjust(10)} ----- {mapping[key]*100:.2f}%")

