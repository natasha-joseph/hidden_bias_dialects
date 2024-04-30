import helpers
import pandas as pd
import tqdm

if __name__ == "__main__":
    loss_df = pd.read_csv("/content/hidden_bias_dialects/data/sae_southern_sentence_pair_losses.csv")

    sen_sim = []
    for row in tqdm.tqdm(loss_df["pairs"]):
      sen_sim.append(helpers.similarity_score(row[0], row[1]))

    loss_df["sen_sim"] = sen_sim

    print(loss_df)
    loss_df.to_csv("/content/hidden_bias_dialects/data/sae_southern_sentence_pair_losses_sen_sim.csv", index = False)