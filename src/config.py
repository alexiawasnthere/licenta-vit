from dataclasses import dataclass
from pathlib import Path
import os
import random

import tensorflow as tf
import numpy as np
import pandas as pd


@dataclass(frozen=True)#obiectul devine read-only
class Config:
    # data root: data/Train/<id>/00001.jpg etc
    DATA_ROOT: Path = Path("data")

    # csv files
    CSV_TRAIN: Path = Path("Train.csv")
    CSV_VAL: Path = Path("Validation.csv")
    CSV_TEST: Path = Path("Test.csv")

    # video settings
    NUM_FRAMES: int = 37
    
    IMG_SIZE: int = 128

    # training settings
    BATCH_SIZE: int = 4
    SEED: int = 42

    # csv columns
    COL_VIDEO_ID: str = "video_id"
    COL_LABEL: str = "label"
    COL_LABEL_ID: str = "label_id"
    COL_FRAMES: str = "frames"

CFG = Config()


def set_seed(seed: int) -> None:
    # seed pentru reproducibilitate (cat se poate)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_csv(csv_path: Path, require_labels: bool = True) -> pd.DataFrame:
    # incarca un csv si verifica daca are coloanele necesare

    if not csv_path.exists():
        raise FileNotFoundError(f"nu se gaseste CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)

    # Test.csv are "id" in loc de "video_id"
    if "id" in df.columns and CFG.COL_VIDEO_ID not in df.columns:
        df = df.rename(columns={"id": CFG.COL_VIDEO_ID})

    needed = [CFG.COL_VIDEO_ID, CFG.COL_FRAMES]
    if require_labels:
        needed += [CFG.COL_LABEL, CFG.COL_LABEL_ID]

    for c in needed:
        if c not in df.columns:
            raise ValueError(f"in {csv_path.name} lipseste coloana: {c}. are: {df.columns.tolist()}")

    # conversii care sunt mereu ok
    df[CFG.COL_VIDEO_ID] = df[CFG.COL_VIDEO_ID].astype(int)
    df[CFG.COL_FRAMES] = df[CFG.COL_FRAMES].astype(int)

    # conversii doar daca avem labels (Train/Val)
    if require_labels:
        df[CFG.COL_LABEL] = df[CFG.COL_LABEL].astype(str)
        df[CFG.COL_LABEL_ID] = df[CFG.COL_LABEL_ID].astype(int)

    return df


def make_label_mapping(df_train: pd.DataFrame):
    # construieste mapping label <-> id doar din train
    label_to_id = {}
    id_to_label = {}

    #se vor verificat doar combinatiile unice
    unique_pairs = df_train[[CFG.COL_LABEL, CFG.COL_LABEL_ID]].drop_duplicates()

    for _, row in unique_pairs.iterrows():
        label = row[CFG.COL_LABEL]
        lid   = int(row[CFG.COL_LABEL_ID])

        # daca label-ul exista deja, trebuie sa aiba acelasi id
        if label in label_to_id and label_to_id[label] != lid:
            raise ValueError(f"in Train: label '{label}' apare cu id-uri diferite: "
                             f"{label_to_id[label]} si {lid}")

        # daca id-ul exista deja, trebuie sa aiba acelasi label
        if lid in id_to_label and id_to_label[lid] != label:
            raise ValueError(f"in Train: id {lid} apare cu label-uri diferite: "
                             f"'{id_to_label[lid]}' si '{label}'")

        label_to_id[label] = lid
        id_to_label[lid] = label

    # VERIFICA DACA ID URILE SUNT 0..K-1
    ids = sorted(id_to_label.keys())
    if ids != list(range(len(ids))):
        raise ValueError(
            f"Label_id-urile nu sunt consecutive 0..K-1. Primele/ultimele: {ids[:5]} ... {ids[-5:]}"
        )
    
    num_classes = len(label_to_id)

    # class_names in ordinea id-urilor (sortate)
    class_names = [id_to_label[i] for i in range(num_classes)]

    return num_classes, class_names, label_to_id, id_to_label

def check_split(df: pd.DataFrame, label_to_id: dict, split_name: str):
    # verifica daca split-ul are aceleasi label_id ca train
    unique_pairs = df[[CFG.COL_LABEL, CFG.COL_LABEL_ID]].drop_duplicates()

    for _, row in unique_pairs.iterrows():
        label = row[CFG.COL_LABEL]
        lid   = int(row[CFG.COL_LABEL_ID])

        if label not in label_to_id:
            raise ValueError(f"{split_name}: label necunoscut (nu e in Train): '{label}'")

        if label_to_id[label] != lid:
            raise ValueError(f"{split_name}: mismatch pentru '{label}': "
                             f"in Train e {label_to_id[label]}, aici e {lid}")
        

def main():
    # test rapid pentru config + csv
    set_seed(CFG.SEED)

    train = load_csv(CFG.CSV_TRAIN, require_labels=True)
    val   = load_csv(CFG.CSV_VAL, require_labels=True)
    test  = load_csv(CFG.CSV_TEST, require_labels=False)

    # verifica frames=37
    for name, df in [("Train", train), ("Validation", val), ("Test", test)]:
        if df[CFG.COL_FRAMES].min() != CFG.NUM_FRAMES or df[CFG.COL_FRAMES].max() != CFG.NUM_FRAMES:
            print(f"[WARN] {name}: frames nu e mereu {CFG.NUM_FRAMES}. "
                  f"min={df[CFG.COL_FRAMES].min()}, max={df[CFG.COL_FRAMES].max()}")

    num_classes, class_names, label_to_id, id_to_label = make_label_mapping(train)
    check_split(val, label_to_id, "Validation")
    # test nu are label uri, asa ca nu avem ce sa verificam

    print("Train rows:", len(train))
    print("Val rows:  ", len(val))
    print("Test rows: ", len(test))
    print("Num classes:", num_classes)
    print("Primele 10 clase:", class_names[:10])

    print("\nexemplu rand Train[0]:")
    print(train.iloc[0].to_dict())


if __name__ == "__main__":
    main()