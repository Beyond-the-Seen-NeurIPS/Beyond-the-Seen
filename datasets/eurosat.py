import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
import numpy as np
from collections import defaultdict
import random
import math
import csv

NEW_CNAMES = {
    "AnnualCrop": "Annual Crop Land",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land",
    "Highway": "Highway or Road",
    "Industrial": "Industrial Buildings",
    "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land",
    "Residential": "Residential Buildings",
    "River": "River",
    "SeaLake": "Sea or Lake",
}


@DATASET_REGISTRY.register()
class EuroSAT(DatasetBase):

    dataset_dir = "eurosat"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "2750")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
        num_shots = cfg.DATASET.NUM_SHOTS
        seed = cfg.SEED
        if cfg.rebuttal_options.long_tailed.enabled:
            if cfg.rebuttal_options.long_tailed.padding:
                preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}_longtailed_imgf-{cfg.rebuttal_options.long_tailed.imb_factor}.pkl")
            else:
                preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}_longtailed_imgf-{cfg.rebuttal_options.long_tailed.imb_factor}_nopadding.pkl")
        else:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

        if num_shots >= 1:
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                if cfg.rebuttal_options.long_tailed.enabled:
                    assert cfg.DATASET.NUM_SHOTS  == 16, "Long tail experiment is only conducted when shot == 16"
                    train = EuroSAT.gen_long_tailed(train, cfg)
                    val = EuroSAT.gen_long_tailed(val, cfg)
                train = self.generate_fewshot_dataset(train, num_shots=num_shots, repeat=cfg.rebuttal_options.long_tailed.enabled and cfg.rebuttal_options.long_tailed.padding)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4), repeat=cfg.rebuttal_options.long_tailed.enabled and cfg.rebuttal_options.long_tailed.padding)
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        if cfg.KD.USE_DATASET_LIST != 0:
            train = test
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        if cfg.rebuttal_options.disjoint_eval:
            # make test set disjoint with generated candidates
            if subsample == 'new':
                OxfordPets.remove_disjoint(test, cfg)

        super().__init__(train_x=train, val=val, test=test)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CLASSNAMES[cname_old]
            item_new = Datum(impath=item_old.impath, label=item_old.label, classname=cname_new)
            dataset_new.append(item_new)
        return dataset_new
    
    @staticmethod
    def gen_long_tailed(data, cfg, num_meta=0):
        #有点臃肿，懒得改了。 实际上不会执行subsample=='new'的相关部分，这部分逻辑删掉也无所谓
        imb_factor = cfg.rebuttal_options.long_tailed.imb_factor
        subsample=cfg.DATASET.SUBSAMPLE_CLASSES

        if subsample not in ['base', 'new']:
            raise NotImplementedError()
        
        
        cls_to_samples = defaultdict(list)
        for i in data:
            cls_to_samples[i.label].append(i)
        cls_num = len(cls_to_samples)
        m = math.ceil(cls_num / 2)
        data_nochange = []
        if subsample == 'base':
            chosen = lambda x:x < m
        elif subsample == 'new':
            chosen = lambda x:x >= m
        cls_to_samples_tmp = {}
        for (cls, samples) in cls_to_samples.items():
            if chosen(cls):
                cls_to_samples_tmp[cls] = samples
            else:
                data_nochange.extend(samples)
        cls_to_samples = cls_to_samples_tmp
        cls_num = len(cls_to_samples)
        cls_to_num = {k:len(v) for k,v in cls_to_samples.items()}
        #num is imbalanced
        img_max = min(cls_to_num.values())

        if imb_factor is None:
            img_num_per_cls = [img_max] * cls_num
        else:
            img_num_per_cls = []
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))

        longtailed_data = []
        for cls, num_samples in enumerate(img_num_per_cls):
            longtailed = random.sample(cls_to_samples[cls], num_samples)
            longtailed_data.extend(longtailed)
        

        return longtailed_data + data_nochange
    
