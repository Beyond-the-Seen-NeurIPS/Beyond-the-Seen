import os
import csv
import glob
import pickle
import random
from collections import defaultdict

from PIL import Image

import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T

from dassl.data.data_manager import build_data_loader
from dassl.data.transforms import INTERPOLATION_MODES, build_transform

from diffusers import StableDiffusionPipeline

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from utils import TimeRecorder

from volcenginesdkarkruntime import Ark
client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

import transformers
model_id = "/path/to/Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
  "text-generation",
  model=model_id,
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cuda",
)

import prompts


class FeatureDataset(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item[1],
            "index": idx
        }

        img0 = item[0].convert("RGB")

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


def api_request(message):

    completion = client.chat.completions.create(
        model="ep-20240916102032-kzjr5",
        messages = [
            {"role": "system", "content": prompts.api_prompt["system_prompt"]},
            {"role": "user", "content": message},
        ],
    )

    return completion.choices[0].message.content

def get_candidate_classes_api(cfg, model):
    matches = []
    cand_classes = []

    classnames = model.dm.dataset.classnames

    print("----- API request -----")
    for classname in classnames:
        message = prompts.api_prompt_notree[cfg.DATASET.NAME] + prompts.api_prompt_notree["user_prompt"] + classname + '?'

        answer = api_request(message)
        print(answer)

        if answer.startswith("A:"):    
            cand_class = answer[3:].split(" is")[0].lower()
        else:
            cand_class = answer.split(" is")[0].lower()

        if cand_class not in cand_classes and cand_class not in classnames and len(cand_class.split()) <= 10:
            matches.append( (classname, cand_class) )
            cand_classes.append(cand_class)
        else:
            matches.append( [classname] )

    print(cand_classes)

    with open(os.path.join( cfg.FUTURE.candidate_data_root, 'candidate_classes.csv' ), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(cand_classes)

    with open(os.path.join( cfg.FUTURE.candidate_data_root, 'candidate_classes_cor.csv' ), 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(matches)):
            writer.writerow(matches[i])


def llama_request(message):

    messages = [
        {"role": "system", "content": prompts.llama_prompt["system_prompt"]},
        {"role": "user", "content": message}
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=512,
    )

    return outputs[0]["generated_text"][-1]["content"]

def get_llama_captions(cfg, model):
    llama_captions = {}

    with open(os.path.join( cfg.FUTURE.candidate_data_root, 'llava_captions.pkl' ), 'rb') as file:
        llava_captions = pickle.load(file)
    print(llava_captions.keys())

    sour2gen = {}
    with open(os.path.join( cfg.FUTURE.candidate_data_root, 'candidate_classes_cor.csv' ), 'rt') as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) == 2:
                sour2gen[line[0]] = line[1]


    k = 8
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(), 
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711])
    ])
    print("----- llama request -----")
    for classname in sour2gen.keys():
        caption_list = llava_captions[classname][0]
        image_list = llava_captions[classname][1]

        caption_list, image_list = zip(*[
            (caption, image) for caption, image in zip(caption_list, image_list) if len( caption.split(' ') ) <= 55
        ])

        image_input = torch.cat( [ transform( Image.open(p).convert("RGB") ).unsqueeze(0) for p in image_list ] )
        similarity_matrix = model.compute_i2t_similarity(image_input, caption_list)

        _, indices = torch.topk(similarity_matrix.flatten(), k)
        caption_indices = indices % similarity_matrix.size(1)
        top_k_captions = [caption_list[i] for i in caption_indices]

        message = prompts.llama_prompt_notree["user_prompt"][0] + sour2gen[classname] + prompts.llama_prompt_notree["user_prompt"][1]
        print(message)

        for caption in top_k_captions:

            message += '\n' + caption

        llama_caption = llama_request(message)
        print(llama_caption)
        print('-----------------')
        llama_caption = llama_caption.split('\n')

        for i in range(len(llama_caption)):
            llama_caption[i] = llama_caption[i][3:]

        llama_captions[sour2gen[classname]] = llama_caption

    with open(os.path.join( cfg.FUTURE.candidate_data_root, 'llama_captions.pkl' ), 'wb') as file:
        pickle.dump(llama_captions, file)


def get_llava_captions(cfg, model):
    captions = {}
    source = model.dm.dataset.train_x
    fewshot_dataset = model.dm.dataset.split_dataset_by_label(source)

    model_path = '/path/to/llava-v1.6-vicuna-13b'

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )

    for label, image_list in fewshot_dataset.items():
        caption_list = []
        image_path_list = []
        classname = image_list[0].classname

        prompt = 'This is an image of ' + classname + '. Summarize the main style, scene, and key elements of this image in one sentence.'

        for image in image_list:
            image_file = image.impath
            image_path_list.append(image_file)

            args = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": prompt,
                "conv_mode": None,
                "image_file": image_file,
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
            })()

            caption = eval_model(args)
            caption_list.append(caption)

        captions[classname] = (caption_list, image_path_list)

    with open(os.path.join( cfg.FUTURE.candidate_data_root, 'llava_captions.pkl' ), 'wb') as file:
        pickle.dump(captions, file)


def get_sd_images(cfg, model):
    #Build Stable Diffsion to generate images of unknown classes
    print("Building Stable Diffsion")
    sd_model = StableDiffusionPipeline.from_pretrained(
        "/path/to/stable-diffusion-2-1", variant="fp16", torch_dtype=torch.float16
    )
    sd_model.to(model.device)

    with open(os.path.join( cfg.FUTURE.candidate_data_root, 'llama_captions.pkl' ), 'rb') as file:
        llama_captions = pickle.load(file)

    for cand_class in llama_captions.keys():
        root_path = os.path.join( cfg.FUTURE.candidate_data_root, 'images', cand_class )
        if not os.path.exists( root_path ):
            os.makedirs( root_path )

        for i in range( len(llama_captions[cand_class]) ):
            text_prompt = llama_captions[cand_class][i]
            print(text_prompt)

            for j in range(8):
                images = sd_model(prompt=text_prompt, num_images_per_prompt=cfg.FUTURE.num_per_class).images

                for k in range( cfg.FUTURE.num_per_class ):
                    images[k].save( os.path.join( root_path, str(i * 16 + j * cfg.FUTURE.num_per_class + k ) + '.jpg' ) )


def generate_candidate_data(cfg, classnames, model):
    if not os.path.exists( cfg.FUTURE.candidate_data_root ):
        os.makedirs( cfg.FUTURE.candidate_data_root )

    if not os.path.exists( os.path.join( cfg.FUTURE.candidate_data_root, 'llava_captions.pkl' ) ):
       get_llava_captions(cfg, model)

    if not os.path.exists( os.path.join( cfg.FUTURE.candidate_data_root, 'candidate_classes.csv' ) ):
       get_candidate_classes_api(cfg, model)

    if not os.path.exists( os.path.join( cfg.FUTURE.candidate_data_root, 'llama_captions.pkl' ) ):
       get_llama_captions(cfg, model)

    if not os.path.exists( os.path.join( cfg.FUTURE.candidate_data_root, 'images' ) ):
       get_sd_images(cfg, model)


def read_candidate_data(cfg):

    with open(os.path.join( cfg.FUTURE.candidate_data_root, 'candidate_classes.csv' ), 'rt') as f:
        reader = csv.reader(f)
        classnames = next(reader)

    print(classnames)

    item_list = []
    for classname in classnames:
        file_list = glob.glob( os.path.join( cfg.FUTURE.candidate_data_root, 'images', classname, '*' ) )

        for img_path in file_list:
            item_list.append( ( Image.open(img_path).convert("RGB"), classname ) )

    return item_list


def get_future_dataloader(cfg, candidate_data=None, train_x=None):
    tfm_train = build_transform(cfg, is_train=True)

    if candidate_data:

        candidate_dataloader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=candidate_data,
            batch_size=cfg.FUTURE.batch_size,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=FeatureDataset
        )

        return candidate_dataloader

    if train_x:

        gen_t_path = os.path.join( cfg.FUTURE.candidate_data_root, "generate_t.pkl" )
        with open(gen_t_path, "rb") as f:
                gen_t = pickle.load(f)

        gen_t_dataloader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=gen_t,
            batch_size=cfg.FUTURE.batch_size,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=None
        )

        train_dataloader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=train_x,
            batch_size=cfg.FUTURE.batch_size,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=None
        )

        return gen_t_dataloader, train_dataloader
