import os
import copy
from dataclasses import dataclass
import json
from typing import Dict,  Sequence

from PIL import Image, ImageFile
import transformers
import torch
from torch.utils.data import Dataset, Subset

from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *


ImageFile.LOAD_TRUNCATED_IMAGES = True


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        # Create unique_idx for each dictionary in the list
        if "unique_idx" not in self.list_data_dict[0]:
            if not self.create_and_check_unique_indices():
                raise ValueError(
                    "Failed to create unique indices. "
                    "Checking the unique idx logic!"
                )
            else:
                print("Successfully created unique_idx!")
        else:
            print("Unique_idx already exists!")
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def create_and_check_unique_indices(self):
        # Dictionary to store the unique_idx values and their occurrences
        unique_indices = {}

        # Dictionary to count instances by image source
        source_counts = {}

        # Counter for handling duplicates
        duplicate_counters = {}

        # Generate unique_idx for each dictionary in the list
        for i, data_dict in enumerate(self.list_data_dict):
            # Create base unique_idx
            if "image" in data_dict.keys():
                base_unique_idx = (
                    str(data_dict["id"]) + "::" + data_dict["image"]
                )

                if "task_name" in data_dict.keys():  # For Vision-Flan
                    image_source = data_dict["task_name"]
                else:
                    # Extract image source (first name in the path)
                    image_path = data_dict["image"]
                    image_source = image_path.split("/")[0]
                    if image_source not in [
                        "coco", "vg", "gqa", "ocr_vqa", "textvqa", "sharegpt"
                    ] and image_source[0].isdigit():
                        image_source = "coco"
            else:
                base_unique_idx = str(data_dict["id"])
                image_source = "sharegpt"
            data_dict["image_source"] = image_source

            # Check if this base_unique_idx already exists
            if base_unique_idx in unique_indices:
                # If it's a duplicate, add a counter
                if base_unique_idx not in duplicate_counters:
                    duplicate_counters[base_unique_idx] = 1

                    # Update the first occurrence with a counter of 0.
                    first_pos = unique_indices[base_unique_idx][0]
                    self.list_data_dict[first_pos]["unique_idx"] = (
                        f"{base_unique_idx}#0"
                    )

                # Increment counter for this duplicate
                duplicate_counters[base_unique_idx] += 1

                # Create a truly unique ID with counter.
                unique_idx = (f"{base_unique_idx}#"
                              f"{duplicate_counters[base_unique_idx] - 1}")
            else:
                unique_idx = base_unique_idx

            # Store the generated unique_idx
            data_dict["unique_idx"] = unique_idx

            # Track occurrences
            if base_unique_idx in unique_indices:
                unique_indices[base_unique_idx].append(i)
            else:
                unique_indices[base_unique_idx] = [i]

            # Count instances by image source
            if image_source in source_counts:
                source_counts[image_source] += 1
            else:
                source_counts[image_source] = 1

        # Print summary of source counts
        print("Instance counts by image source:")
        for source, count in source_counts.items():
            print(f"  {source}: {count} instances")

        # Verify final uniqueness
        final_unique_ids = {}
        for i, data_dict in enumerate(self.list_data_dict):
            if data_dict["unique_idx"] in final_unique_ids:
                print(
                    f"\nError: Still found duplicate unique_idx "
                    f"'{data_dict['unique_idx']}'"
                )
                return False
            final_unique_ids[data_dict["unique_idx"]] = i

        print("\nAll unique_idx values are now unique!")
        return True

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(
            copy.deepcopy(sources["conversations"])
        )
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = self.image_preprocess(image)
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        data_dict['unique_idx'] = sources['unique_idx']
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        unique_indices = [instance['unique_idx'] for instance in instances]
        batch['unique_indices'] = unique_indices
        return batch


@dataclass
class DataCollatorForSupervisedDatasetLLA(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        unique_indices = [instance['unique_idx'] for instance in instances]
        batch['unique_indices'] = unique_indices

        #####################################
        ##### Hardcoded this for now ########
        #####################################
        prompt_length = 31
        image_idx = (batch['input_ids'] == IMAGE_TOKEN_INDEX).sum(dim=-1)
        language_length = torch.tensor([batch['attention_mask'][idx][prompt_length+1:].sum() if image_idx[idx] != 0 else
                                        batch['attention_mask'][idx][prompt_length:].sum() for idx in range(input_ids.shape[0])])
        batch['language_length'] = language_length

        return batch



def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        return_eval_dataset=False,
        last_layer_act=False,
        subset_size=None
    ) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    if subset_size is not None:
        train_dataset = Subset(train_dataset, list(range(subset_size)))
    if not last_layer_act:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else: 
        data_collator = DataCollatorForSupervisedDatasetLLA(tokenizer=tokenizer)
    if not return_eval_dataset:
        return dict(train_dataset=train_dataset,
                    eval_dataset=None,
                    data_collator=data_collator)
    else: 
        return dict(train_dataset=train_dataset,
                    eval_dataset=train_dataset,
                    data_collator=data_collator)