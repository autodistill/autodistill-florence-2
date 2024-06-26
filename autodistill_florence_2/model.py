import os
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
import torch
from autodistill.detection import (CaptionOntology, DetectionBaseModel,
                                   DetectionTargetModel)
from autodistill.helpers import load_image
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DetectionsDataset(Dataset):
    def __init__(self, dataset: sv.DetectionDataset):
        self.dataset = dataset
        self.keys = list(dataset.images.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        keys = list(self.dataset.images.keys())
        key = self.keys[idx]
        image = self.dataset.images[key]
        annotations = self.dataset.annotations[key]
        h, w, _ = image.shape

        boxes = (annotations.xyxy / np.array([w, h, w, h]) * 1000).astype(int).tolist()
        labels = [self.dataset.classes[idx] for idx in annotations.class_id]

        prefix = "<OD>"

        suffix_components = []
        for [x1, y1, x2, y2], label in zip(boxes, labels):
            suffix_component = f"{label}<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"
            suffix_components.append(suffix_component)

        suffix = "".join(suffix_components)

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        return prefix, suffix, image


def run_example(task_prompt, processor, model, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )

    return parsed_answer


@dataclass
class Florence2(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        model_id = "microsoft/Florence-2-large"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, device_map="cuda"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, device_map="cuda"
        )
        self.ontology = ontology

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        image = load_image(input, return_format="PIL")
        ontology_classes = self.ontology.classes()
        result = run_example(
            "<CAPTION_TO_PHRASE_GROUNDING>",
            self.processor,
            self.model,
            image,
            "A photo of " + ", and ".join(ontology_classes) + ".",
        )

        results = result["<CAPTION_TO_PHRASE_GROUNDING>"]

        boxes_and_labels = list(zip(results["bboxes"], results["labels"]))

        if (
            len(
                [
                    box
                    for box, label in boxes_and_labels
                    if label in ontology_classes and ontology_classes
                ]
            )
            == 0
        ):
            return sv.Detections.empty()

        detections = sv.Detections(
            xyxy=np.array(
                [
                    box
                    for box, label in boxes_and_labels
                    if label in ontology_classes and ontology_classes
                ]
            ),
            class_id=np.array(
                [
                    ontology_classes.index(label)
                    for box, label in boxes_and_labels
                    if label in ontology_classes and ontology_classes
                ]
            ),
            confidence=np.array(
                [
                    1.0
                    for box, label in boxes_and_labels
                    if label in ontology_classes and ontology_classes
                ]
            ),
        )

        detections = detections[detections.confidence > confidence]

        return detections


class Florence2Trainer(DetectionTargetModel):
    def __init__(
        self,
        checkpoint: str = "microsoft/Florence-2-base-ft",
    ):
        REVISION = "refs/pr/6"
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, trust_remote_code=True, revision=REVISION
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True, revision=REVISION
        )

        self.model = model
        self.processor = processor
        self.REVISION = REVISION

    def predict(self, input: str, confidence=0.5) -> sv.Detections:
        image = Image.open(input)
        task = "<OD>"
        text = "<OD>"

        inputs = self.processor(text=text, images=image, return_tensors="pt").to(DEVICE)
        generated_ids = self.peft_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        response = self.processor.post_process_generation(
            generated_text, task=task, image_size=(image.width, image.height)
        )
        detections = sv.Detections.from_lmm(
            sv.LMM.FLORENCE_2, response, resolution_wh=image.size
        )

        return detections

    def train(self, dataset_path, epochs=10):
        ds_train = sv.DetectionDataset.from_coco(
            images_directory_path=f"{dataset_path}/train",
            annotations_path=f"{dataset_path}/train/_annotations.coco.json",
        )

        ds_valid = sv.DetectionDataset.from_coco(
            images_directory_path=f"{dataset_path}/valid",
            annotations_path=f"{dataset_path}/valid/_annotations.coco.json",
        )

        BATCH_SIZE = 6
        NUM_WORKERS = 0

        def collate_fn(batch):
            questions, answers, images = zip(*batch)
            inputs = self.processor(
                text=list(questions),
                images=list(images),
                return_tensors="pt",
                padding=True,
            ).to(DEVICE)
            return inputs, answers

        train_dataset = DetectionsDataset(ds_train)
        val_dataset = DetectionsDataset(ds_valid)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
        )

        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=[
                "q_proj",
                "o_proj",
                "k_proj",
                "v_proj",
                "linear",
                "Conv2d",
                "lm_head",
                "fc2",
            ],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
            revision=self.REVISION,
        )

        peft_model = get_peft_model(self.model, config)
        peft_model.print_trainable_parameters()
        self.peft_model = peft_model

        torch.cuda.empty_cache()

        EPOCHS = 10
        LR = 5e-6

        optimizer = AdamW(self.model.parameters(), lr=LR)
        num_training_steps = epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        for epoch in range(EPOCHS):
            self.model.train()
            train_loss = 0
            for inputs, answers in tqdm(
                train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"
            ):

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = self.processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).input_ids.to(DEVICE)

                outputs = self.model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss

                loss.backward(), optimizer.step(), lr_scheduler.step(), optimizer.zero_grad()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"Average Training Loss: {avg_train_loss}")

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, answers in tqdm(
                    val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
                ):

                    input_ids = inputs["input_ids"]
                    pixel_values = inputs["pixel_values"]
                    labels = self.processor.tokenizer(
                        text=answers,
                        return_tensors="pt",
                        padding=True,
                        return_token_type_ids=False,
                    ).input_ids.to(DEVICE)

                    outputs = self.model(
                        input_ids=input_ids, pixel_values=pixel_values, labels=labels
                    )
                    loss = outputs.loss

                    val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                print(f"Average Validation Loss: {avg_val_loss}")

            output_dir = f"./model_checkpoints/epoch_{epoch+1}"
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.processor.save_pretrained(output_dir)
