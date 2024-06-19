import os
from dataclasses import dataclass

import torch

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        result = run_example("<OD>", self.processor, self.model, image)

        ontology_classes = self.ontology.classes()

        results = result["<OD>"]
        boxes_and_labels = list(zip(results["bboxes"], results["labels"]))

        detections = sv.Detections(
            xyxy=np.array(
                [box for box, label in boxes_and_labels if label in ontology_classes]
            ),
            class_id=np.array(
                [
                    ontology_classes.index(label)
                    for box, label in boxes_and_labels
                    if label in ontology_classes
                ]
            ),
            confidence=np.array(
                [1.0 for box, label in boxes_and_labels if label in ontology_classes]
            ),
        )

        detections = detections[detections.confidence > confidence]

        return detections
