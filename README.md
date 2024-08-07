<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png?3"
      >
    </a>
  </p>
</div>

# Autodistill Florence 2 Module

This repository contains the code supporting the CLIP base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[Florence 2](https://huggingface.co/microsoft/Florence-2-large), introduced in the paper [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242) is a multimodal vision model.

You can use Florence 2 to generate object detection annotations for use in training smaller object detection models with Autodistill.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [Florence 2 Autodistill documentation](https://autodistill.github.io/autodistill/base_models/florence2/).

## Installation

To use Florence 2 with Autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-florence-2
```

## Quickstart (Inference from Base Weights)

```python
from autodistill_florence_2 import Florence2
from autodistill.detection import DetectionOntology
from PIL import Image
import supervision as sv

# define an ontology to map class names to our Florence 2 prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = Florence2(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)

image = Image.open("image.jpeg")
result = base_model.predict('image.jpeg')

bounding_box_annotator = sv.BoundingBoxAnnotator()
annotated_frame = bounding_box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
sv.plot_image(image=annotated_frame, size=(16, 16))

# label a dataset
base_model.label("./context_images", extension=".jpeg")
```

## Quickstart (Fine-Tune)

```python
from autodistill_florence_2 import Florence2Trainer

model = Florence2Trainer("dataset")
model.train(dataset.location, epochs=10)
```

## License

This project is licensed under an MIT license. See the [Florence 2 license](https://huggingface.co/microsoft/Florence-2-large) for more information about the Florence 2 model license.

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
