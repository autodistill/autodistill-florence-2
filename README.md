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

## Quickstart

```python
from autodistill_florence_2 import Florence2

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
base_model.label("./context_images", extension=".jpeg")
```

## License

This project is licensed under an MIT license. See the [Florence 2 license](https://huggingface.co/microsoft/Florence-2-large) for more information about the Florence 2 model license.

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!