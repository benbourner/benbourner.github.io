---
layout: post
title: Emergency Vehicle Classifier
subtitle: "My AI Journey with Dutch Emergency Vehicles"
cover-img: /assets/img/emergency.jpeg
thumbnail-img: /assets/img/vehicles-3.png
share-img: /assets/img/emergency.jpeg
tags: [AI, machine learning, emergency vehicles, fastai, gradio, Python, Hugging Face, image recognition, Dutch emergency services]
author: Ben Bourner
---

Using [fastai](https://github.com/fastai/fastai), a powerful Python AI library, I've created and trained an AI model — a specialized tool that can accurately distinguish between various Dutch emergency vehicles, such as police cars, ambulances, and fire engines.

## The Inspiration

The idea was to test out fast AI model training abilities by teaching it to be able to distinguish images of multiple categories.

## The Journey

I chose the [fastai library](https://github.com/fastai/fastai) for its user-friendly interface and robust capabilities in handling complex tasks like image recognition. The journey wasn't easy, involving countless hours of dataset preparation, model training, and testing. My dataset comprised hundreds of images of Dutch emergency vehicles, each meticulously labeled and categorized.

## The Breakthrough

After several iterations and much fine-tuning, the model began showing promising results. It could not only distinguish between the different types of vehicles but do so with remarkable accuracy. This breakthrough was not just a technical accomplishment; it was a moment of realisation about the potential impact of AI in critical areas like emergency response.

## The Deployment

The model is now live, presented with [Gradio](https://www.gradio.app/) - a fast way to demo a machine learning model with a friendly web interface, and hosted on [Hugging Face](https://huggingface.co/) - a leading platform for machine learning models. This represents my first deployment of an AI model to production — a significant step for an AI enthusiast and developer.

## Try It Out Live!

I invite you to experience the model firsthand. You can try it out live below. Upload an image of a Dutch emergency vehicle (or choose one of the examples), and watch the AI work its magic!

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/4.13.0/gradio.js"></script>
<gradio-app src="https://benboai-vehicle-checker.hf.space" theme_mode="light"></gradio-app>

## How it was done

Fast AI makes things pretty simple. Training a model requires providing a few dozen well-labelled image examples modified with data augmentation,  then loading the data into a learner to fine tune a base model over multiple iterations:

```python
from fastai import *

data = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))

vehicles = vehicles.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())

dls = vehicles.dataloaders(path)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
```

This performs 4 iterations fine tuning the model and becomes more accurate each time:

|epoch |	train_loss	| valid_loss | error_rate |time| 
|-----|----|---|---|--|
| 0|	1.235733	|0.212541|	0.087302|	00:05|
1|	0.173855|	0.072306|	0.023810|	00:06
2|	0.147096|	0.039068|	0.015873|	00:06
3|	0.123984|	0.026801|	0.015873|	00:06

Once the model is fine tuned, it is moved to another program which can be used to make predictions of category based on image inputs. This is what you see above in the live demo.


## Conclusion

This project is just the beginning. It represents a blend of technology and social impact, showcasing how AI can play a pivotal role in critical sectors like emergency services. I look forward to exploring further and contributing more to this fascinating field of AI.

Thank you for sharing this journey with me!

---

*This post reflects my personal experience and journey in the field of AI and is not affiliated with any official entity or organization.*

---

[Back to top](#top)
