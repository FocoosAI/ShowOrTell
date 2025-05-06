<div align="center">

<h1>A Simple Image Segmentation Framework via In-Context Examples </h1>

[Yang Liu](https://scholar.google.com/citations?user=9JcQ2hwAAAAJ&hl=en)<sup>1</sup>, &nbsp; 
[Chenchen Jing](https://jingchenchen.github.io/)<sup>1</sup>, &nbsp;
Hengtao Li<sup>1</sup>, &nbsp;
[Muzhi Zhu](https://scholar.google.com/citations?user=064gBH4AAAAJ&hl=en)<sup>1</sup>, &nbsp;
[Hao Chen](https://stan-haochen.github.io/)<sup>1</sup>, &nbsp;
[Xinlong Wang](https://www.xloong.wang/)<sup>2</sup>, &nbsp;
[Chunhua Shen](https://cshen.github.io/)<sup>1</sup>

<sup>1</sup>[Zhejiang University](https://www.zju.edu.cn/english/), &nbsp;
<sup>2</sup>[Beijing Academy of Artificial Intelligence](https://www.baai.ac.cn/english.html)

NeurIPS 2024

</div>

## 🚀 Overview
<div align="center">
<img width="800" alt="image" src="figs/framework.png">
</div>

## 📖 Description

### Overview
- This paper proposes a simple yet effective image segmentation framework that leverages in-context examples.
- The approach allows users to provide a few annotated examples within an image, which the model then uses to segment the rest of the image.
- The framework is designed to be intuitive and user-friendly, enabling non-expert users to perform accurate image segmentation.

- In detail:
Recently, there have been explorations of generalist segmentation models that can effectively tackle a variety of image segmentation tasks within a unified in-context learning framework. 
However, these methods still struggle with task ambiguity in in-context segmentation, as not all in-context examples can accurately convey the task information. 
In order to address this issue, we present SINE, a simple image **S**egmentation framework utilizing **in**-context **e**xamples. 
Our approach leverages a Transformer encoder-decoder structure, where the encoder provides high-quality image representations, and the decoder is designed to yield multiple task-specific output masks to effectively eliminate task ambiguity.

[Paper](https://arxiv.org/abs/2410.04842)


## 👻 Getting Started

- [Training](TRAINING.md). 

- DINOv2-L model trained on ADE20K, COCO, and Objects365, [weight](https://drive.google.com/file/d/1GYQbbUZClbmhVESDLpRwqe-TyijW2kKb/view?usp=sharing).

- [Evaluation - Few-shot Semnatic Segmentation](inference_fss/EVALUATION.md)

- [Evaluation - Few-shot Instance Segmentation](inference_fsod/EVALUATION.md)

- [Evaluation - Video Object Segmentation](inference_vos/EVALUATION.md)



## 🎫 License

For academic use, this project is licensed under [the 2-clause BSD License](LICENSE). For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).

## 🖊️ Citation


If you find this project useful in your research, please consider to cite:


```BibTeX
@article{liu2024simple,
  title={A Simple Image Segmentation Framework via In-Context Examples},
  author={Liu, Yang and Jing, Chenchen and Li, Hengtao and Zhu, Muzhi and Chen, Hao and Wang, Xinlong and Shen, Chunhua},
  journal={Proc. Int. Conference on Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

## Acknowledgement
[DINOv2](https://github.com/facebookresearch/dinov2), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [SegGPT](https://github.com/baaivision/Painter/tree/main/SegGPT), [Matcher](https://github.com/aim-uofa/Matcher), [TFA](https://github.com/ucbdrive/few-shot-object-detection) and [detectron2](https://github.com/facebookresearch/detectron2).

## FAQ
### Key Contributions of the Paper:
- The paper is the first to investigate and address task ambiguity in in-context segmentation.
- It introduces a Matching Transformer that unlocks the potential of frozen pre-trained image models for diverse segmentation tasks with low training costs.

### What is the main challenge in in-context segmentation that SINE aims to address?
- The primary challenge SINE addresses is task ambiguity in in-context segmentation. This ambiguity arises when the in-context examples do not accurately or clearly convey the intended segmentation task. For instance, if the reference image only shows a single object and its annotation, the lack of additional task-related information can lead to incorrect segmentation outputs.

### How does SINE address task ambiguity?
- SINE tackles task ambiguity by predicting multiple output masks, each customized for tasks of varying complexity, ranging from identifying identical objects to instances and overall semantic concepts. This approach allows SINE to disentangle the specific task from the in-context example and interpret the semantic meaning of the prompts to produce results at different levels of task granularity.

### How does SINE compare to SegGPT, another in-context segmentation model?
- Both SINE and SegGPT are in-context segmentation models, but SINE offers several advantages:
Addressing task ambiguity: SINE can handle task ambiguity by generating multiple task-specific output masks, while SegGPT is limited to semantic segmentation and cannot resolve such ambiguities.
- Handling instance segmentation: SINE can perform instance segmentation, a capability lacking in SegGPT.
- Direct mask prediction: SINE directly predicts segmentation masks, avoiding the complex post-processing steps required by SegGPT to convert its RGB pixel output to masks.
- Handling high-resolution images: Unlike SegGPT, which stitches the reference and target images, SINE processes them separately, eliminating limitations in processing high-resolution images.

### What are the limitations of SINE?
- Limited scope of ambiguity resolution: SINE primarily focuses on addressing ambiguities between ID, instance, and semantic segmentation tasks. More complex ambiguities, such as those related to object parts, spatial positions, categories, and colors, are not explicitly addressed. Future work could incorporate multimodal in-context examples (e.g., image and text) to tackle these more intricate ambiguities.
- Performance gap with SegGPT: SINE exhibits a performance gap compared to SegGPT, particularly in handling complex video sequences. This gap is attributed to SINE's use of fewer trainable parameters and a simpler In-context Interaction module, limiting its ability to capture complex inter-frame relationships. Designing a more sophisticated In-context Interaction module is a potential avenue for improvement.
