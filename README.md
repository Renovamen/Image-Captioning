# Image Caption

Pytorch re-implementation of some image caption models. Based on [sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning), thanks for this great work.

&nbsp;
## Model List

- `show_tell`

    **Show and Tell: A Neural Image Caption Generator.** *Oriol Vinyals, et al.* CVPR 2015. [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf) [[Code]](https://github.com/tensorflow/models/tree/master/research/im2txt)

- `att2all`

    **Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.** *Kelvin Xu, et al.* ICML 2015. [[Paper]](http://proceedings.mlr.press/v37/xuc15.pdf) [[Code]](https://github.com/kelvinxu/arctic-captions)


- `adaptive_att` & `spatial_att`

    **Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning.** *Jiasen Lu, et al.* CVPR 2017. [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lu_Knowing_When_to_CVPR_2017_paper.pdf) [[Code]](https://github.com/jiasenlu/AdaptiveAttention)

You can train different models by configuring `caption_model` in  [`config.py`](config.py).

&nbsp;

## Environment

- Python 3.6.5
- Pytorch 1.4.0 (along with torchvision)
- java 1.8.0 (only for computing METEOR)

&nbsp;

## Dataset

For dataset, I use [Flicker30k](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html) and [Karpathy's split](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). It is also okey to use [Flickr8k](https://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b) or [MSCOCO 2014](http://cocodataset.org/#download) (their splits and captions are also contained in Karpathy's split). If you want to use other datasets, you may have to create a JSON file which looks like Karpathy's JSON.

&nbsp;

## Usage

### Configuration

Configure parameters in  [`config.py`](config.py). Refer to this file for more information about each config item.


### Preprocess

First of all, you should preprocess the images along with their captions and store them locally:

```bash
python preprocess.py
```


### Train

To train a model, just run:

```bash
python train.py
```

### Evaluation

Compute evaluation metrics for a trained model on test set:

```bash
python eval.py
```

Now BLEU, CIDEr, METEOR and ROUGE-L are supported. Implementations of these metrics are under [`metrics`](metrics), which are adopted from [tylin/coco-caption](https://github.com/tylin/coco-caption).

During training, the BLEU-4 and CIDEr scores on validation set will be computed after each epoch's validation. However, since the decoder's input at each timestep is the word in ground truth captions, but not the word it generated in the previous timestep (Teacher Forcing), such scores does not reflect the real performance. So you can also consider about using this script to compute the correct scores for a specific trained model on validation set.


### Inference

This is for when you have trained a model and want to generate a caption (and visualize the attention weights, if the model contains an attention network) for a specific image:

First modify following things in [`inference.py`](inference.py):

```python
model_path = 'path_to_trained_model'
wordmap_path = 'path_to_word_map'
img = 'path_to_image'
beam_size = 5 # beam size for beam search
```

Then run:

```bash
python inference.py
```

&nbsp;
## Results

Here are some examples of the captions generated on images in test set. 

I haven't fine-tuned CNN. You'd probably want to try fine-tuning it to get better results.


### Adaptive Attention

#### Good Results
![adaptive-2](docs/adaptive-attention/success/2.png)

#### Okey Results

![adaptive-1](docs/adaptive-attention/success/1.png)Errors: two boys, not a chair...

#### Bad Results

![adaptive-2](docs/adaptive-attention/fail/1.png)

Error: not crying...


### Attention

#### Good Results

![attention-1](docs/attention/success/1.png)
![attention-2](docs/attention/success/2.png)

#### Okey Results

![attention-3](docs/attention/success/3.png)

#### Bad Results

![attention-3](docs/attention/fail/1.png)

Errors: not a woman, and seems to recognize sleeves as jeans...