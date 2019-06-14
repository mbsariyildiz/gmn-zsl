This is the official repository for the **Gradient Matching Generative Networks for Zero-Shot Learning** paper 
published at IEEE CVPR 2019.

Paper link: [CVPR Open Access](http://openaccess.thecvf.com/content_CVPR_2019/html/Sariyildiz_Gradient_Matching_Generative_Networks_for_Zero-Shot_Learning_CVPR_2019_paper.html) </br>

Bibtex entry:
```
@InProceedings{Sariyildiz_2019_CVPR,
	author = {Bulent Sariyildiz, Mert and Gokberk Cinbis, Ramazan},
	title = {Gradient Matching Generative Networks for Zero-Shot Learning},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2019}
} 
```

After the camera ready deadline, we run several more experiments for a follow-up work. We simply enlarged the hyper-parameter space, for instance,
(i) we further tuned the beta parameters of the ADAM optimizer and some of the GMN parameters in GMN training 
(ii) replaced ReLUs with LeakyReLUs in the generator and the discriminator networks,
(iii) decayed learning rates during classifier training stages.
Naturally, with these changes we obtained slightly higher scores on the CUB, SUN and AWA datasets.
Besides, we found the followings:
- _attribute concatenation_ works better than modeling _latent noise spaces_ on the SUN dataset.
- When tuned to maximize the harmonic mean score, a linear classifier performs slightly better than a bilinear multi-modal embedding classifier in the AWA and SUN datasets.
Therefore, to be fairly comparable with the SOTA approaches on the SUN dataset, we give our best results on the SUN dataset which are obtained by the attribute
concatenation method and a linear classifier as in Felix et al. ECCV-2018 or Xian et al. CVPR-2018.
We will publish our latest results on a follow-up work, meanwhile, 
we suggest practitioners to take the results reported in this repository as a baseline when comparing GMN with their own approaches.

The scripts provided in this repository are designed to directly re-produce the scores that we give below.

## Zero-Shot Learning Scores
|             | CUB  | SUN  | AWA  |
|---------    |------|------|------|
| **Updated** | 67.0 | 61.1 | 72.0 |
| Paper       | 64.3 | 63.6 | 71.9 |

## Generalized Zero-Shot Learning Scores
|             |  CUB |      |      |  SUN |      |      | AWA  |      |      |
|---------    |------|------|------|------|------|------|------|------|------|
|             | u    | s    | h    | u    | s    | h    | u    | s    | h    |
| **Updated** | 54.7 | 58.4 | 56.5 | 50.3 | 37.2 | 42.8 | 57.1 | 81.3 | 67.1 |
| Paper       | 56.1 | 54.3 | 55.2 | 53.2 | 33.0 | 40.7 | 61.1 | 71.3 | 65.8 |

