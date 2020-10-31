# Importance Net
ImportanceNet is a network that estimates the relative distribution of a domain for unsupervised domain transfer tasks.

Image-to-image translation methods faithfully transform a source image to the style of a target domain, enabling a breadth of applications in domain adaptation and arts. Existing approaches focus on image quality, little attention is given to the distribution of generated images, which by default follows that of the source domain. We propose a method to re-sample a set of generated images to match the distribution of the target domain. At the core is a NN-module for estimating the relative frequency of image constellations in the source and target domains by matching modes of features.

To run the example we move to the example folder:
```bash
cd cycleGAN_example
```

To train the cycleGAN we can run:
```bash 
python3 train.py --train_GAN
```

Next to train importanceNet using the pretrained cycleGAN we run:
```bash
python3 train.py --continue_train --train_W
```

---------------

The folder `importance_net` in the cycleGAN example contains the network that performs the distribution estimation. To run within your own training loop simply add the pieces of code that are surrounded with `#############` in train.py.

The imports:
```python
from importance_net.models import create_model, importance_model, network
from importance_net.options.train_options import TrainOptions as ImportanceTrainOptions

importance_opt = ImportanceTrainOptions.parse()
```

Initialize the network:
```python
objective_function = network.DiscriminatorLoss(model=model, gan_mode=opt.gan_mode)
importance_model = importance_model.ImportanceModel(importance_opt, objective_function.criterion)
importance_model.setup(importance_opt) # setup using options
```

And run within your training loop:
```python
importance_model.set_input(data)         # unpack data from dataset and apply preprocessing
importance_model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
```
