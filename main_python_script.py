# %% [code]
#Importing
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from PIL import Image

# %% [code]

loc = '/kaggle/input/100-bird-species/test/ANHINGA/2.jpg'
im = Image.open(loc)  
image_gr = im.convert("L")    # convert("L") translate color images into black and white
                              # uses the ITU-R 601-2 Luma transform (there are several 
                              # ways to convert an image to grey scale)
print("\n Original type: %r \n\n" % image_gr)

# convert image to a matrix with values from 0 to 255 (uint8) 
arr = np.asarray(image_gr) 
print("After conversion to numerical representation: \n\n %r" % arr)

# %% [code]

### Activating matplotlib for Ipython

### Plot image

plt.figure(1,figsize=(12,8))
img = mpimg.imread(loc)
plt.imshow(img)
plt.title('original image');
plt.axis('off');
# %% [code]
plt.figure(1,figsize=(12,8))
imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')  #you can experiment different colormaps (Greys,winter,autumn)
print("\n Input image converted to gray scale: \n")
plt.axis('off')
plt.show(imgplot)
# %% [markdown]
# ### Now lets make a conv kernel
# %% [code]
kernel = np.array([[ 0, 1, 0],
                   [ 1,-4, 1],
                   [ 0, 1, 0],]) 

grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')
print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 8))
aux.imshow(np.absolute(grad), cmap='gray');
# %% [code]
type(grad)

grad_biases = np.absolute(grad) + 100

grad_biases[grad_biases > 255] = 255
plt.figure(1,figsize=(18,8))
plt.imshow(np.absolute(grad_biases),cmap='gray');
# %% [code]
from fastai.vision import *
import numpy as np # linear algebra
import pandas as pd
# %% [code]
path = Path('/kaggle/input/100-bird-species/consolidated/')
tfms = get_transforms(do_flip=True,max_lighting=0.1,max_rotate=0.1)
data = (
        ImageDataBunch.from_folder
            (
                path,train='.', valid_pct=0.15,
                ds_tfms=tfms,size=224,num_workers=4
            ).normalize(imagenet_stats)
        )  
# valid size here its 15% of total images,
# train = train folder here we use all the folder
# from_folder take images from folder and labels them like wise
data.show_batch(rows=5)
len(data.classes), len(data.train_ds), len(data.valid_ds)
# %% [code]
fb = FBeta()
fb.average = 'macro'
# We are using fbeta macro average in case some
# class of birds have less train images
learn = cnn_learner(data, models.resnet18, 
                    metrics=[error_rate,fb],
                    model_dir='/kaggle/working/')
# %% [code]
learn.lr_find()
learn.recorder.plot()
# %% [code]
lr = 1e-2 # learning rate
learn.fit_one_cycle(5,lr,moms=(0.8,0.7))  # moms
# %% [code]
interp = ClassificationInterpretation.from_learner(learn)

# %% [code]
interp.plot_top_losses(12,figsize=(20,8))
# %% [code]
interp.most_confused(min_val=3)
# %% [code]
file = '/kaggle/input/100-bird-species/predictor test set/013.jpg'
img = open_image(file)  
# open the image using open_image func from fast.ai
print(learn.predict(img)[0]) 
# lets make some prediction
img
# %% [code]
file = '/kaggle/input/100-bird-species/predictor test set/049.jpg'
img = open_image(file)  
# open the image using open_image func from fast.ai
print(learn.predict(img)[0]) 
# lets make some prediction
img
# %% [code]
file = '/kaggle/input/100-bird-species/predictor test set/075.jpg'
img = open_image(file)  
# open the image using open_image func from fast.ai
print(learn.predict(img)[0]) 
# lets make some prediction
img