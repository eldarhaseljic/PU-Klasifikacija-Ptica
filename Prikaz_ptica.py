# %%
from IPython import get_ipython
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
# %% [code]
dir=r'/kaggle/input/100-bird-species'
dir_list=os.listdir(dir)
print (dir_list)
# %% [code]
test_dir=r'/kaggle/input/100-bird-species/180/test'
classes=len(os.listdir(test_dir))
fig = plt.figure(figsize=(20,100))
if classes % 5==0:
    rows=int(classes/5)
else:
    rows=int(classes/5) +1
for row in range(rows):
    for column in range(5):
        i= row * 5 + column 
        if i>classes:
            break            
        specie=species_list[i]
        species_path=os.path.join(test_dir, specie)
        f_path=os.path.join(species_path, '1.jpg')        
        img = mpimg.imread(f_path)
        a = fig.add_subplot(rows, 10, i+1)
        imgplot=plt.imshow(img)
        a.axis("off")
        a.set_title(specie)