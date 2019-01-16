# 100 Days of ML Code

[Inspired](https://github.com/llSourcell/100_Days_of_ML_Code) by Siraj Raval.

---
## Important Notes

1. [How to run](./jupyter-guide.md) Jupyter Notebook on a remote server and access via ssh on a local machine
2. [Tmux cheatsheet](./tmux-cheatsheet.md)

---

## Day 7. January 16, Wednesday.

Today I finished fastai part 1 v3 lesson 2.

I also set up repos for development of fastai and fastai_docs so that I can contribute to these libraries. There are some standards of contribution, such as stripping notebooks of outputs when committing, the instructions are:

- for [`fastai`](https://docs.fast.ai/dev/develop.html)
- for [`fastai_docs`](https://github.com/fastai/fastai_docs)

## Day 6. January 15, Tuesday.

Today I watched fastai part one, version 3 lecture 2.

### Tips

- `doc(fastai_name)` - nice documentation for any fastai function

### Related Links

- [Ethan Sutin](https://twitter.com/ethansutin) - created a SOTA spectrogram classifier
- [Audacity](https://www.audacityteam.org/) - free audio editor
- [The difference between GitHub repos and gists](https://stackoverflow.com/questions/6767518/what-is-the-difference-between-github-and-gist)
- [Gist it](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/gist_it/readme.html) - make gists from Jupyter Notebooks
- [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/) - create your own Jupyter Notebook widgets [AWESOME]
- [The Jupyter Widget Ecosystem](https://github.com/jupyter-widgets/tutorial) - Tutorial, SciPy 2018
- [Share your work here](https://forums.fast.ai/t/share-your-work-here/27676) - creative applications of deep learning from students of fastai part 1 v3 [access only for fastai international fellows]
- [The Mystery of the Origin — Cancer Type Classification using Fast.AI Library](https://towardsdatascience.com/the-mystery-of-the-origin-cancer-type-classification-using-fast-ai-libray-212eaf8d3f4e) [blog post by Alena Harley]
- Python's [`await`](https://docs.python.org/3/library/asyncio-task.html)
- [Starlette](https://www.starlette.io/) web app framework
- [Matrix Multiplicaion XYZ](http://matrixmultiplication.xyz/)

## Day 5. January 12, Saturday.

### Reading

Today is a reading day. Topics related to data science and careers in data science.

A very nice blog by Tim Hopper:

1. [How I Quit My Ph.D. and Learned to Love Data Science](https://tdhopper.com/blog/how-i-quit-my-ph.d.-and-learned-to-love-data-science/)
2. [How I Became a Data Scientist Despite Having Been a Math Major](https://tdhopper.com/blog/how-i-became-a-data-scientist/)
3. [I Basically Can’t Hire People Who Don’t Know Git](https://tdhopper.com/blog/data-scientists-at-work/)
4. [A Subjective and Anecdotal FAQ on Becoming a Data Scientist](https://tdhopper.com/blog/faq/)

### Coding

Today I fixed the issue with train-validation-test split used for `.from_csv()` in `fastai` library in my [102 flowers classifier]().

I made folders so that I could have dataset as per [ImageDataBunch.from_csv()](https://docs.fast.ai/vision.data.html#ImageDataBunch.from_csv).

We will have to have structure like this:

```
path/
  /train
  /test
  labels.csv
```

In addition, `labels.csv` has to have structure like:

```
filename_1, category
filename_2, category
       ...,      ...
filename_n, category
```

**Notes**:

- I looked at [source code](https://github.com/fastai/fastai/blob/master/fastai/vision/data.py#L132) to figure out the proper way to set up data set structure. Basically, `.from_csv` calls `.from_df()` under the hood. And that calls `.create_from_ll()` which in itself sets up the test folder properly. Test folder parameter is passed through as `**kwargs` all the way from `.from_csv()`.

- `test` parameter should be a folder of images. By default `labels.csv` will be used as train data with random split given by the parameter `valid_pct`. So I used both `train.txt` and `valid.txt` coming with the original 102 flowers data set to generate `labels.csv`. Then I used `test.txt` to generate `labels_test.csv` that can be used for testing later. Note that `labels_test.csv` is not used in training directly, it can only be used later to assess out-of-sample performance of the model.

- `folder` parameter is the root directory realative to which paths in `labels.csv` will be loaded as data. By default it is `.` which is current directory.

- after the proper split I have 2039 images for train and validation. With the default 20% validation this means 1627 images for training and 412 images for validation. Official test set for this data set is 6149 images. Because the train set is so small compared to improperly used yesterday, my today's best accuracy is about 8%. I can easily make it much better by using some data from test data set, but for now this is a good result.

## Day 4. January 11, Friday.

I finished my skin lesion classifier. Error rate about 7%. But true positive rate about 60% which is pretty low. Probably class imbalance. I will come back to this project later when I learn how to make a custom more balanced data set feeder with fastai.

I then started another mini project as homework for fastai v3 lesson 1 - [102 flower classifier](https://github.com/pechyonkin/flowers-classifier). Spend quite some time to sort the files into appropriate folders so that folder structure can be used by `ImageDataBunch.from_csv()`. Documentation is not very clear, test folder was used as training data, validation set was randomly created automatically even though one of the folders was called `valid/`. I will try to fix it tomorrow.

By the way, the data set is quite bad - the authors did not bother to provide a mapping from numerical class labels to actual class names. [This](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html) is the best they could come up with. It is not even in alphabetic order! Maybe it is a riddle where someone who wishes to use the dataset must manually label the data by using photograph count provided. Disappointment.

If you ever create your own data set - do not do it like this. People will thank you.

**TODO**: make a cheatsheet with common file system commands in Python: copy file, make file, create and close directories, check existence etc. All these functions are in different packages which makes is quite messy and hard to remember well.

### Nice Discovery

**TODO**: [This](https://github.com/ageron/tf2_course) tutorial on the upcoming TensorFlow 2.0 by Aurelien Geron.

## Day 3. January 10, Thursday.

Today I was able to run the pretrained fastai model, as per Lesson 1 instructions. At first I got an error and couldn't figure it out but then realized I had a parameter wrong, I used `tfms` instead of `ds_tfms` when creating my dataset object.

Then, training took too long. About 23000 images, 15 minutes per epoch for finetuning on resnet50. That is too slow, as it should take about 2-4 minutes, by my expectation. I then realized my CPU was loaded 100% and GPU power draw was only 80 watts out of 250 watts maximum. Clearly this is a CPU bottleneck, probably doing on the fly data augmentation and resizing. The images are all quite large and all different sizes, so this maybe is a contributing factor.

Maybe I will try to look into ways to optimize this - one way is to resize all images to 300 by 300 and then use those resized images during training.

On the good side, I was able to get to 7.85% error rate. I will try to do more tomorrow.

I also found a little bug in fastai part one version 3 course, I submitted a [pull request](https://github.com/fastai/course-v3/pull/138). Who knows, maybe it will get accepted.


## Day 2. January 9, Wednesday

Today I spent time additionally cleaning the data for skin lesion classifier. I decided to only go with 2 classes for now - malignant and benign.

I split the data into 3 groups Imagenet-style (each train, validation and test in their own folder, each has folders for each existing class):

- train - 80%
- validation - 10%
- test - 10%

Then I copied all the files into their corersponding folders and transferred them to the DL box to try fastai v3 lesson 1. I discovered that my virtual environment broke down so I had to spent time to reinstall it.

I am ready now to try and build a transfer learning classifier tomorrow, because my data is ready.

### Useful snippets from today

It turns out broadcasting works in Pandas similar to Numpy. Below is example of creating filenames with extension based on filenames and a given extension.

```python
df5['fname'] = df5['name'] + '.jpeg'
```

Dropping a column:

```python
df6.drop('name', axis=1, inplace=True)
```
Random integers with given probabilities:

```python
split_ints = np.random.choice([0,1,2], size=df6.shape[0], p=[0.8,0.1,0.1])
```
 Get a directory name based on a path string:
 
 ```python
 os.path.dirname('/Users/fazzl/data/isic/train/benign/ISIC_0026373.jpeg')
 ```

Remove the last 5 chars in a Pandas column of strings, I tried lambda and it unexpectedly worked as expected immidiately. Love such moments! Note: .stem and .name are parts of pathlib's Path.

- [`.stem`](https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.stem) - the final path component, without its suffix - either a filename without extension or the last folder name
- [`.name`](https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.name) - finlename in a given path, if any

```python
# map from filename without extension to filename with extension
images_fnames = {f.stem : f.name for f in Path('/Users/fazzl/data/isic-source/images').iterdir()}

# fix the names of files, some of the files are PNG
df6['fname'] = df6['fname'].apply(lambda x: images_fnames[x[:-5]])
```

Loop to copy files to produce Imagenet-style structure, where folder existence is checked at every iteration. Maybe not the most efficient way, but it worked well:

```python
for idx in df6.index:
    fname = df6.loc[idx]['fname']
    path = df6.loc[idx]['path']
    source_fname = SOURCE_PATH + fname
    target_fname = TARGET_PATH + path
    target_dir = os.path.dirname(target_fname)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    shutil.copy2(source_fname, target_fname)
```


## Day 1. January 8, Tuesday

Today I worked on cleaning [ISIC](https://isic-archive.com) skin cancer data for my [skin-lesion-classifier](https://github.com/pechyonkin/skin-lesion-classifier). Got more familiar with using Pandas to clean data.

### Some useful snippets are below

Combining 2 boolean masks to change a value:
```python
mask_null = df2['benign_malignant'].isnull()
mask_pgk = df2['diagnosis'] == 'pigmented benign keratosis'
df2.loc[mask_null & mask_pgk, 'benign_malignant'] = 'benign'
```

Drop rows with `NaNs` from the dataframe:
```python
idx_to_drop = df2[df2['benign_malignant'].isnull()].index
df3 = df2.drop(idx_to_drop)
```

Count values:
```python
df3['benign_malignant'].value_counts()
```

I also [finished writeup](https://pechyonkin.me/portfolio/breast-cancer-diagnostics/) of my Udacity MLND nanodegree capstone project. I will need to do another go at proofreading as I copied some text from PDF generated by LaTeX and it has line breaks in unexpected places.

## Links and Resources
- [OSSU Data Science](https://github.com/ossu/data-science) curriculum
- [GANs explained](http://kvfrans.com/generative-adversial-networks-explained/) [by kvfrans]
- [Variational Autoencoders Explained](http://kvfrans.com/variational-autoencoders-explained/) [by kvfrans]
- [Tensor Considered Harmful](http://nlp.seas.harvard.edu/NamedTensor)
- [CV compiler](https://cvcompiler.com/)