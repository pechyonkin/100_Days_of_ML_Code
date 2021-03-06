# 100 Days of ML Code

[Inspired](https://github.com/llSourcell/100_Days_of_ML_Code) by Siraj Raval.

---
#### Frequently Used

1. [How to run](./jupyter-guide.md) Jupyter Notebook on a remote server and access via ssh on a local machine
2. [Tmux cheatsheet](./tmux-cheatsheet.md)

---

## Day 41. February 26.

Today I finished [Joining Data in SQL](https://www.datacamp.com/courses/joining-data-in-postgresql) course on DataCamp.

Then, I started [Intermediate SQL](https://www.datacamp.com/courses/intermediate-sql).

## Days 38–40. February 23–25.

I finished the prototype of the eye trracking system. The model trained well and error was relatively low. When I visualized predictions, it turned out in some cases system is hundreds of pixels away from true label but it is always in the correct area. I think if I collect more data, especially for more people (currently it is only me), then the model will be able to generalize well. In addition there are more ways to make it better, for example, force the output to be in the correct range (inside the screen). At the moment, predictions are unbounded. I will work on this in the following days.

## Days 36–37. February 21–22.

I have built a web-based data collection system for my eye tracking system. It was very tough but now it is working. I learned basics of HTML, JavaScript, Ajax and Flask. I will finish the system and then I can collect the data for the project, train the model and see if it makes good predictions.

## Day 35. February 20, Wednesday.

I worked on a web-app to collect data for my next portfolio project, which is an eye tracking system. Had to dig into using Flask, HTML, Javascript and CSS. Also spent quite some time getting my laptops web-camera to work with the app inside the browser.

## Day 34. February 19, Tuesday.

### Other Resources
- [Chris Albon](https://chrisalbon.com/) - a lot of mini-notes on different ML-related topics. Very nice. [Repo](https://github.com/chrisalbon/notes). Which framework does he use to generate the website?
- [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm) is like a vector's $L_2$ norm: $\Vert A \Vert_F=\sqrt{\sum_{i}\sum_{j}|a_{ij}^2|}$

## Day 33. February 18, Monday.

### mAP - mean varage precision:
- [Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- [Average precision and precision@k](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [Precision-recall curves – what are they and how are they used?](https://acutecaretesting.org/en/articles/precision-recall-curves-what-are-they-and-how-are-they-used)
- [Measuring Object Detection models — mAP — What is Mean Average Precision?](https://towardsdatascience.com/what-is-map-understanding-the-statistic-of-choice-for-comparing-object-detection-models-1ea4f67a9dbd)
- PASCAL VOC [paper](http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf) describes hwo mAP is calculated

### Other Resources
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)
- [Gradient Checkpointing](https://github.com/openai/gradient-checkpointing) - fit a larger model into your GPU memory
- [Machine Learning for Artists](https://ml4a.github.io/)

### Read Papers
- [R-CNN](https://arxiv.org/abs/1311.2524) – region proposal -> CNN -> SVM:
	- mAP
	- precision-recall curve
	- non-maximum supression 

## Day 32. February 14, Thursday.

1. Watched [James Powell - So you want to be a Python expert?](https://youtu.be/cKPlPJyQrt4).
2. Finished reading [Python Tricks](https://dbader.org/products/python-tricks-book/) book by Dan Bader.
3. Started [Joining Data in SQL](https://www.datacamp.com/courses/joining-data-in-postgresql) course at DataCamp.

# Day 31. February 13, Wednesday.

Having read the article titled "[SQL: One of the Most Valuable Skills](http://www.craigkerstiens.com/2019/02/12/sql-most-valuable-skill/)", I went and completed the [Intro to SQL for Data Science](https://www.datacamp.com/courses/intro-to-sql-for-data-science) course on DataCamp.

I also started working on the writeup about the leafy greens classifier for the [portfolio page](https://pechyonkin.me/portfolio/) on my website.

### Other interesting links:

- [PostgreSQL](https://en.wikipedia.org/wiki/PostgreSQL)
- [Job loss due to AI — How bad is it going to be?](https://www.skynettoday.com/editorials/ai-automation-job-loss) [Skynet Today]

## Day 30. February 12, Tuesday.

Today I went through the [Kaggle Planet Competition](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) tutorial from Lesson 3.

I read the data [block API docs](https://docs.fast.ai/data_block.html). This API is very flexible but I will need more time and practice to familiarize myself with it closely.

I also started with [mlcourse.ai](https://mlcourse.ai/). Looks exciting. Made by top kagglers from [ods.ai](http://ods.ai/).

## Day 29. February 11, Monday.

Expanding on Python knowledge:

- [Python AST Ecosystem](https://youtu.be/Yq3wTWkoaYY) [materials are [here](https://github.com/hchasestevens/europython-2018)]

I started doing the Kaggle Planet competition as taught in lesson 3. I've set up repos, downloaded data through Kaggle CLI and am ready to go.

## Day 28. February 9, Satruday.

I delved in Python today. I watched and read about decorators and [amazing](https://youtu.be/Z_OAlIhXziw) [things](https://youtu.be/D1twn9kLmYg) that can be done with them. These 2 videos are very advanced and I will only benefit from rewatching. But even if I didn't get everything from the first watch, I still have a much better understanding of Python.

Note: David Beazley is am amazing presenter and I shuould definitely get his book, [Python Cookbook](https://www.amazon.com/Python-Cookbook-Third-David-Beazley/dp/1449340377). 

## Day 27. February 7, Thursday.

### FastAI

I watched Lecture 3 of FastAI v3 course. A lot of information to digest.

### ML Training Video

I also watched [this](https://youtu.be/vVd5d3UTbHA) solution of Kaggle Human Protein Atlas Image Classification competition 4-th place solution (in Russian). They used FastAI! Key takeaways and areas for further research:

- stacking
- adversarial validation (Xception)?
- averaging among validation folds with different random seeds to find optimal thresholds?
- SE-ResNext-50
- Focal loss, [LSEP loss](https://arxiv.org/abs/1704.03135) - add LSEP after model is trained, to finetune
- Lovasz loss
- network head: `AdaptiveConcat -> BN -> Dropout -> Dense`
- augmentation: D4?
- TTA: 32x?
- meta-features for stacking: brightness, contrast, channel correlation
- Stacking (on validation):
	-  LightGBM
	- outer folds: 5
	- inner folds: 3 (up to 5 times depending on class)
	- repeat 20 times (on outer with different seeds)
	- average voting
	- ~375 models for each class
	- best single model - 21st place
- sample pairing
- mixup
- metric learning
- user bestfitting (1-st place)
- sample pairing vs. mixup

### Go language crash course

[This](https://youtu.be/Xj5lo2-ZvRg) tutorial.

## Day 26. February 6, Wednesday.

I got to finally deploy my [leafy greens model](https://herbs.onrender.com/).

[Render](https://render.com) is very easy to use. I used [this]() tutorial to deploy. I had to play with css and html a little bit to adapt it for my needs. Also classes did not want to present themselves as English names and instead were presented as class numbers. I used a Python dictionary to map class number to class name and it worked.

I did not think deployment is so easy. Thanks to services like [Render](https://render.com) for making life of deep learning practitioners easier!

Tomorrow I can finally start the [next lesson](https://course.fast.ai/videos/?lesson=3) of fast.ai course.

## Day 25. February 5, Tuesday.

Today I finished and published [overview of the ZFNet](https://pechyonkin.me/architectures/zfnet/). Writing is a good way to understand small details of a paper, as it makes you think how it was done. Another good way is to implement the paper in code and make it work. I will also do that.

## Day 24. February 4, Monday.

I continued learning about GPU architecture and parallel compute. I learned about processes/threads. And also about architectural details of Nvidia GPUs.

#### CUDA Tutorials [Playlist](https://www.youtube.com/playlist?list=PL33Hr-B6eUxCLjiHaKHTcwEn5JZKCdyjX):
1. [CUDA Part A: GPU Architecture Overview and CUDA Basics; Peter Messmer (NVIDIA)](https://youtu.be/nRSxp5ZKwhQ) [stopped at 59:44]

#### Graphcore Research
- [Simon Knowles - More complex models and more powerful machines](https://www.youtube.com/watch?v=dLvkF_TmyAc) - talk by Simon Knowles, Graphcore's CTO [May 2018]
	- most of customers now focus on capability 
	- existential problem for many businesses
	- "we don't care how much it costs"
	- "we just need to get there before our competitors"
	- designing the most sophisticated chip for research, for finding a way to do *x* with machine intelligence (whatever that *x* is depends on each customer's application)
	- **recurrences are convolution in time instead of in space**
	- **reversible models** - do not require storage of forward pass activations, are computed automatically during backward pass
- [CogX 2018 - Future of Hardware Panel](https://youtu.be/zRPpyabE7p4)

## Day 23. February 3, Sunday.

After doing some research about [Graphcore](https://www.graphcore.ai/), I decided to dive deeper into how linear algebra subroutines are implemented at a low level, both on CPUs, GPUs and in distributed computation. I want to understand how deep learning works on the GPU. So I am spending today researching these quite low level details.

- [How to build a processor for machine intelligence](https://www.youtube.com/watch?v=T8DvHnb3Y9g) - talk by Graphcore's CTO
	- graphical structure of models
	- "I've never seen a workload like this one", in 30 years of making processors @[5:30](https://youtu.be/T8DvHnb3Y9g?t=330)
	- Microsoft Kinect as a "random forest machine"
	- "**What is fundamental about machine intelligence compute task as a workload?**" [answer is [this](https://youtu.be/T8DvHnb3Y9g?t=500) list]
	- [Bulk synchronous parallel](https://en.wikipedia.org/wiki/Bulk_synchronous_parallel)
	- modern DL: save activations in memory (used for backprop step) - requires a lot of memory, but saves compute
	- IPU: don't save all activations, only 1-in-N, recompute them as needed when doing backprop (you can recompute part of the activations starting from last 1-in-N precomputed activation), because you have a lot of compute, you can do it and you don't need a lot of memory (**AWESOME IDEA**)
	- spend 30% more compute for 1-2 orders of magnitude less memory (10x-100x)
	- this roughly means that 300MB of memory per chip as per their recent specification is similar to 3-30GB of "traditional" GPU, but with a much higher compute capability
	- [Photos](https://www.eenewsanalog.com/news/graphcores-colossus-ipu-package) of packaged chip, and [some more](https://www.eenewsanalog.com/news/graphcores-two-chip-colossus-close-launch/page/0/1)
- [Nigel Toon Graphcore Samsung CEO Summit 2017](https://youtu.be/7qp3kLHgsck)
- [Professor Jack Dongarra talks about BLAS and CUDA](https://www.youtube.com/watch?v=lk0qYdnv0bg) - an old video from 2008. All of linear algebra soubroutines are implemented in Fortrann and C. Single vs. double precision performance spread 5x-10x. Most computation in single precision to be fast, then use double precision to refine solution.
- [Linear Algebra on GPU](https://youtu.be/7rkucJiw_TE) - a relatively short overview of a broad topic, cuBLAS and MAGMA. Data parallelism on GPU (vs. task parallelism on CPU) [part of [online seminars](https://www.sharcnet.ca/help/index.php/Online_Seminars) sequence]
	- maximize throughput
	- data parallelism for single task
	- super-threaded ?
	- large-scale [SIMD](https://en.wikipedia.org/wiki/SIMD)
	- memory bandwidth bottlenecks RAM <-fast-> CPU <-slow-> GPU <-very fast-> VRAM
	- CUDA programming model:
		- CPU is host
		- GPU is coprocessor
		- kernels are functions executed in parallel
		- CPU manages all memory allocations, data transfers and invocation of kernels on the GPU
	- LA on CPU: BLAS, LAPACK
	- LA on GPU: CUBLAS, CULA, MAGMA
	- GPU memory is high bandwidth / high latency (truck analogy - slow, but transports a lot)
	- CPU memory is low bandwidth / low latency (sports car analogy - fast and transports very little)
	- GPU shared memory - small region of memory where random access is very fast. Multiple threads don't need to access memory in adjacent way (*coalesced way*).

## Day 22. February 1, Friday.

I skipped 1 day because I had business to attend to. Days like that can make me lose focus and then getting back on track can be difficult. I will have to make a mini-summary of 1/5 of the 100 journey completed.

Today I was researching for my next post about ZFNet and also writing that post. I feel like I could have been more productive today. I will try to regainn my focus and keep working toward finishing the ZFnet post as well as deploying the green veggies project.

## Day 21. January 30, Wednesday.

### Useful Materials

- [Right Code, Right Place, Right Time](https://youtu.be/BvqLgeNVg0Y) - PyData2018 talk by Tim Hopper about productizing ML

## Day 20. January 29, Tuesday.

I continued to write review of ZFNet. I hope I will have enogh grit to finish and publish it within 3 days.

I went back to my green vegetables classisier. I need to deploy the model to a web app. Time to learn Docker, Kubernetes, microservices etc. I watched [this](https://youtu.be/1xo-0gCVhTU) introduction to get an overview of technologies and use cases.

## Day 19. January 28, Monday.

Today I continued doing research for my blog post about MXNet. I watched CS231n 2017 [Lecture 7](https://youtu.be/6wcs6szJWMY) (I actually watched it before, this is how I knew I needed to rewatch it) and that clarified a lot about deconvolutions and visualization of convolutional networks. I think my next project will be visualizing convnets using different methods outlined in the lecure using PyTorch.

## Day 18. January 27, Sunday.

Today I started writing an overview of the [ZFNet](https://arxiv.org/abs/1311.2901) for my [website](https://pechyonkin.me/architectures/).

While doing it I got confused about convolutions, transposed convolutions and backprop throught them. That finally got me to read the [Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/abs/1603.07285) (with [awesome GIFs](https://github.com/vdumoulin/conv_arithmetic)). That clarified it a lot.

## Day 17. January 26, Saturday.

Today I [learned](https://www.youtube.com/watch?v=KdZ4HF1SrFs) that Python has `while-else` and `for-else` loops. The `else` clause is evaluated once when the loop condition becomes false. It won't be evaluated if there is a break statement or an error thrown inside the while loop.

## Day 16. January 25, Friday.

Worked on Jeremy's PyTorch tutorial. PyTorch is much more pleasing than TF. Will add more details later.

### Notes:

- formula for iterations: `for i in range((i - 1) // bs + 1)`. I worked it on paper and can explain it by example, but somehow this *doesn't click* and there is no aha moment with it yet.
- `Sequential`
- You can create a custom layer by subclassiing `nn.Module`
- Python: when subclassing, in `.__init__(self)` you have to call `super().__init__()` to initialize parent class
- any `nn.Parameter` assigned to `.self.<varname>` inside `nn.Module` subclass's `.__init()` will be added to model parameters and can be optimized
- `torch.optim` has optimizers
- `torch.utils.data` has:
	- `TensorDataset` - builds a dataset from tensors
	- `DataLoader` - builds iterable dataloader from dataset, needs to know batch size
- call `model.train()` when optimizing (training) and `model.eval()` before inference (e.g. validation). This is needed for dropout, batchnorm and some other layers to work properly (they have different regimes for training and inference)
- `tensor.view()` - the same as `np.ndarray.resize()`. It is actually a view on data in memory, because data is not changed / moved in memory when you call this method.
- `tensor.item()` gives scalar for a tensor of only one value
- `tensor.size(d)` gives shape of tensor across dimension `d`
- nice trick to create tensors from numpy arrays by mapping `torch.tensor`:
```python
x_tr, y_tr, x_val, y_val = map(
    torch.tensor, (x_tr, y_tr, x_val, y_val)
)
```
- main idea of training steps (generally applies to deep learning):
	1. forward pass: `pred = model(xb)`
	2. calculate loss: `loss = loss_func(pred, yb)`
	3. backward pass, calculates gradients for `model.parameters()`: `loss.backward()`
	4. perform update of parameters accoridng to specific optimizer and learning rate: `opt.step()`
	5. zero out gradients: `opt.zero_grad()`
		- Stackoverflow: [Why do we need to explicitly call zero_grad()?](https://stackoverflow.com/questions/44732217/why-do-we-need-to-explicitly-call-zero-grad)
		- Basically we can use gradient accumulation across batched, that's why.

## Day 15. January 24, Thursday.

I continued with Jeremy Howard's PyTorch tutorial I started yesterday. Did not finish yet. But I have to say that after TensorFlow, PyTorch is a piece of cake. Much more pythonic and elegant, simple syntax and ideas. By comparison, TF feels like a monstrocity piece of software.

I also transferred three articles from Medium to my personal blog:

1. [Key deep learning architectures page](https://pechyonkin.me/architectures/), that I will update as I write more review of architectures (originally [here](https://medium.com/@pechyonkin))
2. [LeNet-5 overview](https://pechyonkin.me/architectures/lenet/) (originally [here](https://medium.com/@pechyonkin/key-deep-learning-architectures-lenet-5-6fc3c59e6f4))
3. [AlexNet overview](https://pechyonkin.me/architectures/alexnet/) (originally [here](https://medium.com/@pechyonkin/key-deep-learning-architectures-alexnet-30bf607595f1))

## Day 14. January 23, Wednesday.

### 1. Leafy Greens Project

I finished cleaning data, more than 75% of original downloaded images were removed. Most of them are images of foods containing greens that I was looking for. I went from more than 8000 images across 29 classes to a bit more than 2000. I spent hours cleaning all that.

Then I built a classifier that achieved 22% error rate, which is high, but satisfactory given very similar classes and small dataset with low quality images.

Mostly the classifier got confused on very similar greens. Examples:

- cilantro and parsley
- cilantro and celery
- choy sum and gai lan (Chinese broccoli
- tricolor daisy and crown daisy
- watercress and pea shoots

This is further set back by the fact that data is not of very high quality and not very big size.

At this point, I am satisfied with result (22% error rate, about 88 mistakes out of 401 validation examples) because the cost of acquiring more data is not very interesting to me for learning purposes.

The next step will be to deploy this model as a web-app for learning purposes.

### 2. What is `torch.nn` really?

I decided to follow along with "[What is torch.nn really?](https://pytorch.org/tutorials/beginner/nn_tutorial.html)" tutorial, typing along in a dedicated [notebook](https://github.com/pechyonkin/learning-pytorch/blob/master/torch-really.ipynb) by hand to train muscle memory, exploring PyTorches basics. So far PyTorch seems much more intuitive and easy than TensorFlow. No wonder fastai decided to use it as their backend.
 

## Day 13. January 22, Tuesday.

I finally got to start cleaning the data. I learned that it is a very boring task, taking a lot of time. And also, Google image search for Chinese terms gives very low quality images. I probably deleted 40-50% of all downloaded images.

## Day 12. January 21, Monday.

Today I finished and finally published an [article](https://pechyonkin.me/deep-learning-vision-non-vision-tasks/) about creative application of deep learning vision to non-vision tasks.

### Useful Links

My yesterday's [tweet](https://twitter.com/max_pechyonkin/status/1086852019376308224) sharing the articleabout `.einsum` from yesterday unexpectedly got popular. Some people shared other posts about it, so I am deepening my understanding of `.einsum`. Two new posts about it read today. 

- [Blog post](https://rockt.github.io/2018/04/30/einsum) about einsum by Tim Rocktäschel
- [A basic introduction to NumPy's einsum](http://ajcr.net/Basic-guide-to-einsum/)
- [Programmer Competency Matrix](https://sijinjoseph.com/programmer-competency-matrix/)

## Day 11. January 20, Sunday.

Today is mostly reading day.

- When reading source code for [`numpy.einsum`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html), I discovered that you can [`.pop`](https://github.com/numpy/numpy/blob/v1.15.4/numpy/core/einsumfunc.py#L1224) values from a dict. Python is amazing!
- **Inner product** of two vectors -> scalar. Another way to think is inner product = row vector x column vector.
- **Outer product** of two vectors -> matrix. Another way to think is that outer product = column vector x row vector.
- **Matrix L2 norm** is the square root of the sum of all squared elements.

### Useful Links

- [Einstein Summation in Numpy](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/) [blog by Olexa Bilaniuk] - this blog needs more attention, very good explanation and it broadened my multidimensional array toolkit. Note that both [PyTorch](https://pytorch.org/docs/stable/torch.html?highlight=einsum#torch.einsum) and [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/einsum) have their own versions of `.einsum`.
- [`numpy.einsum`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html)

## Day 10. January 19, Saturday.

Today I made a notebook and downloaded the data for my [green leafy vegetables](https://github.com/pechyonkin/culinary-herbs-classifier) classifier. The data is really dirty - a lot of irrelevant images. I will have to spend a lot of time cleaning the data for 29 classes tomorrow.

### Useful Links

- [The Evolution of Trust](https://ncase.me/trust/) - an interactive "game" that teaches the basics of game theory. Very interesting implications about the spread of trust and distrust in modern society. **Note**: this is not *directly* related to ML.
- [How to create a deep learning dataset using Google Images](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/) [post by Adrian Rosebrock]
- [What is `torch.nn` really?](https://pytorch.org/tutorials/beginner/nn_tutorial.html) - awesome tutorial [recommended](https://twitter.com/jeremyphoward/status/1085586894543437824) by Jeremy Howard

## Day 9. January 18, Friday.

To get some data for my green leafy herbs I went to a closest vegetables market and recorded all Chinese names of green leafy things. I got a list of 30. Then I went back and meticuluosly converted all of them into [pinyin](https://en.wikipedia.org/wiki/Pinyin) and also translated to English. You can find the list [here](https://github.com/pechyonkin/culinary-herbs-classifier).

### Useful Links

- [Markdown table generator](https://www.tablesgenerator.com/markdown_tables)
- [Chinese characters to pinyin converter](https://www.chineseconverter.com/en/convert/chinese-to-pinyin)

## Day 8. January 17, Thursday.

I started a green leafy culinary herbs classifier. Many culinary herbs look the same to me. So I decided to do this project as homework for fastai v3 lesson 2. My goals for this project:

- use lesson 2 functionality to download images from Google (haven't done it before)
- train a model using transfer learning, preferrably get good performance
- deploy a model (haven't done it before)
- use [Starlette](https://www.starlette.io/) framework as recommnded in class (haven't done it befor)

Out of 4 goals 3 are new to me. So I will do a lot of learning for this project.

## Day 7. January 16, Wednesday.

Today I finished fastai part 1 v3 lesson 2.

I also set up repos for development of fastai and fastai_docs so that I can contribute to these libraries. There are some standards of contribution, such as stripping notebooks of outputs when committing, the instructions are:

- for [`fastai`](https://docs.fast.ai/dev/develop.html)
- for [`fastai_docs`](https://github.com/fastai/fastai_docs)
- [doc](https://docs.fast.ai/gen_doc_main.html) about authoring fastai docs

Today [my first pull request](https://github.com/fastai/course-v3/pull/138#issuecomment-454647053) to the fastai library was merged. It is a very small change, but I am still happy. So I decided to deepen my knowledge about how to contribute. To summarize, the steps are below (taken from [here](https://akrabat.com/the-beginners-guide-to-contributing-to-a-github-project/)).

#### Steps to contribute to fastai

1. Fork the project & clone locally. 
2. Create an upstream remote.
	- `git remote add upstream git@github.com:fastai/course-v3.git`
3. sync your local copy before you branch.
	- `git checkout master`
	- `git pull upstream master && git push origin master`
3. Branch for each separate piece of work.
	- `git checkout -b hotfix/readme-update` (`hotfix/readme-update` is branch name)
4. Do the work, write [good commit messages](https://chris.beams.io/posts/git-commit/), and read the CONTRIBUTING file if there is one.
5. Push to your origin repository.
	- `git push -u origin hotfix/readme-update` (the `-u` flag links this branch with the remote one, so that in the future, you can simply type `git push origin`)
6. Create a new PR in GitHub.
7. Respond to any code review feedback.

### Today's Reading

#### Contributing to Open Source Projects

- [The beginner's guide to contributing to a GitHub project](https://akrabat.com/the-beginners-guide-to-contributing-to-a-github-project/) - a nice step-by-step guide on how to properly set up and contribute to open source projects.
- Google's Python [styleguide](https://google.github.io/styleguide/pyguide.html)
- [Contributing to the new fastai framework](https://forums.fast.ai/t/contributing-to-the-new-fastai-framework/7246) [fastai forums]
- [How to contribute to fastai docs](https://forums.fast.ai/t/how-to-contribute-to-the-fast-ai-docs/27941/3) [fastai forums]
- [Code Reviews: Before You Even Run The Code](https://lornajane.net/posts/2015/code-reviews-before-you-even-run-the-code)
- [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)

#### Data Science

- [Bringing the best out of Jupyter Notebooks for Data Science](https://towardsdatascience.com/bringing-the-best-out-of-jupyter-notebooks-for-data-science-f0871519ca29)
- [An Introduction to the PyData World](https://speakerdeck.com/jakevdp/intro-to-pydata) - deck by Jake VanderPlas

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
