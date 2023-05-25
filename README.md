# Iznik-Pottery-IMAGINE

## Code and results from my internship at Imagine Laboratory (ENPC).

---
---

### Goal of the project

---

I aim to generate a dataset of usable Iznik pottery photos, and then train an adversarial network to create such Iznik tiles.

You'll find two files here :

1. The file `Iznik Dataset Generator` which contains all the code to generate a dataset through active learning (assisted by an operator). All parameters can be changed and this method can adapt to any dataset you want to create.

2. The file `Iznik Tile GAN` which is the core of the Generative Adversarial Network trained to generate his own Iznik tiles and mosaïcs.

---

- Link of the advancement log book : <https://shorturl.at/fqrFR>.

- To launch the dataset generator, just enter `python ./main.py` in the terminal, when you're in the `Iznik Dataset Generator`'folder.

---

### How the code works

---

#### **The Dataset Generator** :

- Downloading images through Bing images, remove duplicatas and flip them randomly to add free data.

- Make them go all through _DINOv2_ and memorize the resulting vectors.

- Through a linear-sigmoid module, compute the 25 best scores and display the corresponding images.

- Label these images by hand, by clicking on the images, using the buttons 'Validate' and 'Unvalidate all'.

- Thanks to these newly labeled images, do a gradient descent on our weight vector to update it.
    - This descent is made through all the labeled images, so the learn grows more rapidly at every iteration
    - The batch size for this descent is **25**.
    - For each batch, there is **100** epochs.

- Reiterate all the steps, but skipping the images that have already been labeled.


After a few steps, we will obtain only 'nice' images every iteration, until there is only 'false' images.

Thanks to this process, it merely takes __20 mins__ to generate a __2000 items__ dataset.


---