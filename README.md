# Iznik-Pottery-IMAGINE

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
