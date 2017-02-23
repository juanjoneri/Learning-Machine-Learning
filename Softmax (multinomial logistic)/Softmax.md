1. The result is that mnist.train.images is a tensor (an n-dimensional array) with a shape of [55000, 784]. The first dimension is an index into the list of images and the second dimension is the index for each pixel in each image. Each entry in the tensor is a pixel intensity between 0 and 1, for a particular pixel in a particular image.

2. For the purposes of this tutorial, we're going to want our labels as "one-hot vectors". A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension. For example, 3 would be [0,0,0,1,0,0,0,0,0,0]. Consequently, mnist.train.labels is a [55000, 10] array of floats

3. A softmax regression has two steps: first we add up the evidence of our input being in certain classes, and then we convert that evidence into probabilities.

    31. To tally up the evidence that a given image is in a particular class, we do a weighted sum of the pixel intensities. The weight is negative if that pixel having a high intensity is evidence against the image being in that class, and positive if it is evidence in favor.
    32. We also add some extra evidence called a bias. Basically, we want to be able to say that some things are more likely independent of the input. E_i = ∑ W_i * x_i + b_i
    33. We then convert the evidence tallies into our predicted probabilities y using the "softmax" function:
        331. normalize(exp(x_i)) // SoftmaxN[A__, n_] := N[ Exp[ A[[n]] ] / Total[ Exp[A] ] ]

4.  Cost function Cross entropy -> Cross-entropy arises from thinking about information compressing codes in information theory but it winds up being an important idea in lots of areas, from gambling to machine learning. It's defined as: H_y′(y) = − ∑ y_i′ log⁡(y_i), Where y is our predicted probability distribution, and y′ is the true distribution 
