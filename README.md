**ControlGAN**

☑ Improving CGAN's ability to create images with correct labels. Focus on details attribute which hard to be done with CGAN.

☑ Making GAN models capable of extrapolation, zero-shot learning.

Dataset was used : [CelebA](https://paperswithcode.com/dataset/celeba)

❎ Highly recommend use VGA Nvidia P100 Tesla (or stronger GPU) to train this model. If you dont have GPU, You can try it on [gg colab](https://colab.research.google.com/)

Network Architecture:

<img src = "https://github.com/Crazylov3/ControlGAN/blob/main/image/clasi.png" width = "350"> <img src = "https://github.com/Crazylov3/ControlGAN/blob/main/image/dis.png" width = "350"> <img src = "https://github.com/Crazylov3/ControlGAN/blob/main/image/gen.png" width = "350">

***Due to the limiting of computation power, I only trained it for 60k iteration. So the result shown below is not the final result.*** If you want to generate more realistic images, you can use
the bigger model depending on what hardware you have. You also can use ProGan backbone to have fun with this classifier. 

Male:

<img src = "https://github.com/Crazylov3/ControlGAN/blob/main/image/GAN.png" width = "500">

Female:

<img src = "https://github.com/Crazylov3/ControlGAN/blob/main/image/female.png" width = "500">

**🔻 Now, let's test the efficiency of using the classification model, and also extrapolation, zero-shot learning**

<img src = "https://github.com/Crazylov3/ControlGAN/blob/main/image/noisuy.png" width = "500">

*We used binary(0, 1) vector for representing the label. In this task we just multiply the label vector by x1, x2, x3. As you can see, the model works quite well one strange label (zero-shot learning).
ControlGAN has shown its outstanding strength compared to CGAN. Now we can change the label of the generated image more easily!*

❌If you want to have pretrained-weight model,  drop me an email (giangletruong2444@gmail.com)

❗ This project is inspired by [this paper](https://ieeexplore.ieee.org/document/8641270)
