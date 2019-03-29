# starGAN-LSGAN
simply implementation of starGAN-LSGAN. this implementation has many issues. I will revise this over time.
original paper:https://arxiv.org/abs/1711.09020
original implementation:https://github.com/yunjey/stargan

## usage
put folders like "train/trainblk", "train/trainbld" and "train/trainblu"... 
as well put folders like "test/testblk", "val/testbld" and "val/testblu"... 
you can fix in line61 of "mainls.py".
After, run "python mainls.py" starts training.
when training is over, run like "python pred.py test/testblk 1" executes predicting. outputs converted images.
