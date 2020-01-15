# starGAN-LSGAN
simply implementation of starGAN-LSGAN using TensorFlow. this implementation has many issues. I will revise this over time.

original paper:https://arxiv.org/abs/1711.09020

original implementation:https://github.com/yunjey/stargan

## usage
like this
```
mainls.py
pred.py
train
  ├ domain0
  |   ├ aaa.png
  |   ...
  |   └ zzz.jpg      
  ├ domain1
  |   ├ bbb.png
  |   ...
  |   └ xxx.jpg
  ...
  └ domainN
      ├ ccc.png
      ...
      └ yyy.jpg      
test
  ├ domain0
  |   ├ AAA.png
  |   ...
  |   └ ZZZ.jpg      
  ├ domain1
  |   ├ BBB.png
  |   ...
  |   └ XXX.jpg
  ...
  └ domainN
      ├ CCC.png
      ...
      └ YYY.jpg   
```

To train

```
python mainls.py
```

To Validate, must feed the folder location and attributes. 

In below example, the images located test/domain2 are converted to 4th attribute (maybe domain4).

example:

```
python test.py test/domain2 3
```

## Result examples
Hair color conversion of anime face

Leftmost is input, others are output.

<img src = 'example/hair_color.png' width = '800px'>
