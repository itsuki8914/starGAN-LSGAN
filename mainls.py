import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from model import *
from btgen import *

SAVE_DIR = "model"
SVIM_DIR = "samples"
TRAIN_DIR = "train"
VAL_DIR = "test"

def tileImage(imgs):
    d = int(math.sqrt(imgs.shape[0]-1))+1
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.zeros((h*d,w*d,3),dtype=np.float32)
    for idx,img in enumerate(imgs):
        idx_y = int(idx/d)
        idx_x = idx-idx_y*d
        r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
    return r

def foloderLength(folder):
    dir = folder
    paths = os.listdir(dir)
    return len(paths)

def printParam(scope):
    total_parameters = 0
    for variable in tf.trainable_variables(scope=scope):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("{} has {} parameters".format(scope, total_parameters))

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(SVIM_DIR):
        os.makedirs(SVIM_DIR)
    img_size = 128
    bs = 16
    lmd = tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)

    trans_lr = 2e-4
    trans_lmd = 10
    max_step = 100000
    gp_lmd = 10
    cls_lmd = 1
    critic = 5

    # loading images on training
    domains= os.listdir(TRAIN_DIR)
    v_domains= os.listdir(VAL_DIR)
    btGens = []
    valGens = []
    genLens = []
    valLens = []
    num_domains = len(domains)
    for i, j in enumerate(domains):
        imgen = BatchGenerator(img_size=img_size, imgdir="{}/{}".format(TRAIN_DIR,j),dirID=i, num_domains=num_domains)
        btGens.append(imgen)
        length = foloderLength("{}/{}".format(TRAIN_DIR,j))
        genLens.append(length)

    for i, j in enumerate(v_domains):
        vlgen = BatchGenerator(img_size=img_size, imgdir="{}/{}".format(VAL_DIR,j),dirID=i, num_domains=num_domains, aug=False)
        valGens.append(vlgen)
        vlength = foloderLength("{}/{}".format(VAL_DIR,j))
        valLens.append(vlength)

    # sample images
    _Z = np.zeros([num_domains,img_size,img_size,3])
    for i in range(num_domains):
        id = np.random.choice(range(genLens[i]),bs)
        _A, d, s  = btGens[i].getBatch(bs,id)
        _Z[i] = (_A[0] + 1)*127.5
        print(d,s)
    _Z = tileImage(_Z)
    cv2.imwrite("input.png",_Z)

    #build models
    start = time.time()

    real_A = tf.placeholder(tf.float32, [bs, img_size, img_size, 3 ])
    real_B = tf.placeholder(tf.float32, [bs, img_size, img_size, 3 ])
    label_A2B = tf.placeholder(tf.float32, [bs, num_domains]) # target image atr
    label_A = tf.placeholder(tf.float32, [bs, num_domains]) # input image atr

    fake_A2B = buildGenerator(real_A,label_A2B,num_domains, reuse=False, nBatch=bs, name="gen")
    fake_B2A = buildGenerator(fake_A2B,label_A,num_domains, reuse=True, nBatch=bs, name="gen")

    #adv: real or fake, cls: domain classification

    #input real image for d_losses
    adv_real_B, cls_real_B = buildDiscriminator(real_B,num_domains, nBatch=bs, reuse=False, name="dis")
    #input fake images for g_losses
    adv_fake_A2B_g, cls_fake_A2B_g = buildDiscriminator(fake_A2B,num_domains, nBatch=bs, reuse=True, name="dis")
    #input fake images for d_losses
    adv_fake_A2B_d, _cls_fake_A2B_d = buildDiscriminator(fake_A2B,num_domains, nBatch=bs, reuse=True, name="dis")

    #ls gan

    d_loss_real = tf.reduce_mean((adv_real_B-tf.ones_like (adv_real_B))**2)
    d_loss_fake = tf.reduce_mean((adv_fake_A2B_d-tf.zeros_like (adv_fake_A2B_d))**2)
    d_cls_loss = -tf.reduce_mean(label_A* tf.log(tf.clip_by_value(cls_real_B, 1e-10, 1.0)))
    d_loss = d_loss_real + d_loss_fake + d_cls_loss
    g_adv_loss      = tf.reduce_mean((adv_fake_A2B_d-tf.ones_like (adv_fake_A2B_d))**2)

    #g_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cls_fake_A2B_g,labels=label_A2B ))
    g_cls_loss = -tf.reduce_mean(label_A2B* tf.log(tf.clip_by_value(cls_fake_A2B_g, 1e-10, 1.0)))

    real_A_img = real_A[:,:,:,:3]
    g_recon_loss = tf.reduce_mean(tf.abs(real_A_img - fake_B2A))
    g_loss = g_adv_loss + cls_lmd * g_cls_loss + lmd * g_recon_loss

    wd_gen = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="gen")
    wd_dis = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="dis")

    wd_gen = tf.reduce_sum(wd_gen)
    wd_dis = tf.reduce_sum(wd_dis)

    g_loss += wd_gen
    d_loss += wd_dis

    g_opt = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(g_loss,
                    var_list=[x for x in tf.trainable_variables() if "gen" in x.name])
    d_opt = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(d_loss,
                    var_list=[x for x in tf.trainable_variables() if "dis" in x.name])

    printParam(scope="gen")
    printParam(scope="dis")

    print("%.4e sec took building model"%(time.time()-start))

    start = time.time()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7))

    sess =tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    summary = tf.summary.merge_all()

    ckpt = tf.train.get_checkpoint_state('model')

    if ckpt: # checkpointがある場合
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        print ("load " + last_model)
        saver.restore(sess, last_model) # 変数データの読み込み
        print("succeed restore model")
    else:
        print("models were not found")
        init = tf.global_variables_initializer()
        #sess.run(init)

    print("%.4e sec took initializing"%(time.time()-start))


    gen_hist= []
    dis_hist= []

    start = time.time()

    for i in range(max_step + 1):
        # id: self domains (real class)
        # bt_images: images
        # direction: transform directions each images (fake class)
        id = np.random.choice(range(num_domains),bs)
        bt_images = np.zeros([bs, img_size, img_size, 3])
        directions = np.zeros([bs])
        selfIDs = np.zeros([bs, num_domains])

        for j,num in enumerate(id):
            imID = np.random.choice(range(genLens[num]),1)
            img, dir, selfID = btGens[num].getBatch(1,imID)
            bt_images[j] = img
            directions[j] = dir

        directions = np.vectorize(int)(directions)
        one_hot_dir = np.identity(num_domains)[directions]
        one_hot_id = np.identity(num_domains)[id]

        feed ={real_A: bt_images, real_B: bt_images, label_A2B: one_hot_dir,
            label_A : one_hot_id, lr: trans_lr}
        _, dis_loss, d_r, d_f, d_c =sess.run(
            [d_opt,d_loss,d_loss_real,d_loss_fake,d_cls_loss], feed_dict=feed)

        feed = {real_A:bt_images,label_A2B: one_hot_dir,label_A: one_hot_id,lr: trans_lr,
            lmd: trans_lmd}

        tmp, gen_loss, g_a, g_c, g_r  =sess.run(
            [g_opt,g_loss,g_adv_loss,g_cls_loss,g_recon_loss], feed_dict=feed)

        trans_lr = trans_lr *(1 - 1/max_step)

        print("in step %s, dis_loss = %.4e,  gen_loss = %.4e"%(i,dis_loss, gen_loss))
        print("d_r=%.3e d_f=%.3e d_c=%.3e g_a=%.3e g_c=%.3e g_r=%.3e "%(d_r,d_f,d_c,g_a,g_c,g_r))

        dis_hist.append(dis_loss)
        gen_hist.append(gen_loss)


        if i %100 ==0:
            id = np.random.choice(range(num_domains),bs)
            bt_images = np.zeros([bs, img_size, img_size, 3])
            directions = np.zeros([bs])
            selfIDs = np.zeros([bs, num_domains])

            for j,num in enumerate(id):
                imID = np.random.choice(range(valLens[num]),1)
                img, dir, selfID = valGens[num].getBatch(1,imID)
                bt_images[j] = img
                directions[j] = dir

            directions = np.vectorize(int)(directions)
            one_hot_dir = np.identity(num_domains)[directions]
            one_hot_id = np.identity(num_domains)[id]

            img_fake_A2B = sess.run(fake_A2B,feed_dict={real_A:bt_images,label_A2B:one_hot_dir})

            for im in range(len(bt_images)):
                cv2.putText(bt_images[im], '{}'.format(directions[im]), (img_size-18, img_size-8), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), lineType=cv2.LINE_AA)

            _A = tileImage(bt_images)
            _A2B = tileImage(img_fake_A2B)

            _Z = np.concatenate([_A,_A2B],axis=1)
            _Z = ( _Z + 1) * 127.5
            cv2.imwrite("%s/%s.png"%(SVIM_DIR, i),_Z)

            fig = plt.figure(figsize=(8,6), dpi=128)
            ax = fig.add_subplot(111)
            plt.title("Loss")
            plt.grid(which="both")
            plt.yscale("log")
            ax.plot(gen_hist,label="g_loss", linewidth = 0.25)
            ax.plot(dis_hist,label="d_loss", linewidth = 0.25)
            plt.xlabel('step', fontsize = 16)
            plt.ylabel('loss', fontsize = 16)
            plt.legend(loc='upper left')
            plt.savefig("histGAN.png")
            plt.close()

            print("%.4e sec took per100steps  lmd = %.4e ,lr = %.4e" %(time.time()-start, trans_lmd, trans_lr))
            start = time.time()

        if i%2500==0 :
            saver.save(sess,os.path.join(SAVE_DIR,"model.ckpt"),i)
    sess.close()

if __name__ == '__main__':
    main()
