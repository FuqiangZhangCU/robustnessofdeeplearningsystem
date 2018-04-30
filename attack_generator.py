
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import matplotlib.pyplot as plt


from testmodels.setup_cifar import CIFAR, CIFARModel
from testmodels.setup_mnist import MNIST, MNISTModel

from lib.l2_attack import CarliniL2
from lib.l0_attack import CarliniL0
from lib.li_attack import CarliniLi


def show(img, shape, channels, name):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    
    size = len(img)
    if len(img) != shape*shape * channels: return
    print("START")
    for i in range(shape):
        print("".join([remap[int(round(x))] for x in img[i*shape:i*shape+shape]]))
    
    img1 = [img[i:i + shape] for i in range(0, size, shape)]
    if channels == 1 :
        scipy.misc.imsave('output/' + name+ '.png', img1)
    if channels == 3 :
        plt.imsave('output/' + name+ '.png', img1)
    
def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


def load_algorithms(fpath = "./lib/config.txt") :
    
    if fpath == "" :
        return dict()
    
    ag_status = dict()
    with open(fpath,"r") as f:
        for line in f:
            lis = line.replace("\n", "").split('=')
            ag_status[lis[0]] = int(lis[1])
    return ag_status

if __name__ == "__main__":
    with tf.Session() as sess:
        
        #data, model =  MNIST(), MNISTModel("inputs/model/mnist-distilled-100", sess)
        data, model =  CIFAR(), CIFARModel("inputs/model/cifar-distilled-100", sess)
        
        attacks = []
        ag_status = load_algorithms()
        
        if ag_status['l2'] == 1:
            attacks.append(CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0))
        if ag_status['l0'] == 1:
            attacks.append(CarliniL0(sess, model, max_iterations=1000))
        if ag_status['li'] == 1:
            attacks.append(CarliniLi(sess, model, max_iterations=1000))
        #attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)
        #attack = CarliniL0(sess, model, max_iterations=1000)
        #attack = CarliniLi(sess, model, max_iterations=1000)
        
        inputs, targets = generate_data(data, samples=1, targeted=True,
                                        start=0, inception=False)
        cnt = 0
        at = 0
        for attack in attacks:
        
            timestart = time.time()
            adv = attack.attack(inputs, targets)
            timeend = time.time()
        
            print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

            for i in range(len(adv)):
                print("Valid:")
                show(inputs[i], model.image_size, model.num_channels, "cifar/initial_" + str(cnt) + '_' + str(at))
                print("Adversarial:")
                show(adv[i], model.image_size, model.num_channels, "cifar/adversarial_"  + str(cnt) + '_' + str(at))
                print("Classification:", model.model.predict(adv[i:i+1]))
                print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)
                cnt = cnt + 1
            at = at + 1
    with tf.Session() as sess:
        
        data, model =  MNIST(), MNISTModel("inputs/model/mnist-distilled-100", sess)
        #data, model =  CIFAR(), CIFARModel("inputs/model/cifar-distilled-100", sess)
        
        attacks = []
        ag_status = load_algorithms()
        
        if ag_status['l2'] == 1:
            attacks.append(CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0))
        
        if ag_status['l0'] == 1:
            attacks.append(CarliniL0(sess, model, max_iterations=1000))
        
        if ag_status['li'] == 1:
            attacks.append(CarliniLi(sess, model, max_iterations=1000))
        #attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)
        #attack = CarliniL0(sess, model, max_iterations=1000)
        #attack = CarliniLi(sess, model, max_iterations=1000)
        
        inputs, targets = generate_data(data, samples=10, targeted=True,
                                        start=0, inception=False)
        cnt = 0
        at = 0
        for attack in attacks:
            
            timestart = time.time()
            adv = attack.attack(inputs, targets)
            timeend = time.time()
        
            print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

            for i in range(len(adv)):
                print("Valid:")
                show(inputs[i], model.image_size, model.num_channels, "mnist/initial_" + str(cnt) + '_' + str(at))
                print("Adversarial:")
                show(adv[i], model.image_size, model.num_channels, "mnist/adversarial_"  + str(cnt) + '_' + str(at))
                print("Classification:", model.model.predict(adv[i:i+1]))
                print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)
                cnt = cnt + 1
            at = at + 1