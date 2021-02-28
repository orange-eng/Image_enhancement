from model import EDSR
import scipy.misc
import argparse
import data
import os
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/General-100")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=2,type=int)
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--iterations",default=2,type=int)
parser.add_argument("--numimgs",default=5,type=int)
parser.add_argument("--outdir",default="out")
parser.add_argument("--image",default='lr0.png')
args = parser.parse_args()
if not os.path.exists(args.outdir):
	os.mkdir(args.outdir)
down_size = args.imgsize//args.scale
network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale)
network.resume(args.savedir)
if args.image:
	x = scipy.misc.imread(args.image)
else:
	print("No image argument given")
inputs = x

x = np.array([inputs])
print("inputs.shape=",inputs.shape )
outputs = network.predict(x)
outputs = np.squeeze(outputs)
print(outputs.shape)
if args.image:
	scipy.misc.imsave(args.outdir+"/input_"+args.image,inputs)
	scipy.misc.imsave(args.outdir+"/output_"+args.image,outputs)
