# TorchLowRankRecon_MRI
This is a low-rank &amp; low-rank plus sparse reconstruction algorithm implemented by PyTorch
## Requirements
- Python 3.x
- PyTorch 
- NumPy
- bart 
- argparse
- tqdm
- matplotlib

## Usage
You can call the functions in ***reconLLR.py*** or ***reconL+S.py*** to perform low-rank or low-rank plus sparse reconstruction, just like in the ***example.py***. 

The data used in the ***example.py*** can be found at https://drive.google.com/file/d/1rW7qOMtbE_gAKuDOX9I-34vXon8XHwv1/view?usp=sharing. 

The results tested on a workstation with an Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz and an NVIDIA RTX A6000 GPU show that the speed of our implemented PyTorch CUDA reconstruction is 5 times faster than BART GPU (35s vs. 220s).


## Reference
https://github.com/mrirecon/bart

https://gregongie.github.io/files/misc/ongie-lrmc-julia.html

https://web.eecs.umich.edu/~fessler/course/598/l/n-05-prox.pdf

https://codeocean.com/capsule/0115983/tree/v1

https://github.com/jinh0park/pytorch-ssim-3D

https://ieeexplore.ieee.org/document/8540024