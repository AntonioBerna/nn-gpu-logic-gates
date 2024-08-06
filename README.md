# Introduction

Having trained a [neural network by combining the electronics of 4 LED diodes with the computational power of a Raspberry Pi 3 Model B](https://github.com/AntonioBerna/nn-rpi-logic-gates), today I propose the same project using a completely different architecture: GPUs.

Working in the embedded field, exploiting the potential of microcontrollers such as Raspberry Pi, Arduino or ESP32 or even better digging deep and using AVR or ARM microcontrollers directly, we do not realize that in some contexts performance is fundamental and often represents an essential requirement for the functioning of a digital system. Often for these reasons we use the famous FPGA (which we may discuss in other repositories) but today, thanks to the great growth of NVIDIA, we can use the GPU, even the one inside our computer, to train artificial intelligence models, and more specifically, in our case, real neural networks.

> [!WARNING]
> If you do not know how to configure the drivers of your NVIDIA GPU you can follow my repository [nvidia-devices](https://github.com/AntonioBerna/nvidia-devices).

> [!WARNING]
> If any of you are unfamiliar with boolean algebra, I have explained how logic gates work in [this project](https://github.com/AntonioBerna/nn-rpi-logic-gates).

# Mini docs

To use this project you don't have to configure anything special, as it happened in the case of Raspberry Pi. So it will be enough to clone this repository with the following command:

```
git clone https://github.com/AntonioBerna/nn-gnu-logic-gates.git
```

So, by accessing the working directory with the command `cd nn-gpu-logic-gates` and using the command:

```
cmake . -B build
```

let's generate the `Makefile` for our operating system. Then using the command

```
cd build && make
```

and finally the command:

```
./nn-gnu-logic-gates
```

we obtain:

```
AND model in progress.
[0 0] -> 0.00339417
[0 1] -> 0.0241356
[1 0] -> 0.0222825
[1 1] -> 0.967421

OR model in progress.
[0 0] -> 0.0303509
[0 1] -> 0.982423
[1 0] -> 0.982574
[1 1] -> 0.994855

NAND model in progress.
[0 0] -> 0.998641
[0 1] -> 0.978928
[1 0] -> 0.977226
[1 1] -> 0.033984

NOR model in progress.
[0 0] -> 0.967394
[0 1] -> 0.0185643
[1 0] -> 0.0184932
[1 1] -> 0.00554564

XOR model in progress.
[0 0] -> 0.0560026
[0 1] -> 0.955456
[1 0] -> 0.948381
[1 1] -> 0.0470075
```
