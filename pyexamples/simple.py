import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input Layer
    to_Conv("input", 24, 1, offset="(0,0,0)", to="(0,0,0)", height=1, depth=24, width=30, caption="Input"),

    # First Convolution + ReLU Layer
    to_ConvConvRelu(name='ccr1', s_filer=30, n_filer=(24, 48), offset="(1,0,0)", to="(input-east)", width=(2, 2), height=1, depth=48, caption="Conv1+ReLU"),
    to_connection("input", "ccr1"),

    # First Pooling Layer
    to_Pool("pool1", offset="(0,0,0)", to="(ccr1-east)", height=1, depth=24, width=15, caption="MaxPool1"),
    to_connection("ccr1", "pool1"),

    # Second Convolution + ReLU Layer
    to_ConvConvRelu(name='ccr2', s_filer=15, n_filer=(48, 96), offset="(1,0,0)", to="(pool1-east)", width=(2, 2), height=1, depth=96, caption="Conv2+ReLU"),
    to_connection("pool1", "ccr2"),

    # Second Pooling Layer
    to_Pool("pool2", offset="(0,0,0)", to="(ccr2-east)", height=1, depth=12, width=7.5, caption="MaxPool2"),
    to_connection("ccr2", "pool2"),

    # Fully Connected Layer (represented as a SoftMax for visualization)
    to_SoftMax("fc", 1, "(3,0,0)", "(pool2-east)", caption="Fully Connected"),
    to_connection("pool2", "fc"),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
