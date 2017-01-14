import six
import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict


class BN_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(BN_Conv, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(in_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return self.conv(self.bn(x, test=not train))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class Conv_BN_ReLU(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(Conv_BN_ReLU, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(out_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return F.relu(self.bn(self.conv(x), test=not train))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class BN_Conv_BN_ReLU_Conv_BN(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_sizes=(3, 3), strides=(1, 1), pads=(1, 1), probability=(1.0, 1.0)):
        super(BN_Conv_BN_ReLU_Conv_BN, self).__init__()
        modules = []
        modules += [('bn1', L.BatchNormalization(in_channel))]
        modules += [('bn1_expansion', L.BatchNormalization(in_channel))]
        modules += [('conv1', L.Convolution2D(in_channel, in_channel, filter_sizes[0], strides[0], pads[0]))]
        modules += [('conv1_expansion', L.Convolution2D(in_channel, out_channel - in_channel, filter_sizes[0], strides[0], pads[0]))]
        modules += [('bn2', L.BatchNormalization(in_channel))]
        modules += [('bn2_expansion', L.BatchNormalization(out_channel - in_channel))]
        modules += [('conv2', L.Convolution2D(out_channel, in_channel, filter_sizes[1], strides[1], pads[1]))]
        modules += [('conv2_expansion', L.Convolution2D(out_channel, out_channel - in_channel, filter_sizes[1], strides[1], pads[1]))]
        modules += [('bn3', L.BatchNormalization(in_channel))]
        modules += [('bn3_expansion', L.BatchNormalization(out_channel - in_channel))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filter_sizes = filter_sizes
        self.strides = strides
        self.pads = pads
        self.probability = probability

    def _conv_initialization(self, conv):
        conv.W.data = self.weight_relu_initialization(conv)
        conv.b.data = self.bias_initialization(conv, constant=0)

    def weight_initialization(self):
        self._conv_initialization(self.conv1)
        self._conv_initialization(self.conv2)
        self._conv_initialization(self.conv1_expansion)
        self._conv_initialization(self.conv2_expansion)

    @staticmethod
    def concatenate_zero_pad(x, h_shape, volatile, h_type):
        _, x_channel, _, _ = x.data.shape
        batch, h_channel, h_y, h_x = h_shape
        if x_channel == h_channel:
            return x
        pad = chainer.Variable(np.zeros((batch, h_channel - x_channel, h_y, h_x), dtype=np.float32), volatile=volatile)
        if h_type is not np.ndarray:
            pad.to_gpu()
        return F.concat((x, pad))

    @staticmethod
    def create_zero_pad(h_shape, volatile, dtype, h_type):
        pad = chainer.Variable(np.zeros(h_shape, dtype=dtype), volatile=volatile)
        if h_type is not np.ndarray:
            pad.to_gpu()
        return pad

    def maybe_pooling(self, x):
        if 2 in self.strides:
            return F.average_pooling_2d(x, 1, 2, 0)
        return x

    def __call__(self, x, train=False):
        batch, channel, height, width = x.data.shape
        _, in_channel, _, _ = self.conv1.W.data.shape
        diff_channel = int(in_channel - channel)
        if diff_channel >= 1:
            x = self.concatenate_zero_pad(x, (batch, in_channel, height, width), x.volatile, type(x.data))
        _x = x
        p = (self.probability[0] >= np.random.rand(), self.probability[1] >= np.random.rand())
        # TODO: sphagettis code here, refractoring is necessary
        if train is False or p == (True, True):
            h = self.bn1(x, test=not train)
            h = self.conv1(h)
            h = self.bn2(h, test=not train)
            h_exp = self.bn1_expansion(x, test=not train)
            h_exp = self.conv1_expansion(h_exp)
            h_exp = self.bn2_expansion(h_exp, test=not train)
            h = F.concat((h, h_exp), axis=1)
            x = F.relu(h)
            h = self.conv2(x)
            h = self.bn3(h, test=not train)
            h_exp = self.conv2_expansion(x)
            h_exp = self.bn3_expansion(h_exp, test=not train)
            if train is False:
                h = h * self.probability[0]
                h_exp = h_exp * self.probability[1]
            h = F.concat((h, h_exp), axis=1)

        elif p == (True, False):
            h = self.bn1(x, test=not train)
            h = self.conv1(h)
            h = self.bn2(h, test=not train)
            batch, _, height, width = h.data.shape
            ch, _,  _, _ = self.conv1_expansion.W.data.shape
            x = F.concat((F.relu(h), self.create_zero_pad((batch, ch, height, width), h.volatile, h.dtype, type(h.data))), axis=1)
            h = self.conv2(x)
            h = self.bn3(h, test=not train)
            batch, _, height, width = h.data.shape
            ch, _,  _, _ = self.conv2_expansion.W.data.shape
            h = F.concat((h, self.create_zero_pad((batch, ch, height, width), h.volatile, h.dtype, type(h.data))), axis=1)

        elif p == (False, True):
            h_exp = self.bn1_expansion(x, test=not train)
            h_exp = self.conv1_expansion(h_exp)
            h_exp = self.bn2_expansion(h_exp, test=not train)
            batch, _, height, width = h_exp.data.shape
            ch, _,  _, _ = self.conv1.W.data.shape
            x = F.concat((self.create_zero_pad((batch, ch, height, width), h_exp.volatile, h_exp.dtype, type(h_exp.data)),  F.relu(h_exp)), axis=1)
            h_exp = self.conv2_expansion(x)
            h_exp = self.bn3_expansion(h_exp, test=not train)
            batch, _, height, width = h_exp.data.shape
            ch, _,  _, _ = self.conv2.W.data.shape
            h = F.concat((self.create_zero_pad((batch, ch, height, width), h_exp.volatile, h_exp.dtype, type(h_exp.data)), h_exp), axis=1)

        elif p == (False, False):
            return _x

        h = h + self.concatenate_zero_pad(self.maybe_pooling(_x), h.data.shape, h.volatile, type(h.data))
        return h

    @staticmethod
    def _count_conv(conv):
        return functools.reduce(lambda a, b: a * b, conv.W.data.shape)

    def count_parameters(self):
        links = [self.conv1, self.conv2, self.conv1_expansion, self.conv2_expansion]
        return int(np.sum([self._count_conv(link) for link in links]))


class PyramidalResNet(nutszebra_chainer.Model):

    def __init__(self, category_num, N=(int(182 / 3. / 2.),) * 3, initial_channel=16, alpha=150, p=(1.0, 0.5)):
        super(PyramidalResNet, self).__init__()
        # conv
        modules = [('conv1', Conv_BN_ReLU(3, initial_channel, 3, 1, 1))]
        # strides
        strides = [[(1, 1) for _ in six.moves.range(i)] for i in N]
        strides[1][0] = (1, 2)
        strides[2][0] = (1, 2)
        # channels
        out_channels = PyramidalResNet.linear_schedule(initial_channel, initial_channel + alpha, N)
        # drop probability
        drop_probability = PyramidalResNet.linear_schedule(p[0], p[1], N)
        in_channel = initial_channel
        for i in six.moves.range(len(strides)):
            for ii in six.moves.range(len(strides[i])):
                out_channel = int(out_channels[i][ii])
                stride = strides[i][ii]
                probability = (drop_probability[i][ii], ) * 2
                name = 'res_block{}_{}'.format(i, ii)
                modules.append((name, BN_Conv_BN_ReLU_Conv_BN(in_channel, out_channel, (3, 3), stride, (1, 1), probability=probability)))
                # in_channel is changed
                in_channel = out_channel
        modules += [('linear', BN_Conv(out_channel, category_num, 1, 1, 0))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.strides = strides
        self.out_channels = out_channels
        self.category_num = category_num
        self.N = N
        self.initial_channel = initial_channel
        self.alpha = alpha
        self.drop_probability = drop_probability
        self.name = 'pyramidal_resnet_{}_{}_{}_{}'.format(category_num, N, initial_channel, alpha)

    @staticmethod
    def linear_schedule(bottom_layer, top_layer, N):
        total_block = sum(N)

        def y(x):
            return (float(-1 * bottom_layer) + top_layer) / (total_block) * x + bottom_layer
        theta = []
        count = 1
        for num in N:
            tmp = []
            for i in six.moves.range(count, count + num):
                tmp.append(y(i))
            theta.append(tmp)
            count += num
        return theta

    def weight_initialization(self):
        for name, link in self.modules:
            link.weight_initialization()

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += link.count_parameters()
        return count

    def __call__(self, x, train=False):
        h = self.conv1(x, train=train)
        for i in six.moves.range(len(self.strides)):
            for ii in six.moves.range(len(self.strides[i])):
                name = 'res_block{}_{}'.format(i, ii)
                h = self[name](h, train=train)
        batch, channels, height, width = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (height, width)), (batch, channels, 1, 1))
        _, in_channel, _, _ = self.linear.conv.W.data.shape
        h = BN_Conv_BN_ReLU_Conv_BN.concatenate_zero_pad(h, (batch, in_channel, 1, 1), h.volatile, type(h.data))
        return F.reshape(self.linear(h, train), (batch, self.category_num))

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        print(loss.data)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
