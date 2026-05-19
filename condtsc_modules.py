import time
import numpy as np
from collections import Counter

import torch
import torch.nn as nn

from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils import weight_norm


# DATA AUGMENTATIONS- applied to synth data to prevent overfitting

def jitter(x, args):
    # adds small awgn to input
    if hasattr(args, "device"):
        return x + torch.normal(mean=0., std=args.jitter_ratio, size=x.shape).to(args.device)
    return x + np.random.normal(loc=0., scale=args.jitter_ratio, size=x.shape)

def DataTransform_TD(sample, args):
    # right now only noise added- didn't have compute power to do scaling too
    return jitter(sample, args)

# dataset initialisation- k center selection instead of random

class Base:
    def __init__(self, data, args, device="cuda"):
        self.data = data
        self.args = args
        self.device = device
        self.nnodes_syn = args.ipc
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)

    def generate_labels_syn(self, data):
        counter = Counter(data.labels_train)
        num_class_dict = {}
        labels_syn = []
        self.syn_class_indices = {}
        total = 0

        for idx, (class_id, _) in enumerate(sorted(counter.items(), key=lambda x: x[1])):
            num_class_dict[class_id] = max(self.args.ipc, 1)
            total += num_class_dict[class_id]
            self.syn_class_indices[class_id] = [len(labels_syn), len(labels_syn) + num_class_dict[class_id]]
            labels_syn += [class_id] * num_class_dict[class_id]

        self.num_class_dict = num_class_dict
        return labels_syn

    def select(self):
        return

class KCenter(Base):
    # greedily picks samples that maximize the min distance to already-selected center to give good starting point

    def __init__(self, data, args, device="cuda", **kwargs):
        super(KCenter, self).__init__(data, args, device, **kwargs)

    def select(self, embeds, inductive=True):
        idx_train = np.arange(len(self.data.idx_train)) if inductive else self.data.idx_train
        labels_train = self.data.labels_train
        selected_indices = []

        for class_id, count in self.num_class_dict.items():
            idx = idx_train[labels_train == class_id]
            feature = embeds[idx]

            # start from sample closest to mean
            mean = torch.mean(feature, dim=0, keepdim=True)
            distances = torch.cdist(feature, mean)[:, 0]
            center_indices = torch.argsort(distances)[:1].tolist()

            # add farthest point from current centers
            for _ in range(count - 1):
                selected_features = feature[torch.tensor(center_indices, dtype=torch.long, device=feature.device)]
                dist_to_centers = torch.cdist(feature, selected_features)
                min_dist, _ = torch.min(dist_to_centers, dim=-1)
                farthest = torch.argmax(min_dist).item()
                center_indices.append(farthest)

            selected_indices.append(idx[center_indices])

        return np.hstack(selected_indices)


# MODEL ARCHITECTURES

class _RNN_Base(nn.Module):
    def __init__(self, c_in, c_out, hidden_size=100, n_layers=1, bias=True,
                 rnn_dropout=0, bidirectional=False, fc_dropout=0., init_weights=True):
        super(_RNN_Base, self).__init__()

        self.rnn = self._cell(c_in, hidden_size, num_layers=n_layers, bias=bias,
                              batch_first=True, dropout=rnn_dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Identity()
        self.final_rep = hidden_size * (1 + bidirectional)
        self.fc = nn.Linear(self.final_rep, c_out)

        if init_weights:
            self.apply(self._weights_init)

    def forward(self, x):
        x = x.transpose(2, 1)
        output, _ = self.rnn(x)
        output = output[:, -1]
        return self.fc(self.dropout(output))

    def embed(self, x):
        x = x.transpose(2, 1)
        output, _ = self.rnn(x)
        return output[:, -1]

    def _weights_init(self, m):
        for name, params in m.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_normal_(params)
            elif "weight_hh" in name:
                nn.init.orthogonal_(params)
            elif "bias_ih" in name:
                params.data.fill_(0)
                n = params.size(0)
                params.data[(n // 4):(n // 2)].fill_(1)   # forget gate bias = 1
            elif "bias_hh" in name:
                params.data.fill_(0)

class LSTM(_RNN_Base):
    _cell = nn.LSTM

class Chomp1d(nn.Module):
    #Trims extra padding added by causal convolutions so output length matches input

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module): # residual block in TCN

    def __init__(self, ni, nf, ks, stride, dilation, padding, dropout=0.):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(ni, nf, ks, stride=stride, padding=padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(nf, nf, ks, stride=stride, padding=padding, dilation=dilation))
        self.net = nn.Sequential(
            self.conv1, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            self.conv2, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
        )

        # 1x1 conv to match channels if ni not equal to nf
        self.downsample = nn.Conv1d(ni, nf, 1) if ni != nf else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)

def TemporalConvNet(c_in, layers, ks=2, dropout=0.): # stacking the temporal blocks
    return nn.Sequential(*[
        TemporalBlock(
            ni=c_in if i == 0 else layers[i - 1],
            nf=layers[i],
            ks=ks,
            stride=1,
            dilation=2 ** i,
            padding=(ks - 1) * 2 ** i,
            dropout=dropout
        )
        for i in range(len(layers))
    ])


class GAP1d(nn.Module):
    #Global average pooling over time
    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
    def forward(self, x):
        return self.gap(x).reshape(x.shape[0], -1)


class TCN(nn.Module): # tcn stacks conv blocks over time then classifies

    def __init__(self, c_in, c_out, layers=4 * [32], ks=7, conv_dropout=0., fc_dropout=0.):
        super(TCN, self).__init__()

        self.tcn = TemporalConvNet(c_in, layers, ks=ks, dropout=conv_dropout)
        self.gap = GAP1d()
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1], c_out)
        self.final_rep = layers[-1]
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)

    def embed(self, x):
        x = self.tcn(x)
        x = self.gap(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class MLP(nn.Module): # 3 layer MLP

    def __init__(self, input_dim, hid_dim, hid2_dim, out_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid_dim, hid2_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hid2_dim, out_dim)
        self.final_rep = hid2_dim

    def forward(self, x):
        x = self.relu(self.fc1(x.reshape(x.shape[0], -1)))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

    def embed(self, x):
        """Returns the second hidden layer's activations."""
        x = self.relu(self.fc1(x.reshape(x.shape[0], -1)))
        return self.fc2(x)


class ConvNet(nn.Module): # 1d cnn for teacher

    def __init__(self, channel, num_classes, net_width, net_depth,
                 net_act, net_norm, net_pooling, im_size):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size
        )
        num_feat = shape_feat[0] * shape_feat[1]
        self.classifier = nn.Linear(num_feat, num_classes)
        self.final_rep = num_feat

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)

    def embed(self, x):
        out = self.features(x)
        return out.view(out.size(0), -1)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        shape_feat = [channel, im_size]

        for _ in range(net_depth):
            layers.append(nn.Conv1d(in_channels, net_width, kernel_size=3, padding=1))
            shape_feat[0] = net_width

            if net_norm == "BN":
                layers.append(nn.BatchNorm1d(net_width, affine=True))
            elif net_norm == "IN":
                layers.append(nn.InstanceNorm1d(net_width, affine=True))

            layers.append(nn.ReLU(inplace=True))
            in_channels = net_width

            if net_pooling == "maxpooling":
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                shape_feat[1] //= 2

        return nn.Sequential(*layers), shape_feat


# selecting network

def get_network(args):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    dual = getattr(args, "dual", 0)

    if args.model == "MLP":
        hid = 64 if dual else 128
        return MLP(input_dim=args.time_step * args.channel, hid_dim=128, hid2_dim=hid, out_dim=args.num_classes)

    elif args.model == "CNNBN":
        w = 16 if dual else 32
        return ConvNet(channel=args.channel, num_classes=args.num_classes, net_width=w,
                       net_depth=3, net_act="relu", net_norm="BN", net_pooling="maxpooling", im_size=args.time_step)

    elif args.model == "TCN":
        ch = 48 if dual else 64
        return TCN(c_in=args.channel, c_out=args.num_classes, layers=[ch])

    elif args.model == "LSTM":
        h = 64 if dual else 100
        return LSTM(c_in=args.channel, c_out=args.num_classes, hidden_size=h)

    else:
        raise NotImplementedError(f"Model not supported.")

# dual domain implenentation

class Dualmodel(nn.Module):

    def __init__(self, args):
        super(Dualmodel, self).__init__()

        self.args = args

        self.t_model = get_network(args).to(args.device)   # time domain encoder
        self.f_model = get_network(args).to(args.device)   # frequency domain encoder

        self.t_model.train()
        self.f_model.train()

        self.final_rep = self.t_model.final_rep
        self.mlp = nn.Linear(self.final_rep * 2, args.num_classes).to(args.device)

    def forward(self, x):
        x_f = torch.fft.rfft(x, dim=-1)
        x_f = torch.view_as_real(x_f).reshape(x_f.shape[0], x_f.shape[1], -1)
        x_f = x_f[:, :, :x.shape[-1]]   # trim to same length as time signal

        t_emb = self.t_model.embed(x)
        f_emb = self.f_model.embed(x_f)

        emb = torch.cat([t_emb, f_emb], dim=-1)

        # uncomment below line for ablation study to see result without freq domain
        # emb = torch.cat([t_emb, torch.zeros_like(f_emb)], dim=-1)

        out = self.mlp(emb)
        return out, t_emb, f_emb