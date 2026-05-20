from nets.layer import *
import copy
from spikingjelly.activation_based import functional
feature_cfg = {
    'VGG5': [64, 'A', 128, 128, 'A', 'AA'],
    'VGG9': [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512, 'AA'],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512, 'AA'],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512],
    'CIFAR': [128, 256, 'A', 512, 'A', 1024, 512],
    'VGGSNN_CIFAR': [64, 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512, 'AA3'],
    'VGGSNN_DVS': [64, 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512, 'A'],
}

clasifier_cfg = {
    'VGG5': [128, 10],
    'VGG11': [512, 10],
    'VGG13': [512, 10],
    'VGG16': [2048, 4096, 4096, 10],
    'VGG19': [2048, 4096, 4096, 10],
    'VGGSNN_CIFAR': [4608, 100],
    'VGGSNN_DVS': [4608, 10]
}

class BiasLayer_BIC(nn.Module):
    def __init__(self):
        super(BiasLayer_BIC, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
            self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())

class VGG(nn.Module):
    def __init__(self, architecture='VGG16', kernel_size=3, in_channel=3, use_bias=True,
                 num_class=10, **kwargs_spikes):
        super(VGG, self).__init__()
        # print(kwargs_spikes)
        self.architecture = architecture
        self.kwargs_spikes = kwargs_spikes['kwargs_spikes']
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.use_bias = use_bias
        self.num_class = num_class
        clasifier_cfg[architecture][-1] = num_class
        self.feature = self._make_feature(feature_cfg[architecture])
        self.classifier = self._make_classifier(clasifier_cfg[architecture])
        self.readout = ReadOut()
        self._initialize_weights()
        self.model_name = kwargs_spikes["model_name"]
        if self.model_name == "bic" or self.model_name == "icarl" or self.model_name == "replay" or self.model_name == "finetune" or self.model_name == "lwf":
            if self.model_name == "bic":
                self.bias_layers = nn.ModuleList([])
                self.task_sizes = []
            self.step_mode = kwargs_spikes["step_mode"]
            self.T_max = kwargs_spikes["T"]
    def _make_feature(self, config):
        layers = []
        channel = self.in_channel
        for x in config:
            if x == 'A':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))   # Max_pool
            elif x == 'AA':
                layers.append(nn.AdaptivePool2d((1, 1)))
            elif x == 'AA3':
                layers.append(nn.AdaptiveAvgPool2d((3, 3)))
            else:
                layers.append(nn.Conv2d(in_channels=channel, out_channels=x, kernel_size=self.kernel_size,
                                        stride=1, padding=self.kernel_size // 2, bias=self.use_bias))

                layers.append(nn.BatchNorm2d(x))
                layers.append(LIFLayer(**self.kwargs_spikes))    # nn.relu
                channel = x
        return nn.Sequential(*layers)

    def _make_classifier(self, config):
        layers = []
        for i in range(len(config) - 1):
            layers.append(nn.Linear(config[i], config[i + 1], bias=self.use_bias))
            layers.append(LIFLayer(**self.kwargs_spikes))  # nn.relu
        layers.pop()
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.5)
                # m.weight.data.normal_(0, 0.5)
                # n = m.weight.size(1)
                # m.weight.data.normal_(0, 1.0 / float(n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def updata_classifier(self, classes):
        self.num_class = classes
        clasifier_cfg[self.architecture][-1] = classes
        classifier = copy.deepcopy(self.classifier)
        nb_output = self.classifier[-1].out_features
        weight = copy.deepcopy(self.classifier[-1].weight.data)
        bias = copy.deepcopy(self.classifier[-1].bias.data)
        classifier[-1] = nn.Linear(self.classifier[-1].in_features, classes, bias=self.use_bias)
        classifier[-1].weight.data[:nb_output] = weight
        classifier[-1].bias.data[:nb_output] = bias

        del self.classifier
        self.classifier = classifier
        if self.model_name == "bic":
            new_task_size = classes - sum(self.task_sizes)
            self.task_sizes.append(new_task_size)
            self.bias_layers.append(BiasLayer_BIC())
    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        if self.model_name == "bic":
            for i, layer in enumerate(self.bias_layers):
                x = layer(
                    x, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
        x = self.readout(x)
        return x
    @property
    def feature_dim(self):
        return self.classifier[-1].in_features

    def extract_res(self, x, dataset):
        functional.reset_net(self.feature)
        if self.step_mode == 's':
            out_spikes = []
            if dataset == 'dvs':
                x = x.permute(1, 0, 2, 3, 4)
            for t in range(self.T_max):
                if dataset == 'dvs':
                    out = self.feature(x[t])
                else:
                    out = self.feature(x)
                out = out.view(out.shape[0], -1)
                out = self.classifier(out)
                out_spikes.append(out)
            output = torch.stack(out_spikes, dim=0)
            avg_fr = output.mean(dim=0)
        else:
            if dataset == 'dvs':
                in_data = x.permute(1, 0, 2, 3, 4)
            else:
                in_data, _ = torch.broadcast_tensors(x, torch.zeros((self.T_max,) + x.shape))
            in_data = in_data.reshape(-1, *in_data.shape[2:])
            output = self.feature(in_data)
            output = output.view(output.shape[0], -1)
            output = self.classifier(output)
            avg_fr = output.mean(dim=0)
        x = avg_fr.view(avg_fr.shape[0], -1)
        return x

    def extract_vector(self, x, dataset):
        functional.reset_net(self.feature)
        if self.step_mode == 's':
            out_spikes = []
            if dataset == 'dvs':
                x = x.permute(1, 0, 2, 3, 4)
            for t in range(self.T_max):
                if dataset == 'dvs':
                    out = self.feature(x[t])
                else:
                    out = self.feature(x)
                out_spikes.append(out)
            output = torch.stack(out_spikes, dim=0)
            avg_fr = output.mean(dim=0)
        else:
            if dataset == 'dvs':
                in_data = x.permute(1, 0, 2, 3, 4)
            else:
                in_data, _ = torch.broadcast_tensors(x, torch.zeros((self.T_max,) + x.shape))
            in_data = in_data.reshape(-1, *in_data.shape[2:])
            output = self.feature(in_data)
            avg_fr = output.mean(dim=0)
        x = avg_fr.view(avg_fr.shape[0], -1)
        return x

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
def vggsnn_cifar(num_classes=10, in_channel=3, **kwargs):
    return VGG(architecture="VGGSNN_CIFAR", in_channel=in_channel, num_class=num_classes, **kwargs)

def vggsnn_dvs(num_classes=10, in_channel=3, **kwargs):
    return VGG(architecture="VGGSNN_DVS", in_channel=in_channel, num_class=num_classes, **kwargs)

def vgg11(num_classes=10, in_channel=3, **kwargs):
    return VGG(architecture="VGG11", in_channel=in_channel, num_class=num_classes, **kwargs)


def vgg13(num_classes=10, in_channel=3, **kwargs):
    return VGG(architecture="VGG13", in_channel=in_channel, num_class=num_classes, **kwargs)
