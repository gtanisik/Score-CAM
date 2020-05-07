'''
Part of code borrows from https://github.com/1Konny/gradcam_plus_plus-pytorch
'''

import torch
from score_cam_utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, \
    find_squeezenet_layer, find_layer, find_googlenet_layer, find_mobilenet_layer, find_shufflenet_layer

class BaseCAM(object):
    """ Base class for Class activation mapping.

        : Args
            - **model_dict -** : Dict. Has format as dict(type='vgg', arch=torchvision.models.vgg16(pretrained=True),
            layer_name='features',input_size=(224, 224)).

    """

    def __init__(self, model_dict):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        
        self.model_arch = model_dict['arch']
        self.model_arch.eval()
        if torch.cuda.is_available():
          self.model_arch.cuda()
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            if torch.cuda.is_available():
              self.gradients['value0'] = grad_output[0].cuda()
            else:
              self.gradients['value0'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            if torch.cuda.is_available():
              self.activations['value0'] = output.cuda()
            else:
              self.activations['value0'] = output
            return None

        def backward_hook1(module, grad_input, grad_output):
            if torch.cuda.is_available():
              self.gradients['value1'] = grad_output[0].cuda()
            else:
              self.gradients['value1'] = grad_output[0]
            return None

        def forward_hook1(module, input, output):
            if torch.cuda.is_available():
              self.activations['value1'] = output.cuda()
            else:
              self.activations['value1'] = output
            return None

        def backward_hook2(module, grad_input, grad_output):
            if torch.cuda.is_available():
              self.gradients['value2'] = grad_output[0].cuda()
            else:
              self.gradients['value2'] = grad_output[0]
            return None

        def forward_hook2(module, input, output):
            if torch.cuda.is_available():
              self.activations['value2'] = output.cuda()
            else:
              self.activations['value2'] = output
            return None

        def backward_hook3(module, grad_input, grad_output):
            if torch.cuda.is_available():
              self.gradients['value3'] = grad_output[0].cuda()
            else:
              self.gradients['value3'] = grad_output[0]
            return None

        def forward_hook3(module, input, output):
            if torch.cuda.is_available():
              self.activations['value3'] = output.cuda()
            else:
              self.activations['value3'] = output
            return None

        if 'vgg' in model_type.lower():
            self.target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            self.target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            self.target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            self.target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            self.target_layer = find_squeezenet_layer(self.model_arch, layer_name)
        elif 'googlenet' in model_type.lower():
            self.target_layer = find_googlenet_layer(self.model_arch, layer_name)
        elif 'shufflenet' in model_type.lower():
            self.target_layer = find_shufflenet_layer(self.model_arch, layer_name)
        elif 'mobilenet' in model_type.lower():
            self.target_layer = find_mobilenet_layer(self.model_arch, layer_name)
        else:
            self.target_layer = find_layer(self.model_arch, layer_name)

        if isinstance(self.target_layer, list):
            self.target_layer[0].register_forward_hook(forward_hook)
            self.target_layer[0].register_backward_hook(backward_hook)
            self.target_layer[1].register_forward_hook(forward_hook1)
            self.target_layer[1].register_backward_hook(backward_hook1)
            if len(self.target_layer) >= 3:
                self.target_layer[2].register_forward_hook(forward_hook2)
                self.target_layer[2].register_backward_hook(backward_hook2)
            if len(self.target_layer) >= 4:
                self.target_layer[3].register_forward_hook(forward_hook3)
                self.target_layer[3].register_backward_hook(backward_hook3)
        else:
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=False):
        return None

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)