import torch
import torch.nn.functional as F
from cam.basecam import *


class ScoreCAM(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
        
        # predication on raw input
        logit = self.model_arch(input).cuda()
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()
        
        logit = F.softmax(logit)

        if torch.cuda.is_available():
          predicted_class= predicted_class.cuda()
          score = score.cuda()
          logit = logit.cuda()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']


        b, k, u, v = activations.size()
        
        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
          for i in range(k):

              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
              
              if saliency_map.max() == saliency_map.min():
                continue
              
              # normalize to 0-1
              norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

              # how much increase if keeping the highlighted region
              # predication on masked input
              output = self.model_arch(input * norm_saliency_map)
              output = F.softmax(output)
              score = output[0][predicted_class]

              score_saliency_map +=  score * saliency_map
                
        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def forward_multistream(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input[0].size()

        # predication on raw input
        if len(input) == 1:
            logit = self.model_arch(input[0]).cuda()
        else:
            logit = self.model_arch(input).cuda()

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        logit = F.softmax(logit)

        if torch.cuda.is_available():
            predicted_class = predicted_class.cuda()
            score = score.cuda()
            logit = logit.cuda()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)

        activations = list()
        for _, item in self.activations.items():
            activations.append(item)
        b, k, u, v = activations[0].size()

        score_saliency_maps = list()
        for _ in input:
            score_saliency_maps.append(torch.zeros((1, 1, h, w)))

        if torch.cuda.is_available():
            for inp_i in range(len(input)):
                activations[inp_i] = activations[0].cuda()
                score_saliency_maps[inp_i] = score_saliency_maps[inp_i].cuda()

        with torch.no_grad():
            for i in range(k):
                norm_saliency_maps = list()
                saliency_maps = list()
                skip=False
                for inp_i in range(len(input)):
                    # upsampling
                    saliency_map = torch.unsqueeze(activations[inp_i][:, i, :, :], 1)
                    saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
                    saliency_maps.append(saliency_map)

                    if saliency_map.max() == saliency_map.min():
                        skip=True
                        break

                    # normalize to 0-1
                    norm_saliency_maps.append((saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min()))

                if skip:
                    continue

                salient_input = list()
                for inp, norm_saliency_map in zip(input, norm_saliency_maps):
                    salient_input.append(inp * norm_saliency_map)

                # how much increase if keeping the highlighted region
                # predication on masked input
                if len(input) == 1:
                    output = self.model_arch(salient_input[0])
                else:
                    output = self.model_arch(salient_input)
                output = F.softmax(output)
                score = output[0][predicted_class]

                for inp_i in range(len(input)):
                    score_saliency_maps[inp_i] += score * saliency_maps[inp_i]

        for inp_i in range(len(input)):
            score_saliency_maps[inp_i] = F.relu(score_saliency_maps[inp_i])
            score_saliency_map_min, score_saliency_map_max = score_saliency_maps[inp_i].min(), score_saliency_maps[inp_i].max()

            if score_saliency_map_min == score_saliency_map_max:
                return None

            score_saliency_maps[inp_i] = (score_saliency_maps[inp_i] - score_saliency_map_min).div(
                score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_maps

    def __call__(self, input, class_idx=None, retain_graph=False):
        if isinstance(input, list):
            return self.forward_multistream(input, class_idx, retain_graph)
        else:
            return self.forward(input, class_idx, retain_graph)
