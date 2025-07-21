import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import random


class GMD:
    def __init__(self, optimizer, reduction='mean', writer=None):
        self._optim, self._reduction = optimizer, reduction
        self.iter = 0
        self.writer = writer

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives, ddp_model=None):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''

        (grads_rgb, shapes_rgb, has_grads_rgb, grads_t, shapes_t, has_grads_t,
         grads_sam2Decoder, shapes_sam2Decoder, has_grads_sam2Decoder) = self._pack_grad(objectives, ddp_model)
        pc_grad_rgb = self._project_conflicting(grads_rgb, has_grads_rgb)
        pc_grad_rgb = self._unflatten_grad(pc_grad_rgb, shapes_rgb[0])

        pc_grad_t = self._project_conflicting(grads_t, has_grads_t)
        pc_grad_t = self._unflatten_grad(pc_grad_t, shapes_t[0])

        pc_grad_sam2Decoder = self._project_conflicting(grads_sam2Decoder, has_grads_sam2Decoder)
        pc_grad_sam2Decoder = self._unflatten_grad(pc_grad_sam2Decoder, shapes_sam2Decoder[0])
        pc_grad = pc_grad_sam2Decoder + pc_grad_rgb + pc_grad_t
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        coefs = torch.ones(num_task, dtype=torch.float32, device=grads[0].device)
        for g_i in pc_grad:
            indices = list(range(num_task))
            random.shuffle(list(range(num_task)))
            random.shuffle(grads)
            for index in indices:
                g_j = grads[index]
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    coef = g_i_g_j / (g_j.norm() ** 2)

                    g_i -= coef * g_j
                    coefs[index] -= coef
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)

        self.iter += 1
        # for ii, coef in enumerate(coefs):
        #     self.writer.add_scalar(f'coef/pc_grad_coef_{ii}', coef.item(), self.iter)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives, ddp):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad1: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads_rgb, shapes_rgb, has_grads_rgb = [], [], []
        grads_t, shapes_t, has_grads_t = [], [], []
        grads_sam2Decoder, shapes_sam2Decoder, has_grads_sam2Decoder = [], [], []

        # fusion
        self._optim.zero_grad(set_to_none=True)
        objectives[0].backward(retain_graph=True)
        grad_info = self._retrieve_grad()

        grads_rgb.append(self._flatten_grad_rgb(grad_info))
        has_grads_rgb.append(self._flatten_grad_rgb(grad_info))
        shapes_rgb.append([g['shape'] for g in grad_info if 'rgb' in g['name']])

        grads_t.append(self._flatten_grad_t(grad_info))
        has_grads_t.append(self._flatten_grad_t(grad_info))
        shapes_t.append([g['shape'] for g in grad_info if 'depth' in g['name']])

        grads_sam2Decoder.append(self._flatten_grad_sam2Decoder(grad_info))
        has_grads_sam2Decoder.append(self._flatten_grad_sam2Decoder(grad_info))
        shapes_sam2Decoder.append([g['shape'] for g in grad_info if 'sam_mask_decoder' in g['name']])

        # rgb
        self._optim.zero_grad(set_to_none=True)
        objectives[1].backward(retain_graph=True)

        grad_info = self._retrieve_grad()
        grads_rgb.append(self._flatten_grad_rgb(grad_info))
        has_grads_rgb.append(self._flatten_grad_rgb(grad_info))
        shapes_rgb.append([g['shape'] for g in grad_info if 'rgb' in g['name']])

        grads_sam2Decoder.append(self._flatten_grad_sam2Decoder(grad_info))
        has_grads_sam2Decoder.append(self._flatten_grad_sam2Decoder(grad_info))
        shapes_sam2Decoder.append([g['shape'] for g in grad_info if 'sam_mask_decoder' in g['name']])

        # t
        objectives[2].backward()
        grad_info = self._retrieve_grad()

        grads_t.append(self._flatten_grad_t(grad_info))
        has_grads_t.append(self._flatten_grad_t(grad_info))
        shapes_t.append([g['shape'] for g in grad_info if 'depth' in g['name']])

        grads_sam2Decoder.append(self._flatten_grad_sam2Decoder(grad_info))
        has_grads_sam2Decoder.append(self._flatten_grad_sam2Decoder(grad_info))
        shapes_sam2Decoder.append([g['shape'] for g in grad_info if 'sam_mask_decoder' in g['name']])

        return grads_rgb, shapes_rgb, has_grads_rgb, grads_t, shapes_t, has_grads_t, grads_sam2Decoder, shapes_sam2Decoder, has_grads_sam2Decoder

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad_rgb(self, grads):
        flatten_grad = [g['grad'].flatten() for g in grads if 'rgb' in g['name']]
        return torch.cat(flatten_grad)

    def _flatten_grad_t(self, grads):
        flatten_grad = [g['grad'].flatten() for g in grads if 'depth' in g['name']]
        return torch.cat(flatten_grad)

    def _flatten_grad_sam2Decoder(self, grads):
        flatten_grad = [g['grad'].flatten() for g in grads if 'sam_mask_decoder' in g['name']]
        return torch.cat(flatten_grad)
    def _retrieve_grad(self):
        param_info_list = []  # 存储每个参数的信息
        for group in self._optim.param_groups:
            for p, name in zip(group['params'], group['name']):
                if p.grad is None:
                    # print(name)
                    param_info = {
                        'name': name,
                        'grad': torch.zeros_like(p).to(p.device),
                        'shape': p.shape,
                        'has_grad': torch.zeros_like(p).to(p.device)
                    }
                else:
                    param_info = {
                        'name': name,
                        'grad': p.grad.clone(),
                        'shape': p.grad.shape,
                        'has_grad': torch.ones_like(p).to(p.device)
                    }
                param_info_list.append(param_info)

        return param_info_list
