import os
import torch
import torch.nn as nn
from src.egnn import Dynamics
from src.edm import EDM, InpaintingEDM
from src import utils
from typing import Dict, List, Optional
from tqdm import tqdm
from src.datasets import ProtacDataset, get_dataloader, collate, create_templates_for_linker_generation
from torch import Tensor
from torch import acos, sin, pi, cos
from src.visualizer import save_xyz_file
import numpy as np
import random




def get_activation(activation):
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        raise Exception("activation fn not supported yet. Add it here.")

class DDPM(nn.Module):
    def __init__(
        self,
        in_node_nf, n_dims, context_node_nf, hidden_nf, activation, tanh, n_layers, attention, norm_constant,
        inv_sublayers, sin_embedding, normalization_factor, aggregation_method,
        diffusion_steps, diffusion_noise_schedule, diffusion_noise_precision, diffusion_loss_type,
        normalize_factors, include_charges, model,
        batch_size, lr, torch_device, test_epochs, n_stability_samples,
        normalization=None, log_iterations=None, samples_dir=None, data_augmentation=False,
        center_of_mass='fragments', inpainting=False, anchors_context=True,
        ):
        super(DDPM, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        # torch_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.torch_device = torch_device
        self.include_charges = include_charges
        self.test_epochs = test_epochs
        self.n_stability_samples = n_stability_samples
        self.log_iterations = log_iterations
        self.samples_dir = samples_dir
        self.data_augmentation = data_augmentation
        self.center_of_mass = center_of_mass
        self.inpainting = inpainting
        self.loss_type = diffusion_loss_type

        self.n_dims = n_dims
        self.num_classes = in_node_nf - include_charges
        self.include_charges = include_charges
        self.anchors_context = anchors_context

        if type(activation) is str:
            activation = get_activation(activation)

        dynamics_class = Dynamics
        
        dynamics = dynamics_class(
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            context_node_nf=context_node_nf,
            device=torch_device,
            hidden_nf=hidden_nf,
            activation=activation,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            model=model,
            normalization=normalization,
            centering=inpainting,
        )
        edm_class = InpaintingEDM if inpainting else EDM
        self.edm = edm_class(
            dynamics=dynamics,
            in_node_nf=in_node_nf, # 8
            n_dims=n_dims, # 3
            timesteps=diffusion_steps, # 500
            noise_schedule=diffusion_noise_schedule, # polynomial_2
            noise_precision=diffusion_noise_precision, # 1e-5
            loss_type=diffusion_loss_type, # l2
            norm_values=normalize_factors, # [1, 4, 10]
        )
        self.init_optimizer()

    def init_optimizer(self):
        self.optim = torch.optim.AdamW(self.edm.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)


    def forward(self, data, training):
        x = data['positions'] # [128, 33, 3] [batchsize, max_num_of_atoms, 3Dcoordinates]
        h = data['one_hot'] # [128, 33, 8] one-hot, 8 types of atoms 
        node_mask = data['atom_mask'] # [128, 33, 1] flag it 1 if it is an node
        edge_mask = data['edge_mask'] # [139392, 1] 128*33*33, 
        anchors = data['anchors'] # [128, 33, 1] flag it 1 if an anchor node
        fragment_mask = data['fragment_mask'] # [128, 33, 1] flag it 1 if a fragment node
        linker_mask = data['linker_mask'] # [128, 33, 1] flag it 1 if a linker node
        
        if self.anchors_context:
            context = torch.cat([anchors, fragment_mask], dim=-1)
        else:
            context = fragment_mask


        # Removing COM of fragment from the atom coordinates
        if self.inpainting:
            center_of_mass_mask = node_mask
        elif self.center_of_mass == 'fragments':
            center_of_mass_mask = fragment_mask
        elif self.center_of_mass == 'anchors':
            center_of_mass_mask = anchors
        else:
            raise NotImplementedError(self.center_of_mass)
        
        x = utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
        utils.assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask)
     
        # Applying random rotation
        if training and self.data_augmentation:
            x = utils.random_rotation(x)

        return self.edm.forward(
            x=x,
            h=h,
            node_mask=node_mask,
            fragment_mask=fragment_mask,
            linker_mask=linker_mask,
            edge_mask=edge_mask,
            context=context
        )
    

    def training_step(self, data, idx):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=True)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)

        training_metrics = {
            'loss': loss, # 只传递这个loss
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }
              
        return training_metrics  # loss

    def validation_step(self, data, idx):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=False)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)
        return {
            'loss': loss,
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }

 
    def training_epoch_end(self, training_step_outputs):
        avg_loss = {}
        for metric in training_step_outputs.keys():
            avg_metric = self.aggregate_metric(training_step_outputs, metric)
            avg_loss[metric] = avg_metric
        return avg_loss
        

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = {}
        for metric in validation_step_outputs.keys():
            avg_metric = self.aggregate_metric(validation_step_outputs, metric)
            avg_loss[metric] = avg_metric

        # if (self.current_epoch + 1) % self.test_epochs == 0:
        #     sampling_results = self.sample_and_analyze(self.val_dataloader())
        #     for metric_name, metric_value in sampling_results.items():
        #         pass
        

            # Logging the results corresponding to the best validation_and_connectivity
            # best_metrics, best_epoch = self.compute_best_validation_metrics()
            # self.log('best_epoch', int(best_epoch), prog_bar=True, batch_size=self.batch_size)
            # for metric, value in best_metrics.items():
            #     self.log(f'best_{metric}', value, prog_bar=True, batch_size=self.batch_size)
        return avg_loss

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor(step_outputs[metric]).mean()
    

    def sample_chain(self, data, linker_sizes, keep_frames=None):
        

        if self.inpainting: #  self.inpainting: False
            template_data = data
        else:
            template_data = create_templates_for_linker_generation(data, linker_sizes)

        # print(data,'data ***********')
        # print(template_data,'template_data ***************')
        # print(linker_sizes,'linkersize ***********')


        x = template_data['positions']
        node_mask = template_data['atom_mask']
        edge_mask = template_data['edge_mask']
        h = template_data['one_hot']
        anchors = template_data['anchors']
        fragment_mask = template_data['fragment_mask']
        linker_mask = template_data['linker_mask']
        
        # Anchors and fragments labels are used as context
        if self.anchors_context:
            context = torch.cat([anchors, fragment_mask], dim=-1)
        else:
            context = fragment_mask

        # Removing COM of fragment from the atom coordinates
        if self.inpainting: #False
            center_of_mass_mask = node_mask
        elif self.center_of_mass == 'fragments':
            center_of_mass_mask = fragment_mask
        elif self.center_of_mass == 'anchors':
            center_of_mass_mask = anchors
        else:
            raise NotImplementedError(self.center_of_mass)
        

        # print(x,data['uuid'],'in lightning.py *******')
        x = utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
        # print(x,data['uuid'],'in lightning.py *******')


        # 下周要做的，检查utils.remove_partial_mean_with_mask，检查edm和egnn的sample_chain函数
        chain = self.edm.sample_chain(
            x=x,
            h=h,
            node_mask=node_mask,
            edge_mask=edge_mask,
            fragment_mask=fragment_mask,
            linker_mask=linker_mask,
            context=context,
            keep_frames=keep_frames,
        )
        return chain, node_mask

    

class Trainer(nn.Module):
    def __init__(self, n_epochs, data_path, train_data_prefix, val_data_prefix, collate_fn=collate, batch_size=64, accelerator='gpu', gpu_ids='0'):
        super(Trainer, self).__init__()
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        # GPU or CPU
        self.device = ("cuda:0" if torch.cuda.is_available() and accelerator == 'gpu' else "cpu")
        if accelerator == 'gpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"Using {self.device} device")
        
        # dataset
        self.train_dataset = ProtacDataset(
            data_path=data_path,
            prefix=train_data_prefix,
            device=self.device
            )
        self.val_dataset = ProtacDataset(
            data_path=data_path,
            prefix=val_data_prefix,
            device=self.device
            )
        
        # dataloader
        self.train_dataloader = get_dataloader(self.train_dataset, self.batch_size, collate_fn=collate_fn, shuffle=True)
        self.val_dataloader = get_dataloader(self.val_dataset, self.batch_size, collate_fn=collate_fn)
    
    def train(self, model, checkpoints_dir, ckpt_path=None):
        start_epoch = 1
        if ckpt_path:
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            start_epoch = checkpoint['epoch']
        model.to(self.device)
        
        for epoch in range(start_epoch, self.n_epochs+1):
            total_train_loss = {
                'loss': [], 
                'delta_log_px': [],
                'kl_prior': [],
                'loss_term_t': [],
                'loss_term_0': [],
                'l2_loss': [],
                'vlb_loss': [],
                'noise_t': [],
                'noise_0': []
            }
            for idx, data in enumerate(self.train_dataloader):
                model.optim.zero_grad()
                training_metrics = model.training_step(data, idx)
                for metric, value in training_metrics.items():
                    total_train_loss[metric].append(value)
                
                loss = training_metrics['loss']
                loss.backward()
                model.optim.step()
                if idx % 100 == 0:
                    print(f'EPOCH: {epoch}/{self.n_epochs} ITERS: {idx}/{len(self.train_dataloader)}')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(checkpoints_dir, f'checkpoints_EPOCH_{epoch}.ckpt'))
            
            with torch.no_grad():
                avg_train_loss = model.training_epoch_end(total_train_loss)
                total_val_loss = {
                    'loss': [], 
                    'delta_log_px': [],
                    'kl_prior': [],
                    'loss_term_t': [],
                    'loss_term_0': [],
                    'l2_loss': [],
                    'vlb_loss': [],
                    'noise_t': [],
                    'noise_0': []
                }
                for idx, data in enumerate(self.val_dataloader):
                    val_metrics = model.validation_step(data, idx)
                    for metric, value in val_metrics.items():
                        total_val_loss[metric].append(value)
                avg_val_loss = model.validation_epoch_end(total_val_loss)
                print('EPOCH: {}/{} TRAIN_LOSS: {} VAL_LOSS: {}'.format(epoch, self.n_epochs, avg_train_loss['loss'].item(), avg_val_loss['loss'].item()))

                
class Sampler(nn.Module):
    def __init__(self, data_path, test_data_prefix, accelerator='gpu', gpu_ids='0'):
        super(Sampler, self).__init__()
        # GPU or CPU
        self.device = ("cuda:0" if torch.cuda.is_available() and accelerator == 'gpu' else "cpu")
        if accelerator == 'gpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"Using {self.device} device")

        self.batch_size = 2
        # dataset
        self.test_dataset = ProtacDataset(
            data_path=data_path,
            prefix=test_data_prefix,
            device=self.device
            )
        
        # dataloader
        collate_fn = collate
        self.test_dataloader = get_dataloader(self.test_dataset, self.batch_size, collate_fn=collate_fn, shuffle=False)
        print(f'Dataloader contains {len(self.test_dataloader)} batches')
    

    def check_if_generated(self, _output_dir, _uuids, n_samples):
        generated = True
        starting_points = []
        for _uuid in _uuids:
            uuid_dir = os.path.join(_output_dir, _uuid)
            numbers = []
            for fname in os.listdir(uuid_dir):
                try:
                    num = int(fname.split('_')[0])
                    numbers.append(num)
                except:
                    continue
            if len(numbers) == 0 or max(numbers) != n_samples - 1:
                generated = False
                if len(numbers) == 0:
                    starting_points.append(0)
                else:
                    starting_points.append(max(numbers) - 1)

        if len(starting_points) > 0:
            starting = min(starting_points)
        else:
            starting = None

        return generated, starting
        
    # N:调整锚点距离
    def transform(self, data, N):
        # print(data['positions'])
        for x, anchors, fragment_mask, inE1_mask in zip(data['positions'], torch.squeeze(data['anchors'],-1), torch.squeeze(data['fragment_mask'],-1), torch.squeeze(data['infragment1_mask'],-1)):    
            coords = x
            E1_index = []
            W1_index = []
            anchors_index = []
            dev = x.device
            for i, v in enumerate(fragment_mask):
                if v.item() == 0.:continue
                if inE1_mask[i].item()==1.:E1_index.append(i)
                else:W1_index.append(i)
                if anchors[i].item()==1: anchors_index.append(i)
            
            E1 = coords[E1_index,:]
            W1 = coords[W1_index,:]
            are1 = coords[anchors_index[0],:]
            arw1 = coords[anchors_index[1],:]
            
            E1_anchor_index, W1_anchor_index = anchors_index[0], anchors_index[1]-E1.shape[0]
            # print('E1_anchor:',E1_anchor_index,'W1_anchor:',W1_anchor_index)

            cene1 = self.cal_geometric_center(E1)
            cenw1 = self.cal_geometric_center(W1)

            a = are1 - cene1 # torch.Size([3])
            b = arw1 - cenw1

            z = torch.tensor([0., 0., 1.], device=dev)
            x = torch.tensor([1., 0., 0.], device=dev)

            xa=torch.cross(a,z)
            xb=torch.cross(b,z)
            # print(a,b)
            # print(xa,xb)
            # print(torch.dot(xa,a), torch.dot(xb,b))

            thxa = self.cal_cos(a, z)
            thxb = self.cal_cos(b, z)
            thza = self.cal_cos(xa, x)
            thzb = self.cal_cos(xb, x)

            Xa = torch.tensor([[1, 0, 0],
                [0, thxa, -sin(acos(thxa))],
                [0, sin(acos(thxa)), thxa]], device=dev)

            Za = torch.tensor([[thza, -sin(acos(thza)), 0],
                [sin(acos(thza)), thza, 0],
                [0, 0, 1]], device=dev)

            Xb = torch.tensor([[1, 0, 0],
                [0, thxb, -sin(acos(thxb))],
                [0, sin(acos(thxb)), thxb]], device=dev)

            Zb = torch.tensor([[thzb, -sin(acos(thzb)), 0],
                [sin(acos(thzb)), thzb, 0],
                [0, 0, 1]], device=dev)

            if xa[1] > 0:
                Za[0][1] = sin(acos(thza))
                Za[1][0] = -sin(acos(thza))
            if xb[1] > 0:
                Zb[0][1] = sin(acos(thzb))
                Zb[1][0] = -sin(acos(thzb))

            Ma = torch.matmul(Xa, Za)
            Mb = torch.matmul(Xb, Zb)
            a = a.reshape(3, 1)
            b = b.reshape(3, 1)
            
            # print(Ma)
            # print(Mb)

            # print(torch.matmul(Ma, a), '\n', torch.matmul(Mb, b))
            Ma_inver = torch.inverse(Ma)
            rotmax = torch.matmul(Ma_inver, Mb)
            c = torch.matmul(rotmax, b)
            # print(a, '\n', b, '\n', c)
            # print(a/c)
            
            A = torch.matmul(Ma, a)
            B = torch.matmul(Mb, b)

            E = torch.transpose(torch.matmul(Ma, torch.transpose(E1,0,1)), 0, 1)
            W = torch.transpose(torch.matmul(Mb, torch.transpose(W1,0,1)), 0, 1)

            are = E[E1_anchor_index,:].reshape(1, 3)
            arw = W[W1_anchor_index,:].reshape(1, 3)
            La = torch.repeat_interleave(are, E.shape[0], 0)
            Lb = torch.repeat_interleave(arw, W.shape[0], 0)

            # print(La.shape, Lb.shape)

            E = E - La
            W = W - Lb
            are = E[E1_anchor_index,:].reshape(1, 3)
            arw = W[W1_anchor_index,:].reshape(1, 3)
            cene = self.cal_geometric_center(E)
            cenw = self.cal_geometric_center(W)
            if c[0][0]/a[0][0]>0:
                XX = torch.tensor([
                [1., 0., 0.],
                [0., -1., 0.],
                [0., 0., -1.]
                ], device=dev)
                E = torch.transpose(torch.matmul(XX, torch.transpose(E,0,1)), 0, 1)
            
            W[:,2] += N

            aref = E[E1_anchor_index,:]
            arwf = W[W1_anchor_index,:]
            cenef = self.cal_geometric_center(E)
            cenwf = self.cal_geometric_center(W)
            # print('aref: ', aref, 'arwf: ', arwf, 'cenef: ', cenef, 'cenwf: ', cenwf)
            # print('E:', E)
            # print('W:', W)
            

            #------------------------------------------------
            # !!!!!! 分子朝向
            Flag = False
            x_angle = pi/2
            y_angle = pi/4
            z_angle = -pi/2
            if Flag:
                thxc = torch.tensor(x_angle, device=dev)
                thyc = torch.tensor(y_angle, device=dev)
                
                thzc = torch.tensor(z_angle, device=dev)

                Xc = torch.tensor([[1, 0, 0],
                    [0, cos(thxc), -sin(thxc)],
                    [0, sin(thxc), cos(thxc)]], device=dev)

                Yc = torch.tensor([
                    [cos(thyc) , 0, sin(thyc)],
                    [0, 1, 0],
                    [-sin(thyc), 0, cos(thyc)]], device=dev)

                Zc = torch.tensor([[cos(thzc), -sin(thzc), 0],
                    [sin(thzc), cos(thzc), 0],
                    [0, 0, 1]], device=dev)
                
                Mc = torch.matmul(Yc, torch.matmul(Xc, Zc))
                
                arwf = W[W1_anchor_index,:].clone()
                W -= arwf
                W = torch.transpose(torch.matmul(Mc, torch.transpose(W,0,1)), 0, 1)
                W += arwf
            #------------------------------------------------
            new_x = torch.cat([E, W], 0)
            
            data['positions'][0] = new_x

        # print(data['positions'])


            
    def cal_cos(self, a, b):
        return torch.dot(a,b)/torch.norm(a)/torch.norm(b)

    def cal_geometric_center(self, positions):
        return positions.sum(dim=0)/positions.shape[0]
   

    def sample(self, model, n_samples, output_dir, user_defined_linker_size, move_N, n_steps=None):
        if user_defined_linker_size > 0:
            sizes = torch.tensor([user_defined_linker_size], device=self.device)
            user_defined_linker_size = sizes
        else:
            raise Exception("invalid linker_sizes! (linker_sizes < 0)")
        
        
        model = model.eval().to(self.device)
        model.torch_device = self.device
        if n_steps is not None:
            model.edm.T = n_steps
        N = move_N
        
        for batch_idx, data in enumerate(self.test_dataloader):
            uuids = []
            true_names = []
            frag_names = []
            pock_names = []
            for uuid in data['uuid']:
                uuid = str(uuid)
                uuids.append(uuid)
                true_names.append(f'{uuid}/true')
                frag_names.append(f'{uuid}/frag')
                pock_names.append(f'{uuid}/pock')
                os.makedirs(os.path.join(output_dir, uuid), exist_ok=True)

            
            print(uuids)
            generated, starting_point = self.check_if_generated(output_dir, uuids, n_samples)
            if generated:
                print(f'Already generated batch={batch_idx}, max_uuid={max(uuids)}')
                continue
            if starting_point > 0:
                print(f'Generating {n_samples - starting_point} for batch={batch_idx}')

            self.transform(data, N) # translation distance

            # Removing COM of fragment from the atom coordinates
            h, x, node_mask, frag_mask = data['one_hot'], data['positions'], data['atom_mask'], data['fragment_mask']
            
            if model.inpainting:
                center_of_mass_mask = node_mask
            elif model.center_of_mass == 'fragments':
                center_of_mass_mask = data['fragment_mask']
            elif model.center_of_mass == 'anchors':
                center_of_mass_mask = data['anchors']
            else:
                raise NotImplementedError(model.center_of_mass)
            x = utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
            utils.assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask)

            # Saving ground-truth molecules
            save_xyz_file(output_dir, h, x, node_mask, true_names)

            # Saving fragments
            save_xyz_file(output_dir, h, x, frag_mask, frag_names)


            # Sampling and saving generated molecules
            for i in tqdm(range(starting_point, n_samples), desc=str(batch_idx)):
                # ????sample_fn的问题
                chain, node_mask = model.sample_chain(data, linker_sizes=user_defined_linker_size, keep_frames=1)
                x = chain[0][:, :, :model.n_dims] # 坐标
                h = chain[0][:, :, model.n_dims:] # 原子类别

                pred_names = [f'{uuid}/{i}' for uuid in uuids]
                save_xyz_file(output_dir, h, x, node_mask, pred_names)

                
            
