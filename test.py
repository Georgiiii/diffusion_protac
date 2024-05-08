import argparse
import os
import pwd
import sys
import yaml

from datetime import datetime
import torch
from src.const import NUMBER_OF_ATOM_TYPES
from src.ddpm import DDPM, Sampler
from src.utils import disable_rdkit_logging, Logger

#train
# nohup python -W ignore train.py --config train_given_anchors.yml > train_diff_given_anchors.log 2>&1 &

#test
# frag1_anchor_index = [11, 1, 13, 1, 10, 1, 2, 1]
# frag2_anchor_index = [18, 5, 16, 16, 1, 35, 21, 6]

# nohup python -W ignore test.py --ckpt models/checkpoints_EPOCH_1.pt --data datasets --test_data_prefix test_protac_0 --n_samples 20 > sample.log 2>&1 &
# nohup python -W ignore test.py --ckpt models/protac_difflinker_998.ckpt --data datasets --test_data_prefix test_protac_0 --n_samples 100 > sample.log 2>&1 &



def main(args):
    start_time = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
    run_name = f'{os.path.splitext(os.path.basename(args.config))[0]}_{pwd.getpwuid(os.getuid())[0]}_{args.exp_name}_bs{args.batch_size}_{start_time}'
    experiment = run_name if args.resume is None else args.resume
   
    os.makedirs(args.logs, exist_ok=True)

    samples_dir = os.path.join(args.logs, 'samples', experiment)

    torch_device = 'cuda:0' if args.device == 'gpu' else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # !!!!! set offline, or warning appear (wandb: Network error (SSLError), entering retry loop.)
    os.environ["WANDB_API_KEY"] = 'd1eedb78926f7ee15938e8d7cf77351314beef34'
    os.environ["WANDB_MODE"] = "offline"


    number_of_atoms =  NUMBER_OF_ATOM_TYPES
    # in_node_nf: node feature中h的维度，可选择是否加入原子序数
    in_node_nf = number_of_atoms + args.include_charges # 12
    # 在h中是否加入锚点原子信息
    anchors_context = not args.remove_anchors_context # anchors_context：True
    context_node_nf = 2 if anchors_context else 1 # 2
    if '.' in args.train_data_prefix:
        context_node_nf += 1

    ddpm = DDPM(
        in_node_nf=in_node_nf,
        n_dims=3,
        context_node_nf=context_node_nf, 
        hidden_nf=args.nf,
        activation=args.activation,
        n_layers=args.n_layers,
        attention=args.attention,
        tanh=args.tanh,
        norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers,
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method,
        diffusion_steps=args.diffusion_steps,
        diffusion_noise_schedule=args.diffusion_noise_schedule,
        diffusion_noise_precision=args.diffusion_noise_precision,
        diffusion_loss_type=args.diffusion_loss_type,
        normalize_factors=args.normalize_factors,
        include_charges=args.include_charges,
        lr=args.lr,
        batch_size=args.batch_size,
        torch_device=torch_device,
        model=args.model,
        test_epochs=args.test_epochs,
        n_stability_samples=args.n_stability_samples,
        normalization=args.normalization,
        log_iterations=args.log_iterations,
        samples_dir=samples_dir,
        data_augmentation=args.data_augmentation,
        center_of_mass=args.center_of_mass,
        inpainting=args.inpainting,
        anchors_context=anchors_context,
    )

    checkpoint = torch.load(args.ckpt_path)
    ddpm.load_state_dict(checkpoint['model_state_dict'], strict=True)

    sampler = Sampler(args.data, args.test_data_prefix, accelerator=args.device, gpu_ids=args.gpu_ids, )
    print('Start sampling')
    sampler.sample(model=ddpm, n_samples=args.n_samples, output_dir=samples_dir, user_defined_linker_size=args.user_defined_linker_size, move_N=args.move_N, n_steps=args.n_steps)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='E3Diffusion')
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='train_given_anchors.yml')
    p.add_argument('--data', action='store', type=str,  default="datasets")
    p.add_argument('--test_data_prefix', action='store', type=str, default='zinc_final_test')
    p.add_argument('--checkpoints', action='store', type=str, default='models')
    p.add_argument('--logs', action='store', type=str, default='logs')
    p.add_argument('--device', action='store', type=str, default='cpu')
    p.add_argument('--gpu_ids', action='store', type=str, default='1')
    p.add_argument('--trainer_params', type=dict, help='parameters with keywords of the lightning trainer')
    p.add_argument('--log_iterations', action='store', type=str, default=20)

    p.add_argument('--exp_name', type=str, default='YourName')
    p.add_argument('--model', type=str, default='egnn_dynamics',help='our_dynamics | schnet | simple_dynamics | kernel_dynamics | egnn_dynamics |gnn_dynamics')
    p.add_argument('--probabilistic_model', type=str, default='diffusion', help='diffusion')

    # Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
    p.add_argument('--diffusion_steps', type=int, default=500)
    p.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2', help='learned, cosine')
    p.add_argument('--diffusion_noise_precision', type=float, default=1e-5, )
    p.add_argument('--diffusion_loss_type', type=str, default='l2', help='vlb, l2')

    p.add_argument('--n_epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--brute_force', type=eval, default=False,help='True | False')
    p.add_argument('--actnorm', type=eval, default=True,help='True | False')
    p.add_argument('--break_train_epoch', type=eval, default=False,help='True | False')
    p.add_argument('--dp', type=eval, default=True,help='True | False')
    p.add_argument('--condition_time', type=eval, default=True,help='True | False')
    p.add_argument('--clip_grad', type=eval, default=True,help='True | False')
    p.add_argument('--trace', type=str, default='hutch',help='hutch | exact')
    # EGNN args -->
    p.add_argument('--n_layers', type=int, default=6,   help='number of layers')
    p.add_argument('--inv_sublayers', type=int, default=1, help='number of layers')
    p.add_argument('--nf', type=int, default=128,  help='number of layers')
    p.add_argument('--tanh', type=eval, default=True, help='use tanh in the coord_mlp')
    p.add_argument('--attention', type=eval, default=True, help='use attention in the EGNN')
    p.add_argument('--norm_constant', type=float, default=1,help='diff/(|diff| + norm_constant)')
    p.add_argument('--sin_embedding', type=eval, default=False, help='whether using or not the sin embedding')
    p.add_argument('--ode_regularization', type=float, default=1e-3)
    p.add_argument('--dataset', type=str, default='qm9',  help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
    p.add_argument('--datadir', type=str, default='qm9/temp',  help='qm9 directory')
    p.add_argument('--filter_n_atoms', type=int, default=None, help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
    p.add_argument('--dequantization', type=str, default='argmax_variational',  help='uniform | variational | argmax_variational | deterministic')
    p.add_argument('--n_report_steps', type=int, default=1)
    p.add_argument('--wandb_usr', type=str)
    p.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    p.add_argument('--enable_progress_bar', action='store_true', help='Disable wandb')
    p.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
    p.add_argument('--no-cuda', action='store_true', default=False,  help='enables CUDA training')
    p.add_argument('--save_model', type=eval, default=True, help='save model')
    p.add_argument('--generate_epochs', type=int, default=1,help='save model')
    p.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
    p.add_argument('--test_epochs', type=int, default=1)
    p.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
    p.add_argument("--conditioning", nargs='+', default=[], help='arguments : homo | lumo | alpha | gap | mu | Cv')
    p.add_argument('--resume', type=str, default=None, help='')
    p.add_argument('--start_epoch', type=int, default=0, help='')
    p.add_argument('--ema_decay', type=float, default=0.999, help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')
    p.add_argument('--augment_noise', type=float, default=0)
    p.add_argument('--n_stability_samples', type=int, default=500,help='Number of samples to compute the stability')
    p.add_argument('--normalize_factors', type=eval, default=[1, 4, 1], help='normalize factors for [x, categorical, integer]')
    p.add_argument('--remove_h', action='store_true')
    p.add_argument('--include_charges', type=eval, default=True,help='include atom charge or not')
    p.add_argument('--visualize_every_batch', type=int, default=1e8,help="Can be used to visualize multiple times per epoch")
    p.add_argument('--normalization_factor', type=float, default=1,help="Normalize the sum aggregation of EGNN")
    p.add_argument('--aggregation_method', type=str, default='sum',help='"sum" or "mean"')
    p.add_argument('--normalization', type=str, default='batch_norm', help='batch_norm')
    p.add_argument('--wandb_entity', type=str, default='geometric', help='Entity (project) name')
    p.add_argument('--center_of_mass', type=str, default='fragments', help='Where to center the data: fragments | anchors')
    p.add_argument('--inpainting', action='store_true', default=False, help='Inpainting mode (full generation)')
    p.add_argument('--remove_anchors_context', action='store_true', default=False, help='Remove anchors context')

    disable_rdkit_logging()


    # Tester parameters
    p.add_argument('--ckpt_path', action='store', type=str, required=True)
    p.add_argument('--n_samples', action='store', type=int, required=True)
    p.add_argument('--n_steps', action='store', type=int, required=False, default=None)
    p.add_argument('--move_N', action='store', type=int, required=False, default=-25)
    p.add_argument('--user_defined_linker_size', action='store', type=int, required=False, default=28)



    args = p.parse_args()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list) and key != 'normalize_factors':
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}
    
    print(args)
    
    main(args=args)
