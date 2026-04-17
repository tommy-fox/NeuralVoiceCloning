# load packages
import random
import yaml
import time
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import click
import shutil
import warnings
warnings.simplefilter('ignore')
from torch.utils.tensorboard import SummaryWriter

from meldataset import build_dataloader

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

from models.losses.slmadv import SLMAdversarialLoss
from models.diffusion import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from models_util import *

from optimizers import build_optimizer

import os.path as osp
import os
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


@click.command()
@click.option('-p', '--config_path', default='configs/config_ft.yml', type=str)
def main(config_path):
    # cuDNN autotuner: speeds up conv operations by finding optimal algorithms.
    # Previously disabled to debug NaN issues, now safe to re-enable since
    # the predictor_encoder fix resolved the root cause.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # ---- Distributed training setup ----
    # Compatible with both single-GPU (python) and multi-GPU (torchrun) launch.
    distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if distributed:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        print(f"[Rank {rank}/{world_size}] Using device cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    def all_reduce_grads(model_dict, keys):
        """Average gradients across all ranks for the specified module keys."""
        for key in keys:
            for p in model_dict[key].parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data /= world_size

    def sync_skip(flag):
        """Returns True if ANY rank wants to skip this batch."""
        t = torch.tensor([1.0 if flag else 0.0], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item() > 0

    config_path = "/home/hice1/tfox35/scratch/StyleTTS2/configs/config_ft.yml"
    config = yaml.safe_load(open(config_path))

    # BEGIN CUSTOM EMBEDDINGS
    # Load external speaker embeddings if specified
    custom_embed_path = "/home/hice1/tfox35/scratch/StyleTTS2/CustomEmbeddings/generated_speaker_embeddings_epoch814_utterance.pt" 
    speaker_embedding_map = None
    if custom_embed_path:
        print("Loading external speaker embeddings...")
        loaded = torch.load(custom_embed_path)
        speaker_embedding_map = dict(loaded)
    # END CUSTOM EMBEDDINGS
    
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    # Gradient accumulation: use batch_size=1 per micro-step to fit in V100 memory,
    # accumulate over multiple micro-steps to simulate the configured batch_size.
    # With DDP, each GPU processes batch_size=1 and gradients are averaged across GPUs,
    # so accum_steps is reduced by world_size.
    effective_batch_size = config.get('batch_size', 2)
    batch_size = 1  # per micro-step batch size (fits in 16GB V100)
    accum_steps = max(1, effective_batch_size // (batch_size * world_size))
    print(f"Gradient accumulation: effective_batch_size={effective_batch_size}, "
          f"micro_batch_size={batch_size}, world_size={world_size}, accum_steps={accum_steps}")

    epochs = config.get('epochs', 200)
    save_freq = config.get('save_freq', 2)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']

    max_len = config.get('max_len', 200)
    # Cap max_len to prevent OOM on 16GB V100 with batch_size=1.
    # The decoder + discriminator memory scales with audio length.
    # 200 frames = 2 seconds of audio, fits comfortably in 16GB.
    if max_len > 200:
        print(f"Capping max_len from {max_len} to 200 to fit in V100 memory")
        max_len = 200
    
    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    
    optimizer_params = Munch(config['optimizer_params'])
    
    train_list, val_list = get_data_path_list(train_path, val_path)
    if not distributed:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    print(f"Using device {device}")

    from meldataset import FilePathDataset, Collater

    train_dataset = FilePathDataset(train_list, root_path, OOD_data=OOD_data,
                                    min_length=min_length, validation=False)
    collate_fn = Collater()
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  num_workers=2,
                                  drop_last=True,
                                  collate_fn=collate_fn,
                                  pin_memory=True)

    val_dataset = FilePathDataset(val_list, root_path, OOD_data=OOD_data,
                                  min_length=min_length, validation=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                drop_last=False,
                                collate_fn=collate_fn,
                                pin_memory=True)
    
    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    
    # load PL-BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)
    
    # build model
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]
    
    # DataParallel removed — it causes CUDA illegal memory access during
    # backward when multiple sequential DP forward calls create/destroy
    # GPU 1 replicas.  Using single-GPU with autocast (fp16) instead to
    # fit batch_size=2 in memory.
            
    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)
    
    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            print('Loading the first stage model at %s ...' % first_stage_path)
            model, _, start_epoch, iters = load_checkpoint(model, 
                None, 
                first_stage_path,
                load_only_params=True,
                ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'msd', 'mpd', 'wd', 'diffusion']) # keep starting epoch for tensorboard log

            # these epochs should be counted from the start epoch
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch
            
            # NOTE: copy.deepcopy breaks PyTorch's spectral_norm parametrizations,
            # causing predictor_encoder to use un-normalized weights during forward
            # (producing outputs of ~10^21 instead of ~0.1). Instead, create a fresh
            # StyleEncoder with proper spectral_norm and load the state dict.
            from models_util import StyleEncoder
            model.predictor_encoder = StyleEncoder(
                dim_in=model_params.dim_in,
                style_dim=model_params.style_dim,
                max_conv_dim=model_params.hidden_dim
            ).to(device)
            model.predictor_encoder.load_state_dict(model.style_encoder.state_dict())
        else:
            raise ValueError('You need to specify the path to the first stage model.') 

    # ---- Diagnostic: check all model weights for NaN/Inf after loading ----
    print("=" * 60)
    print("DIAGNOSTIC: Checking model weights for NaN/Inf...")
    for key in model:
        n_params = 0
        n_nan = 0
        n_inf = 0
        param_min = float('inf')
        param_max = float('-inf')
        for name, p in model[key].named_parameters():
            n_params += p.numel()
            nan_count = torch.isnan(p.data).sum().item()
            inf_count = torch.isinf(p.data).sum().item()
            if nan_count > 0:
                n_nan += nan_count
                print(f"  WARNING: {key}.{name} has {nan_count} NaN values!")
            if inf_count > 0:
                n_inf += inf_count
                print(f"  WARNING: {key}.{name} has {inf_count} Inf values!")
            param_min = min(param_min, p.data.min().item())
            param_max = max(param_max, p.data.max().item())
        status = "OK" if (n_nan == 0 and n_inf == 0) else "CORRUPTED"
        print(f"  {key}: {n_params} params, range=[{param_min:.4f}, {param_max:.4f}] -> {status}")
    print("=" * 60)

    # ---- Diagnostic: sanity-test style_encoder and predictor_encoder ----
    with torch.no_grad():
        _test_in = torch.randn(1, 1, 80, 200, device=device).clamp(-3, 3)
        _test_se = model.style_encoder(_test_in)
        _test_pe = model.predictor_encoder(_test_in)
        print(f"Style encoder sanity test:     output range=[{_test_se.min():.4f}, {_test_se.max():.4f}]")
        print(f"Predictor encoder sanity test: output range=[{_test_pe.min():.4f}, {_test_pe.max():.4f}]")
        if _test_pe.abs().max() > 100:
            print("  ERROR: predictor_encoder still producing extreme values!")
        else:
            print("  Both encoders producing normal values.")
    print("=" * 60)

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, 
                   model.wd, 
                   sr, 
                   model_params.slm.sr).to(device)

    # No DataParallel on loss modules or model — single GPU with fp16 autocast
    
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )
    
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    scheduler_params_dict= {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                          scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)
    
    # adjust BERT learning rate
    for g in optimizer.optimizers['bert'].param_groups:
        g['betas'] = (0.9, 0.99)
        g['lr'] = optimizer_params.bert_lr
        g['initial_lr'] = optimizer_params.bert_lr
        g['min_lr'] = 0
        g['weight_decay'] = 0.01
        
    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4
        
    # ---- Determine whether to resume from a training checkpoint or load pretrained ----
    resume_checkpoint_path = osp.join(log_dir, 'resume_checkpoint.pth')
    resume_batch = 0
    resuming = False

    # Check for resume checkpoint (mid-epoch) first, then epoch-level checkpoints
    import glob as glob_module
    epoch_ckpts = sorted(glob_module.glob(osp.join(log_dir, 'epoch_2nd_*.pth')))

    if osp.exists(resume_checkpoint_path):
        # Mid-epoch resume checkpoint takes priority
        if rank == 0:
            print(f"Found mid-epoch resume checkpoint, loading...")
        resume_state = torch.load(resume_checkpoint_path, map_location='cpu')
        for key in model:
            if key in resume_state['net']:
                model[key].load_state_dict(resume_state['net'][key], strict=False)
                if rank == 0: print(f'{key} loaded (resume)')
        optimizer.load_state_dict(resume_state['optimizer'])
        start_epoch = resume_state['epoch']
        iters = resume_state['iters']
        resume_batch = resume_state.get('batch_idx', 0)
        best_loss = resume_state.get('best_loss', float('inf'))
        resuming = True
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}, batch {resume_batch}, iters {iters}")

    elif epoch_ckpts:
        # Resume from the latest epoch-level checkpoint
        latest_ckpt = epoch_ckpts[-1]
        if rank == 0:
            print(f"Found epoch checkpoint {latest_ckpt}, loading...")
        resume_state = torch.load(latest_ckpt, map_location='cpu')
        for key in model:
            if key in resume_state['net']:
                model[key].load_state_dict(resume_state['net'][key], strict=False)
                if rank == 0: print(f'{key} loaded (resume)')
        optimizer.load_state_dict(resume_state['optimizer'])
        start_epoch = resume_state['epoch'] + 1  # start from the NEXT epoch
        iters = resume_state['iters']
        best_loss = resume_state.get('val_loss', float('inf'))
        resuming = True
        if rank == 0:
            print(f"Resuming from epoch {start_epoch} (after checkpoint epoch {resume_state['epoch']})")

    else:
        # No resume checkpoint — load pretrained and rebuild predictor_encoder
        if load_pretrained:
            model, optimizer, start_epoch, iters = load_checkpoint(model, optimizer, config['pretrained_model'],
                                        load_only_params=config.get('load_only_params', True))

        # Fix predictor_encoder: the pretrained checkpoint (and copy.deepcopy in first-stage
        # training) can break PyTorch's spectral_norm parametrizations, causing
        # predictor_encoder to output ~10^21 instead of ~0.1.
        # Rebuild with fresh spectral_norm and transfer weights via state_dict.
        from models_util import StyleEncoder
        from optimizers import define_scheduler
        _pe_state = model.predictor_encoder.state_dict()
        model.predictor_encoder = StyleEncoder(
            dim_in=model_params.dim_in,
            style_dim=model_params.style_dim,
            max_conv_dim=model_params.hidden_dim
        ).to(device)
        model.predictor_encoder.load_state_dict(_pe_state)
        # Rebuild optimizer and scheduler for the new predictor_encoder parameters
        optimizer.optimizers['predictor_encoder'] = torch.optim.AdamW(
            model.predictor_encoder.parameters(),
            lr=optimizer_params.lr, weight_decay=1e-4, betas=(0.0, 0.99), eps=1e-9)
        optimizer.schedulers['predictor_encoder'] = define_scheduler(
            optimizer.optimizers['predictor_encoder'],
            scheduler_params_dict['predictor_encoder'])
        if rank == 0:
            print("Rebuilt predictor_encoder with fresh spectral_norm parametrizations")

    n_down = model.text_aligner.n_down

    best_loss = best_loss if resuming else float('inf')
    loss_train_record = list([])
    loss_test_record = list([])
    if not resuming:
        iters = 0

    # Time-based checkpoint interval (seconds)
    CHECKPOINT_INTERVAL = 60 * 60  # save every 60 minutes
    last_checkpoint_time = time.time()

    criterion = nn.L1Loss() # F0 loss (regression)
    torch.cuda.empty_cache()
    
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    print('BERT', optimizer.optimizers['bert'])
    print('decoder', optimizer.optimizers['decoder'])

    start_ds = False
    
    running_std = []
    
    slmadv_params = Munch(config['slmadv_params'])
    slmadv = SLMAdversarialLoss(model, wl, sampler, 
                                slmadv_params.min_len, 
                                slmadv_params.max_len,
                                batch_percentage=slmadv_params.batch_percentage,
                                skip_update=slmadv_params.iter, 
                                sig=slmadv_params.sig
                               )

    for epoch in range(start_epoch, epochs):
        if rank == 0:
            print(f"epoch {epoch}")
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running_loss = 0
        start_time = time.time()

        _ = [model[key].eval() for key in model]
        
        model.text_aligner.train()
        model.text_encoder.train()
        
        model.predictor.train()
        model.bert_encoder.train()
        model.bert.train()
        model.msd.train()
        model.mpd.train()

        accum_count = 0  # gradient accumulation counter
        STYLE_CLAMP = 10.0  # clamp style vectors to prevent numerical explosion

        # When resuming mid-epoch, skip directly to the saved batch index
        # by creating a fresh dataloader subset instead of iterating through.
        start_batch = resume_batch if epoch == start_epoch else 0
        if start_batch > 0 and rank == 0:
            print(f"Skipping to batch {start_batch} (resume)")

        batch_iter = iter(train_dataloader)
        for i in range(start_batch, len(train_dataloader)):
            skip_batch = False
            try:
                batch = next(batch_iter)
            except StopIteration:
                break
            except Exception as e:
                if rank == 0:
                    print(f"DataLoader error at batch {i}, skipping: {repr(e)}")
                skip_batch = True
            if i % 10 == 0 and rank == 0: print(f"sample {i}")
            if skip_batch:
                if distributed:
                    skip_batch = sync_skip(True)
                continue
            waves = batch[0]
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels, speaker_ids, utterance_names = batch
            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                mel_mask = length_to_mask(mel_input_length).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                # BEGIN CUSTOM EMBEDDINGS
                if multispeaker and epoch >= diff_epoch:
                    if speaker_embedding_map is not None:
                        # ref_ss = []
                        # for b in range(len(speaker_ids)):
                        #     embedding_key = f"{speaker_ids[b]}_{utterance_names[b]}"
                        #     if embedding_key not in speaker_embedding_map:
                        #         raise KeyError(f"Missing speaker embedding for key: {embedding_key}")

                        #     emb = speaker_embedding_map[embedding_key].to(device)  # your custom embedding
                        #     ref_ss.append(emb)
                        # ref_ss = torch.cat(ref_ss, dim=0)

                        ##### BEGIN FILTERING...
                        ###
                        # Filter for valid speaker embeddings
                        valid_indices = []
                        ref_ss = []

                        for b in range(len(speaker_ids)):
                            embedding_key = f"{speaker_ids[b]}_{utterance_names[b]}"
                            if embedding_key in speaker_embedding_map:
                                ref_ss.append(speaker_embedding_map[embedding_key].to(device))
                                valid_indices.append(b)
                            # else:
                            #     print(f"Skipping missing embedding: {embedding_key}")

                        if len(valid_indices) == 0:
                            skip_batch = True

                        if not skip_batch:
                            # Apply filtering to all batch components
                            waves = [waves[i] for i in valid_indices]
                            texts = texts[valid_indices]
                            input_lengths = input_lengths[valid_indices]
                            ref_texts = ref_texts[valid_indices]
                            ref_lengths = ref_lengths[valid_indices]
                            mels = mels[valid_indices]
                            mel_input_length = mel_input_length[valid_indices]
                            ref_mels = ref_mels[valid_indices]
                            speaker_ids = [speaker_ids[i] for i in valid_indices]
                            utterance_names = [utterance_names[i] for i in valid_indices]

                            ref_ss = torch.cat(ref_ss, dim=0)
                            ##### END FILTERING

                            # Recompute masks after filtering to match new batch size
                            # Pass max_len from the actual tensor dims so masks match padding
                            mask = length_to_mask(mel_input_length // (2 ** n_down), max_len=mels.shape[-1] // (2 ** n_down)).to(device)
                            mel_mask = length_to_mask(mel_input_length, max_len=mels.shape[-1]).to(device)
                            text_mask = length_to_mask(input_lengths, max_len=texts.shape[1]).to(texts.device)

                            # compute prosodic style
                            ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))

                            # concatenate acoustic style and prosodic style
                            ref = torch.cat([ref_ss, ref_sp], dim=1)

                    else:
                        ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                        ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                        ref = torch.cat([ref_ss, ref_sp], dim=1)
                    # END CUSTOM EMBEDDINGS

            if not skip_batch:
                try:
                    ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)
                except:
                    skip_batch = True

            # SYNC POINT 1: all ranks agree before proceeding to forward pass
            if distributed:
                skip_batch = sync_skip(skip_batch)
            if skip_batch:
                continue

            # DIAGNOSTIC: check for NaN after text_aligner (first 3 batches only)
            if i < 3 and rank == 0:
                print(f"  [batch {i}] mels: min={mels.min():.4f} max={mels.max():.4f} nan={torch.isnan(mels).any()}")
                print(f"  [batch {i}] s2s_attn: min={s2s_attn.min():.4f} max={s2s_attn.max():.4f} nan={torch.isnan(s2s_attn).any()}")
                print(f"  [batch {i}] ppgs: min={ppgs.min():.4f} max={ppgs.max():.4f} nan={torch.isnan(ppgs).any()}")

            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)

            # 50% of chance of using monotonic version
            if bool(random.getrandbits(1)):
                asr = (t_en @ s2s_attn)
            else:
                asr = (t_en @ s2s_attn_mono)

            d_gt = s2s_attn_mono.sum(axis=-1).detach()

            # compute the style of the entire utterance
            # this operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
            ss = []
            gs = []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item())
                mel = mels[bib, :, :mel_input_length[bib]]
                s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)
                s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                gs.append(s)

            s_dur = torch.stack(ss).squeeze(1)  # global prosodic styles
            gs = torch.stack(gs).squeeze(1) # global acoustic styles

            # Clamp per-utterance style vectors too (used for diffusion/denoiser)
            s_dur = s_dur.clamp(-STYLE_CLAMP, STYLE_CLAMP)
            gs = gs.clamp(-STYLE_CLAMP, STYLE_CLAMP)

            s_trg = torch.cat([gs, s_dur], dim=-1).detach() # ground truth for denoiser

            if i == 0 and rank == 0: print("bert")
            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

            if i == 0 and rank == 0: print("denoiser")
            # denoiser training
            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)

                if model_params.diffusion.dist.estimate_sigma_data:
                    model.diffusion.diffusion.sigma_data = s_trg.std(axis=-1).mean().item() # batch-wise std estimation
                    running_std.append(model.diffusion.diffusion.sigma_data)

                if multispeaker:
                    s_preds = sampler(noise = torch.randn_like(s_trg).unsqueeze(1).to(device), 
                          embedding=bert_dur,
                          embedding_scale=1,
                                   features=ref, # reference from the same speaker as the embedding
                             embedding_mask_proba=0.1,
                             num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=ref).mean() # EDM loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach()) # style reconstruction loss
                else:
                    s_preds = sampler(noise = torch.randn_like(s_trg).unsqueeze(1).to(device), 
                          embedding=bert_dur,
                          embedding_scale=1,
                             embedding_mask_proba=0.1,
                             num_steps=num_steps).squeeze(1)                    
                    loss_diff = model.diffusion.diffusion(s_trg.unsqueeze(1), embedding=bert_dur).mean() # EDM loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach()) # style reconstruction loss
            else:
                loss_sty = 0
                loss_diff = 0


            s_loss = 0

            if i == 0 and rank == 0: print("predictor")
            d, p = model.predictor(d_en, s_dur, 
                                                    input_lengths, 
                                                    s2s_attn_mono, 
                                                    text_mask)

            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            en = []
            gt = []
            p_en = []
            wav = []
            st = []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start+mel_len])
                p_en.append(p[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

                y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))

                # style reference (better to be different from the GT)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])

            wav = torch.stack(wav).float().detach()

            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()


            if not skip_batch and gt.size(-1) < 80:
                skip_batch = True

            # SYNC POINT 2: all ranks agree before style encoder + backward passes
            if distributed:
                skip_batch = sync_skip(skip_batch)
            if skip_batch:
                continue

            if i == 0 and rank == 0: print("style encoder")
            s = model.style_encoder(gt.unsqueeze(1))
            s_dur = model.predictor_encoder(gt.unsqueeze(1))

            # DIAGNOSTIC: print style vectors immediately after encoder (first 3 batches)
            if i < 3 and rank == 0:
                print(f"  [batch {i}] gt: shape={list(gt.shape)} min={gt.min():.4f} max={gt.max():.4f}")
                print(f"  [batch {i}] s (raw style_encoder): min={s.min():.4f} max={s.max():.4f} nan={torch.isnan(s).any()}")
                print(f"  [batch {i}] s_dur (raw predictor_encoder): min={s_dur.min():.4f} max={s_dur.max():.4f} nan={torch.isnan(s_dur).any()}")

            # Clamp style vectors to prevent numerical explosion in decoder AdaIN layers.
            # The decoder uses AdaIN: (1 + gamma) * norm(x) + beta, where gamma/beta = Linear(s).
            # Extreme s values (>10^20 observed) cause OOM and NaN in all downstream computation.
            if s.abs().max() > STYLE_CLAMP:
                if i < 10 and rank == 0:
                    print(f"  [batch {i}] WARNING: clamping s from [{s.min():.2e}, {s.max():.2e}] to [-{STYLE_CLAMP}, {STYLE_CLAMP}]")
                s = s.clamp(-STYLE_CLAMP, STYLE_CLAMP)
            if s_dur.abs().max() > STYLE_CLAMP:
                if i < 10 and rank == 0:
                    print(f"  [batch {i}] WARNING: clamping s_dur from [{s_dur.min():.2e}, {s_dur.max():.2e}] to [-{STYLE_CLAMP}, {STYLE_CLAMP}]")
                s_dur = s_dur.clamp(-STYLE_CLAMP, STYLE_CLAMP)

            with torch.no_grad():
                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze(-1)

                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)

                y_rec_gt = wav.unsqueeze(1)
                y_rec_gt_pred = model.decoder(en, F0_real, N_real, s)

                wav = y_rec_gt

            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)

            y_rec = model.decoder(en, F0_fake, N_fake, s)

            # DIAGNOSTIC: check decoder output and key tensors (first 3 batches)
            if i < 3 and rank == 0:
                print(f"  [batch {i}] en: min={en.min():.4f} max={en.max():.4f} nan={torch.isnan(en).any()}")
                print(f"  [batch {i}] F0_fake: min={F0_fake.min():.4f} max={F0_fake.max():.4f} nan={torch.isnan(F0_fake).any()}")
                print(f"  [batch {i}] N_fake: min={N_fake.min():.4f} max={N_fake.max():.4f} nan={torch.isnan(N_fake).any()}")
                print(f"  [batch {i}] s (style): min={s.min():.4f} max={s.max():.4f} nan={torch.isnan(s).any()}")
                print(f"  [batch {i}] y_rec: min={y_rec.min():.4f} max={y_rec.max():.4f} nan={torch.isnan(y_rec).any()}")
                print(f"  [batch {i}] y_rec_gt_pred: min={y_rec_gt_pred.min():.4f} max={y_rec_gt_pred.max():.4f} nan={torch.isnan(y_rec_gt_pred).any()}")
                print(f"  [batch {i}] wav: min={wav.min():.4f} max={wav.max():.4f} nan={torch.isnan(wav).any()}")
                print(f"  [batch {i}] F0_real: min={F0_real.min():.4f} max={F0_real.max():.4f} nan={torch.isnan(F0_real).any()}")

            loss_F0_rec =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            # ============================================================
            # Gradient accumulation: disc steps every micro-batch,
            # gen grads accumulate over accum_steps micro-batches.
            # ============================================================

            # --- Discriminator: step every micro-batch ---
            optimizer.zero_grad('msd')
            optimizer.zero_grad('mpd')
            d_loss = dl(wav.detach(), y_rec.detach()).mean()
            d_loss.backward()
            if distributed:
                all_reduce_grads(model, ['msd', 'mpd'])
            optimizer.step('msd')
            optimizer.step('mpd')

            if i == 0 and rank == 0: print("loss")
            # --- Generator: accumulate grads, step every accum_steps ---
            # On the first micro-batch of each window, zero gen grads
            if accum_count == 0:
                gen_keys = ['bert_encoder', 'bert', 'predictor', 'predictor_encoder',
                            'style_encoder', 'decoder', 'text_encoder', 'text_aligner']
                for key in gen_keys:
                    optimizer.zero_grad(key)
                if epoch >= diff_epoch:
                    optimizer.zero_grad('diffusion')

            loss_mel = stft_loss(y_rec, wav)
            loss_gen_all = gl(wav, y_rec).mean()
            loss_lm = wl(wav.detach().squeeze(1), y_rec.squeeze(1)).mean()

            loss_ce = 0
            loss_dur = 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p in range(_s2s_trg.shape[0]):
                    _s2s_trg[p, :_text_input[p]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(_dur_pred[1:_text_length-1],
                                       _text_input[1:_text_length-1])
                loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)

            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= texts.size(0)

            loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

            g_loss = loss_params.lambda_mel * loss_mel + \
                     loss_params.lambda_F0 * loss_F0_rec + \
                     loss_params.lambda_ce * loss_ce + \
                     loss_params.lambda_norm * loss_norm_rec + \
                     loss_params.lambda_dur * loss_dur + \
                     loss_params.lambda_gen * loss_gen_all + \
                     loss_params.lambda_slm * loss_lm + \
                     loss_params.lambda_sty * loss_sty + \
                     loss_params.lambda_diff * loss_diff + \
                    loss_params.lambda_mono * loss_mono + \
                    loss_params.lambda_s2s * loss_s2s

            # DIAGNOSTIC: print all loss components for first 3 batches
            if i < 3 and rank == 0:
                print(f"  [batch {i}] LOSSES: mel={loss_mel.item():.6f} F0={loss_F0_rec.item():.6f} "
                      f"ce={loss_ce if isinstance(loss_ce, (int,float)) else loss_ce.item():.6f} "
                      f"norm={loss_norm_rec.item():.6f} dur={loss_dur if isinstance(loss_dur, (int,float)) else loss_dur.item():.6f} "
                      f"gen={loss_gen_all.item():.6f} lm={loss_lm.item():.6f} "
                      f"sty={loss_sty if isinstance(loss_sty, (int,float)) else loss_sty.item():.6f} "
                      f"diff={loss_diff if isinstance(loss_diff, (int,float)) else loss_diff.item():.6f} "
                      f"mono={loss_mono.item():.6f} s2s={loss_s2s if isinstance(loss_s2s, (int,float)) else loss_s2s.item():.6f}")
                print(f"  [batch {i}] g_loss={g_loss.item():.6f} d_loss={d_loss.item():.6f}")

            running_loss += loss_mel.item()

            # NaN guard: skip backward if g_loss is NaN to prevent CUDA crash
            # SYNC POINT 3: all ranks must agree on NaN skip (before gen all_reduce)
            nan_skip = torch.isnan(g_loss) or torch.isinf(g_loss)
            if distributed:
                nan_skip = sync_skip(nan_skip)
            if nan_skip:
                if rank == 0:
                    print(f"  [batch {i}] WARNING: g_loss is NaN/Inf, skipping backward!")
                accum_count = 0
                for key in gen_keys:
                    optimizer.zero_grad(key)
                if epoch >= diff_epoch:
                    optimizer.zero_grad('diffusion')
                continue

            # Scale gen loss for gradient accumulation (average over micro-batches)
            (g_loss / accum_steps).backward()

            accum_count += 1

            # Step generator when we've accumulated enough micro-batches
            if accum_count >= accum_steps:
                if distributed:
                    all_reduce_grads(model, gen_keys)
                    if epoch >= diff_epoch:
                        all_reduce_grads(model, ['diffusion'])

                optimizer.step('bert_encoder')
                optimizer.step('bert')
                optimizer.step('predictor')
                optimizer.step('predictor_encoder')
                optimizer.step('style_encoder')
                optimizer.step('decoder')

                optimizer.step('text_encoder')
                optimizer.step('text_aligner')

                if epoch >= diff_epoch:
                    optimizer.step('diffusion')

                accum_count = 0  # reset for next accumulation window

            d_loss_slm, loss_gen_lm = 0, 0
            if epoch >= joint_epoch:
                if i == 0 and rank == 0: print("slm")
                # randomly pick whether to use in-distribution text
                # Use seeded random so all ranks make the same choice
                use_ind = bool(random.getrandbits(1))

                if use_ind:
                    ref_lengths = input_lengths
                    ref_texts = texts

                slm_out = slmadv(i,
                                 y_rec_gt,
                                 y_rec_gt_pred,
                                 waves,
                                 mel_input_length,
                                 ref_texts,
                                 ref_lengths, use_ind, s_trg.detach(), ref if multispeaker else None)

                if slm_out is not None:
                    d_loss_slm, loss_gen_lm, y_pred = slm_out

                    # SLM generator loss
                    optimizer.zero_grad()
                    loss_gen_lm.backward()

                    if distributed:
                        all_reduce_grads(model, ['bert_encoder', 'bert', 'predictor', 'diffusion'])

                    # compute the gradient norm
                    total_norm = {}
                    for key in model.keys():
                        total_norm[key] = 0
                        parameters = [p for p in model[key].parameters() if p.grad is not None and p.requires_grad]
                        for p in parameters:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm[key] += param_norm.item() ** 2
                        total_norm[key] = total_norm[key] ** 0.5

                    # gradient scaling
                    if total_norm['predictor'] > slmadv_params.thresh:
                        for key in model.keys():
                            for p in model[key].parameters():
                                if p.grad is not None:
                                    p.grad *= (1 / total_norm['predictor'])

                    for p in model.predictor.duration_proj.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.predictor.lstm.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.diffusion.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    optimizer.step('bert_encoder')
                    optimizer.step('bert')
                    optimizer.step('predictor')
                    optimizer.step('diffusion')

                    # SLM discriminator loss
                    if d_loss_slm != 0:
                        optimizer.zero_grad()
                        d_loss_slm.backward()
                        if distributed:
                            all_reduce_grads(model, ['wd'])
                        optimizer.step('wd')

            iters = iters + 1

            if (i+1)%log_interval == 0 and rank == 0:
                logger.info ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f, DiscLM Loss: %.5f, GenLM Loss: %.5f, SLoss: %.5f, S2S Loss: %.5f, Mono Loss: %.5f'
                    %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, d_loss, loss_dur, loss_ce, loss_norm_rec, loss_F0_rec, loss_lm, loss_gen_all, loss_sty, loss_diff, d_loss_slm, loss_gen_lm, s_loss, loss_s2s, loss_mono))

                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/ce_loss', loss_ce, iters)
                writer.add_scalar('train/dur_loss', loss_dur, iters)
                writer.add_scalar('train/slm_loss', loss_lm, iters)
                writer.add_scalar('train/norm_loss', loss_norm_rec, iters)
                writer.add_scalar('train/F0_loss', loss_F0_rec, iters)
                writer.add_scalar('train/sty_loss', loss_sty, iters)
                writer.add_scalar('train/diff_loss', loss_diff, iters)
                writer.add_scalar('train/d_loss_slm', d_loss_slm, iters)
                writer.add_scalar('train/gen_loss_slm', loss_gen_lm, iters)

                running_loss = 0

                if rank == 0:
                    print('Time elasped:', time.time()-start_time)

            # ---- Time-based mid-epoch checkpoint (rank 0 only) ----
            if rank == 0 and (time.time() - last_checkpoint_time) >= CHECKPOINT_INTERVAL:
                print(f"  Saving mid-epoch checkpoint at epoch {epoch}, batch {i}...")
                resume_state = {
                    'net': {key: model[key].state_dict() for key in model},
                    'optimizer': optimizer.state_dict(),
                    'iters': iters,
                    'epoch': epoch,
                    'batch_idx': i + 1,
                    'best_loss': best_loss,
                }
                torch.save(resume_state, resume_checkpoint_path)
                last_checkpoint_time = time.time()
                print(f"  Mid-epoch checkpoint saved.")

        # Clear resume_batch after first epoch so subsequent epochs start from batch 0
        resume_batch = 0

        loss_test = 0
        loss_align = 0
        loss_f = 0
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            val_iter = iter(val_dataloader)
            for batch_idx in range(len(val_dataloader)):
                try:
                    batch = next(val_iter)
                except Exception as e:
                    if rank == 0:
                        print(f"Val dataloader error at batch {batch_idx}, skipping: {repr(e)}")
                    continue

                optimizer.zero_grad()

                try:
                    waves = batch[0]
                    batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch[1:]]
                    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels, speaker_ids, utterance_names = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                        text_mask = length_to_mask(input_lengths).to(texts.device)

                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = (t_en @ s2s_attn_mono)

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    ss = []
                    gs = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item())
                        mel = mels[bib, :, :mel_input_length[bib]]
                        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                        gs.append(s)

                    s = torch.stack(ss).squeeze(1)
                    gs = torch.stack(gs).squeeze(1)
                    s_trg = torch.cat([s, gs], dim=-1).detach()

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
                    d, p = model.predictor(d_en, s, 
                                                        input_lengths, 
                                                        s2s_attn_mono, 
                                                        text_mask)
                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    gt = []

                    p_en = []
                    wav = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start:random_start+mel_len])
                        p_en.append(p[bib, :, random_start:random_start+mel_len])

                        gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                        y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()

                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()
                    s = model.predictor_encoder(gt.unsqueeze(1))

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, :_text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                               _text_input[1:_text_length-1])

                    loss_dur /= texts.size(0)

                    s = model.style_encoder(gt.unsqueeze(1))

                    y_rec = model.decoder(en, F0_fake, N_fake, s)
                    loss_mel = stft_loss(y_rec.squeeze(1), wav.detach())

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1)) 

                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_F0).mean()

                    iters_test += 1
                except:
                    continue

        if rank == 0:
            print('Epochs:', epoch + 1)
        if iters_test == 0:
            iters_test = 1
        if rank == 0:
            logger.info('Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f' % (loss_test / iters_test, loss_align / iters_test, loss_f / iters_test) + '\n\n\n')
            print('\n\n\n')
            writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
            writer.add_scalar('eval/dur_loss', loss_test / iters_test, epoch + 1)
            writer.add_scalar('eval/F0_loss', loss_f / iters_test, epoch + 1)


        if (epoch + 1) % save_freq == 0 and rank == 0:
            if iters_test > 0 and (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            print('Saving..')
            state = {
                'net':  {key: model[key].state_dict() for key in model},
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            torch.save(state, save_path)
            # Delete previous epoch checkpoint(s) to save disk space (keep only the latest)
            # NEVER delete the original pretrained checkpoint used to start fine-tuning
            import glob
            pretrained_path = osp.realpath(config.get('pretrained_model', ''))
            for old_ckpt in sorted(glob.glob(osp.join(log_dir, 'epoch_2nd_*.pth'))):
                if old_ckpt != save_path and osp.realpath(old_ckpt) != pretrained_path:
                    os.remove(old_ckpt)
                    print(f"Deleted old checkpoint: {old_ckpt}")
            # Clean up mid-epoch resume checkpoint since we have a full epoch save
            if osp.exists(resume_checkpoint_path):
                os.remove(resume_checkpoint_path)
                print("Removed mid-epoch resume checkpoint (full epoch save done)")

            # if estimate sigma, save the estimated simga
            if model_params.diffusion.dist.estimate_sigma_data:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))

                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)

        # Synchronize all ranks at end of epoch before starting next
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__=="__main__":
    main()