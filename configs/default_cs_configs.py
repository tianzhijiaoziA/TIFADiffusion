import ml_collections
import torch

def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 1#64
  training.epochs = 1000
  training.n_iters = 2400001
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.n_jitted_steps = 1
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.206
  sampling.projection_sigma_rate = 1.586
  sampling.cs_solver = 'projection'
  sampling.expansion = 4
  sampling.coeff = 1.
  sampling.n_projections = 23
  sampling.task = 'ct'
  sampling.lambd = 0.5
  sampling.denoise_override = True

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.ckpt_id = 101
  evaluate.batch_size = 1#128
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'
  evaluate.begin_ckpt = 10
  evaluate.end_ckpt = 30

  # data
  config.data = data = ml_collections.ConfigDict()
  data.random_flip = True
  data.uniform_dequantization = False
  data.centered = False
  data.num_channels = 1

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 378.
  model.sigma_min = 0.01
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return config