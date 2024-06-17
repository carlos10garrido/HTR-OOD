# Logger class to wrap-up logging functionality
import torchmetrics
import torch
import torchvision
import wandb

class MetricLogger():
  def __init__(
    self, 
    logger=None,
    train_datasets=[],
    val_datasets=[],
    test_datasets=[]):

    # self.pl_logger = pl_logger
    print(f'Initializing MetricLogger...')
    print(f'Logger: {logger}')
    self.logger = logger
    self.current_epoch = 0
    self.train_datasets = train_datasets
    self.val_datasets = val_datasets
    self.test_datasets = test_datasets

    self.train_losses = []
    self.train_accs = []
    
    # Set validation metris per dataset as a dict of lists
    self.val_losses = {val_dataset: [] for val_dataset in self.val_datasets}
    self.val_accs = {val_dataset: [] for val_dataset in self.val_datasets}
    self.val_cers = {val_dataset: torchmetrics.CharErrorRate() for val_dataset in self.val_datasets}
    self.val_wers = {val_dataset: torchmetrics.WordErrorRate() for val_dataset in self.val_datasets}
    self.best_val_cers = {val_dataset: 1e5 for val_dataset in self.val_datasets}
    self.best_val_wers = {val_dataset: 1e5 for val_dataset in self.val_datasets}
    self.mean_best_val_cer = 1e5
    self.mean_best_val_wer = 1e5
    self.best_in_domain_val_cer = 1e5
    self.best_out_domain_val_cer = 1e5
    self.best_in_domain_val_wer = 1e5
    self.best_out_domain_val_wer = 1e5

    # Set test metrics per dataset as a dict()
    self.cer_test = {test_dataset: torchmetrics.CharErrorRate() for test_dataset in self.test_datasets}
    self.wer_test = {test_dataset: torchmetrics.WordErrorRate() for test_dataset in self.test_datasets}

    print(f'Train datasets: {self.train_datasets}')
    print(f'Validation datasets: {self.val_datasets}')
    print(f'Test datasets: {self.test_datasets}')
    print(f'Validation losses: {self.val_losses}')
    print(f'Validation accs: {self.val_accs}')
    print(f'Validation cers: {self.val_cers}')
    

  def update_epoch(self, epoch):
    self.current_epoch = epoch

  def log_learning_rate(self, lr, epoch):
    self.logger.experiment.log({f'training/lr': lr, 'epoch': epoch})

  def log_images(self, images, dataset):
    img_grid = torchvision.utils.make_grid(images, normalize=False)
    # To PIL image
    img_grid = torchvision.transforms.ToPILImage()(img_grid)
    self.logger.experiment.log({f'{dataset}': [wandb.Image(img_grid, mode='RGB')]})

    return img_grid

  def log_train_step(self, train_loss, train_acc):
    self.train_losses.append(train_loss)
    self.train_accs.append(train_acc)

  def log_val_step_cer(self, output, label, val_dataset):
    self.val_cers[val_dataset].update(output, label)

  def log_val_step_wer(self, output, label, val_dataset):
    self.val_wers[val_dataset].update(output, label)

  def log_val_step(self, val_loss, val_acc, val_dataset):
    self.val_losses[val_dataset].append(val_loss)
    self.val_accs[val_dataset].append(val_acc)

  def log_test_step_cer(self, output, label, test_dataset):
    self.cer_test[test_dataset].update(output, label)

  def log_test_step_wer(self, output, label, test_dataset):
    self.wer_test[test_dataset].update(output, label)

  def log_train_metrics(self):
    train_loss = torch.stack(self.train_losses).mean()
    train_acc = torch.stack(self.train_accs).mean()

    print(f'TRAIN ACC = {train_acc}')
    self.logger.experiment.log({f'train_loss_epoch': train_loss, 'epoch': self.current_epoch})
    self.logger.experiment.log({f'train_acc_epoch': train_loss, 'epoch': self.current_epoch})
    self.train_losses = []
    self.train_accs = []

  def compute_heldout_domain_cer(self, val_cers):
    # Calculate heldout-domain CERs
    heldout_domain_cers = {}
    for target_dataset in set(self.val_datasets) - set(self.train_datasets):
      heldout_domain_cer = 0.0
      heldout_domain_datasets = set(self.val_datasets) - set([target_dataset]) - set(self.train_datasets)
      if len(heldout_domain_datasets) > 0:
        for dataset in heldout_domain_datasets:
          heldout_domain_cer += val_cers[dataset] / len(heldout_domain_datasets)
        
      heldout_domain_cers[target_dataset] = heldout_domain_cer
        
      print(f'HELDOUT-DOMAIN CER excluding {target_dataset} as target and computing for heldout_domain_datasets {heldout_domain_datasets} = {heldout_domain_cer}. Train datasets = {self.train_datasets}')
      self.logger.experiment.log({f'val/heldout_target_{target_dataset}': heldout_domain_cer, 'epoch': self.current_epoch})

    # Also add the heldout-domain CER in the ID dataset
    

    return heldout_domain_cers
      

  def log_val_metrics(self):
    """Log validation metrics and return mean_val_cer, in_domain_cer, out_of_domain_cer, heldout_domain_cer."""

    val_loss, val_acc, val_cer, val_wer = {}, {}, {}, {}

    # Compute losses, accuracies and CERs per dataset
    for dataset in self.val_datasets:
      # print(f'Computing val_loss, val_acc and val_cer for {dataset}')
      # print(f'len(self.val_losses[dataset]) = {len(self.val_losses[dataset])}')
      # print(f'len(self.val_accs[dataset]) = {len(self.val_accs[dataset])}')
      # Loss
      if len(self.val_losses[dataset]) != 0:
        val_loss[dataset] = torch.stack(self.val_losses[dataset]).mean()
        self.logger.experiment.log({f'val_loss_' + dataset + '_epoch': val_loss[dataset], 'epoch': self.current_epoch})

      # Accuracy
      if len(self.val_accs[dataset]) != 0:
        val_acc[dataset] = torch.stack(self.val_accs[dataset]).mean()
        self.logger.experiment.log({f'val_acc_' + dataset + '_epoch': val_acc[dataset], 'epoch': self.current_epoch})
        # self.log('val_acc_' + dataset + '_epoch', val_acc[dataset], on_epoch=True, prog_bar=True, logger=True)

      # CER
      val_cer[dataset] = self.val_cers[dataset].compute()
      # Check if CER is NaN
      if torch.isnan(val_cer[dataset]):
        val_cer[dataset] = 1e5
      self.logger.experiment.log({f'val_cer_' + dataset + '_epoch': val_cer[dataset], 'epoch': self.current_epoch})

      # WER
      val_wer[dataset] = self.val_wers[dataset].compute()
      # Check if WER is NaN
      if torch.isnan(val_wer[dataset]):
        val_wer[dataset] = 1e5
      self.logger.experiment.log({f'val_wer_' + dataset + '_epoch': val_wer[dataset], 'epoch': self.current_epoch})


    # Compute mean of means for CER
    total_val_cer, count_nonzero = 0.0, 0
    for dataset in self.val_datasets:
      print(f'val_cer[{dataset}] = {val_cer[dataset]}')
      total_val_cer += val_cer[dataset]
      count_nonzero += 1

      if self.best_val_cers[dataset] > val_cer[dataset]:
        self.best_val_cers[dataset] = val_cer[dataset]
        print(f'IMPROVED best val_cer[{dataset}]: {self.best_val_cers[dataset]}')
        self.logger.experiment.log({f'best_val_cer_' + dataset: self.best_val_cers[dataset], 'epoch': self.current_epoch})

    # Compute heldout-domain CERs
    heldout_domain_cers = self.compute_heldout_domain_cer(val_cer)
    

    total_val_cer /= count_nonzero
    mean_val_cer = total_val_cer
    print(f'TOTAL VAL CER = {total_val_cer}')

    # Compute mean of means for WER
    total_val_wer, count_nonzero = 0.0, 0
    for dataset in self.val_datasets:
      print(f'val_wer[{dataset}] = {val_wer[dataset]}')
      total_val_wer += val_wer[dataset]
      count_nonzero += 1

      if self.best_val_wers[dataset] > val_wer[dataset]:
        self.best_val_wers[dataset] = val_wer[dataset]
        print(f'IMPROVED best val_wer[{dataset}]: {self.best_val_wers[dataset]}')
        self.logger.experiment.log({f'best_val_wer_' + dataset: self.best_val_wers[dataset], 'epoch': self.current_epoch})

    total_val_wer /= count_nonzero
    mean_val_wer = total_val_wer 
    print(f'TOTAL VAL WER = {mean_val_wer}')
    
    # Save best val_cer mean of means for CER
    if total_val_cer < self.mean_best_val_cer:
      self.mean_best_val_cer = total_val_cer
      print(f'IMPROVED mean best val_cer (across datasets): {self.mean_best_val_cer}')
      self.logger.experiment.log({f'mean_best_val_cer': self.mean_best_val_cer, 'epoch': self.current_epoch})

    # Save best val_wer mean of means for WER
    if total_val_wer < self.mean_best_val_wer:
      self.mean_best_val_wer = total_val_wer
      print(f'IMPROVED mean best val_wer (across datasets): {self.mean_best_val_wer}')
      self.logger.experiment.log({f'mean_best_val_wer': self.mean_best_val_wer, 'epoch': self.current_epoch})


    # Calculate in-domain, out-of-domain, heldout-domain CERs and WERs
    in_domain_cer, out_of_domain_cer, heldout_domain_cer = 0.0, 0.0, 0.0
    in_domain_wer, out_of_domain_wer, heldout_domain_wer = 0.0, 0.0, 0.0


    in_domain_datasets = set(self.train_datasets)
    out_of_domain_datasets = set(self.val_datasets) - set(self.train_datasets)
    # Heldout = Test (always one) - Train (always one). Val that are not in train and also not in test
    # heldout_domain_datasets = set(self.val_datasets) - set(self.train_datasets) - set(self.test_datasets)

    print(f'IN-DOMAIN datasets = {in_domain_datasets}')
    print(f'OUT-OF-DOMAIN datasets = {out_of_domain_datasets}')
    # print(f'HELDOUT-DOMAIN datasets = {heldout_domain_datasets}')

    
    for dataset in in_domain_datasets:
      if dataset in self.val_datasets:
        in_domain_cer += val_cer[dataset] / len(in_domain_datasets)
        in_domain_wer += val_wer[dataset] / len(in_domain_datasets)

    if len(out_of_domain_datasets) > 0:
      for dataset in out_of_domain_datasets:
        if dataset in self.val_datasets:
          out_of_domain_cer += val_cer[dataset] / len(out_of_domain_datasets)
          out_of_domain_wer += val_wer[dataset] / len(out_of_domain_datasets)


    # Heldout-domains CER
    # if len(heldout_domain_datasets) > 0:
    #   for dataset in heldout_domain_datasets:
    #     if dataset in self.val_datasets:
    #       heldout_domain_cer += val_cer[dataset] / len(heldout_domain_datasets)
    #       heldout_domain_wer += val_wer[dataset] / len(heldout_domain_datasets)
    

    print(f'IN-DOMAIN CER = {in_domain_cer}')
    print(f'OUT-OF-DOMAIN CER = {out_of_domain_cer}')
    print(f'IN-DOMAIN WER = {in_domain_wer}')
    print(f'OUT-OF-DOMAIN WER = {out_of_domain_wer}')
    # print(f'HELDOUT-DOMAIN CER = {heldout_domain_cer}')
    # print(f'HELDOUT-DOMAIN WER = {heldout_domain_wer}')

    # self.log('in_domain_cer_epoch', in_domain_cer, on_epoch=True, prog_bar=True, logger=True)
    self.logger.experiment.log({f'val/in_domain_val_cer_epoch': in_domain_cer, 'epoch': self.current_epoch})
    self.logger.experiment.log({f'val/in_domain_val_wer_epoch': in_domain_wer, 'epoch': self.current_epoch})
    self.logger.experiment.log({f'val/out_of_domain_val_cer_epoch': out_of_domain_cer, 'epoch': self.current_epoch})
    self.logger.experiment.log({f'val/out_of_domain_val_wer_epoch': out_of_domain_wer, 'epoch': self.current_epoch})

    # self.logger.experiment.log({f'val/heldout_domain_val_cer_epoch': heldout_domain_cer, 'epoch': self.current_epoch})
    # self.logger.experiment.log({f'val/heldout_domain_val_wer_epoch': heldout_domain_wer, 'epoch': self.current_epoch})

    # Check if the dataset is minusc or not
    # convert set to list and check if minusc is in the list
    if "minusc" not in list(in_domain_datasets)[0]: # Assuming there's only one in_domain_dataset
      if self.best_in_domain_val_cer > in_domain_cer:
        self.best_in_domain_val_cer = in_domain_cer
        print(f'IMPROVED best in_domain_cer: {self.best_in_domain_val_cer}')
        self.logger.experiment.log({f'best_in_domain_val_cer': self.best_in_domain_val_cer, 'epoch': self.current_epoch})

    if "minusc" not in list(in_domain_datasets)[0]: # Assuming there's only one in_domain_dataset
      if self.best_in_domain_val_wer > in_domain_wer:
        self.best_in_domain_val_wer = in_domain_wer
        print(f'IMPROVED best in_domain_wer: {self.best_in_domain_val_wer}')
        self.logger.experiment.log({f'best_in_domain_val_wer': self.best_in_domain_val_wer, 'epoch': self.current_epoch})

    # self.log('out_of_domain_cer_epoch', out_of_domain_cer, on_epoch=True, prog_bar=True, logger=True)
    self.logger.experiment.log({f'out_of_domain_val_cer_epoch': out_of_domain_cer, 'epoch': self.current_epoch})
    self.logger.experiment.log({f'out_of_domain_val_wer_epoch': out_of_domain_wer, 'epoch': self.current_epoch})

    if "minusc" not in list(in_domain_datasets)[0]: # Assuming there's only one in_domain_dataset
      if self.best_out_domain_val_cer > out_of_domain_cer:
        self.best_out_domain_val_cer = out_of_domain_cer
        print(f'IMPROVED best out_domain_cer: {self.best_out_domain_val_cer}')
        self.logger.experiment.log({f'best_out_domain_val_cer': self.best_out_domain_val_cer, 'epoch': self.current_epoch})
    
    if "minusc" not in list(in_domain_datasets)[0]: # Assuming there's only one in_domain_dataset
      if self.best_out_domain_val_wer > out_of_domain_wer:
        self.best_out_domain_val_wer = out_of_domain_wer
        print(f'IMPROVED best out_domain_wer: {self.best_out_domain_val_wer}')
        self.logger.experiment.log({f'best_out_domain_val_wer': self.best_out_domain_val_wer, 'epoch': self.current_epoch})
      

    # Reset lists
    self.val_losses = {val_dataset: [] for val_dataset in self.val_datasets}
    self.val_accs = {val_dataset: [] for val_dataset in self.val_datasets}
    self.val_cers = {val_dataset: torchmetrics.CharErrorRate() for val_dataset in self.val_datasets}
    self.val_wers = {val_dataset: torchmetrics.WordErrorRate() for val_dataset in self.val_datasets}

    # Return mean_val_cer, in_domain_cer, out_of_domain_cer, heldout_domain_cer

    return mean_val_cer, in_domain_cer, out_of_domain_cer, heldout_domain_cers


  def log_test_metrics(self):
    """Log test metrics and return test_cers, test_wers."""
    # Calculate test CERs
    mean_cer_test, mean_wer_test = 0.0, 0.0

    for dataset in self.test_datasets: # Test dataset is always one
      test_cer = self.cer_test[dataset].compute()
      mean_cer_test += test_cer
      print(f'Test CER on dataset {dataset}: {test_cer}')
      self.logger.experiment.log({f'test/test_cer_' + dataset: test_cer, 'epoch': self.current_epoch})

      test_wer = self.wer_test[dataset].compute()
      mean_wer_test += test_wer
      print(f'Test WER on dataset {dataset}: {test_wer}')
      self.logger.experiment.log({f'test/test_wer_' + dataset: test_wer, 'epoch': self.current_epoch})

    dataset = self.test_datasets[0]

    return test_cer, test_wer


      
  # def log(self, message):
  #     self.log_file.write(message + '\n')
  #     self.log_file.flush()

  def close(self):
      self.log_file.close()