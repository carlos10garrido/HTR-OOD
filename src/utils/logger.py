# Logger class to wrap-up logging functionality
import torchmetrics
import torch
import torch.nn as nn
import torchvision
import wandb
import torchmetrics.text as text_metrics
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MetricLogger():
  def __init__(
    self, 
    logger=None,
    tokenizer=None,
    train_datasets=[],
    val_datasets=[],
    test_datasets=[]):

    # self.pl_logger = pl_logger
    print(f'Initializing MetricLogger...')
    print(f'Logger: {logger}')
    self.logger = logger
    self.tokenizer = tokenizer
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
    
    # Other metrics: confidence, perplexity, calibration
    self.val_confidences = {val_dataset: [] for val_dataset in self.val_datasets}
    self.val_calibrations = {val_dataset: [] for val_dataset in self.val_datasets}
    self.val_int_perplexities = {val_dataset: [] for val_dataset in self.val_datasets} # Ignore BOS, EOS, PAD
    self.val_ext_perplexities = {val_dataset: torchmetrics.text.Perplexity(ignore_index=2).to(device) for val_dataset in self.val_datasets} # Ignore BOS, EOS, PAD
    
    
    # Set test metrics per dataset as a dict()
    self.cer_test = {test_dataset: torchmetrics.CharErrorRate() for test_dataset in self.test_datasets}
    self.wer_test = {test_dataset: torchmetrics.WordErrorRate() for test_dataset in self.test_datasets}
    self.test_confidences = {test_dataset: [] for test_dataset in self.test_datasets}
    self.test_calibrations = {test_dataset: [] for test_dataset in self.test_datasets}
    self.test_int_perplexities = {test_dataset: [] for test_dataset in self.test_datasets}
    self.test_ext_perplexities = {test_dataset: torchmetrics.text.Perplexity(ignore_index=2).to(device) for test_dataset in self.test_datasets} # Ignore BOS, EOS, PAD

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
    
  def calculate_confidence(self, raw_preds):
    return torch.nn.functional.softmax(raw_preds, dim=-1).max(dim=-1).values.mean().item()
    
  def log_val_step_confidence(self, raw_preds, dataset):
    # Compute confidence
    confidences = []
    
    # Calculate confidence for each sequence
    for raw_pred in raw_preds:
      confidence = self.calculate_confidence(raw_pred)
      confidences.append(confidence)
    
    print(f'Confidences: {confidences}')
    self.val_confidences[dataset].extend(confidences)
    
    return confidences
  
  def log_val_step_int_perplexity(self, raw_preds, dataset):
    """ Calculate internal perplexity for each sequence.
        It is basically to exponentiate the confidence of the model in the prediction.
    """
    
    perplexities = []
    # Calculate perplexity for each sequence
    raw_preds = raw_preds.softmax(-1).max(dim=-1).values
    
    for raw_pred in raw_preds:
      perplexity = 2 ** (-1/len(raw_pred) * torch.log(torch.prod(raw_pred, dtype=torch.float64))) # Perplexity = 2^(-1/N * log(P(x))). Float64 to avoid overflow
      perplexities.append(perplexity)
      
    print(f'Perplexities: {perplexities}')
    self.val_int_perplexities[dataset].extend(perplexities)
    
    
  def log_val_step_ext_perplexity(self, preds, labels, dataset):
    """ Calculate external perplexity for each sequence.
        It is the exponentiation of the cross-entropy val-test, since we're (teacher) forcing 
        with the groundtruth predictions. 
        
    """
    ext_perp = self.val_ext_perplexities[dataset].forward(preds, labels)
    print(f'Ext perp {ext_perp}')
      
    
  def log_val_step_calibration(self, raw_preds, labels, dataset):
    calibrations = []
    # Calculate CER for each sequence
    for raw_pred, label in zip(raw_preds, labels):
      pred = raw_pred.softmax(-1).argmax(-1).tolist()
      pred = self.tokenizer.detokenize(pred)
      label = self.tokenizer.detokenize(label.tolist())
      cer_seq = torchmetrics.CharErrorRate()(pred, label)
      confidence = self.calculate_confidence(raw_pred)
      calibration = torch.abs(cer_seq - confidence)
      calibrations.append(calibration)
      
    self.val_calibrations[dataset].extend(calibrations)
    
  ##### 
  
  def log_test_step_confidence(self, raw_preds, dataset):
    # Compute confidence
    confidences = []
    
    # Calculate confidence for each sequence
    for raw_pred in raw_preds:
      confidence = self.calculate_confidence(raw_pred)
      confidences.append(confidence)
    
    print(f'Confidences: {confidences}')
    self.test_confidences[dataset].extend(confidences)
    
    return confidences
  
  def log_test_step_int_perplexity(self, raw_preds, dataset):
    """ Calculate internal perplexity for each sequence.
        It is basically to exponentiate the confidence of the model in the prediction.
    """
    
    perplexities = []
    # Calculate perplexity for each sequence
    raw_preds = raw_preds.softmax(-1).max(dim=-1).values
    
    for raw_pred in raw_preds:
      perplexity = 2 ** (-1/len(raw_pred) * torch.log(torch.prod(raw_pred, dtype=torch.float64))) # Perplexity = 2^(-1/N * log(P(x))). Float64 to avoid overflow
      perplexities.append(perplexity)
      
    print(f'Perplexities: {perplexities}')
    self.test_int_perplexities[dataset].extend(perplexities)
    
    
  def log_test_step_ext_perplexity(self, raw_preds, labels, dataset):
    """ Calculate external perplexity for each sequence.
        It is the exponentiation of the cross-entropy test-test, since we're (teacher) forcing 
        with the groundtruth predictions. 
        
    """
    ext_perp = self.test_ext_perplexities[dataset].forward(raw_preds, labels)
    print(f'Ext perp {ext_perp}')
      
    
  def log_test_step_calibration(self, raw_preds, labels, dataset):
    calibrations = []
    # Calculate CER for each sequence
    for raw_pred, label in zip(raw_preds, labels):
      pred = raw_pred.softmax(-1).argmax(-1).tolist()
      pred = self.tokenizer.detokenize(pred)
      label = self.tokenizer.detokenize(label.tolist())
      cer_seq = torchmetrics.CharErrorRate()(pred, label)
      confidence = self.calculate_confidence(raw_pred)
      calibration = torch.abs(cer_seq - confidence)
      calibrations.append(calibration)
      
    self.test_calibrations[dataset].extend(calibrations)
          
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

    return heldout_domain_cers
      

  def log_val_metrics(self):
    """Log validation metrics and return mean_val_cer, in_domain_cer, out_of_domain_cer, heldout_domain_cer."""

    val_loss, val_acc, val_cer, val_wer, val_conf = {}, {}, {}, {}, {}

    # Compute losses, accuracies and CERs per dataset
    for dataset in self.val_datasets:
      # Loss
      if len(self.val_losses[dataset]) != 0:
        val_loss[dataset] = torch.stack(self.val_losses[dataset]).mean()
        self.logger.experiment.log({f'val/val_loss_' + dataset + '_epoch': val_loss[dataset], 'epoch': self.current_epoch})

      # Accuracy
      if len(self.val_accs[dataset]) != 0:
        val_acc[dataset] = torch.stack(self.val_accs[dataset]).mean()
        self.logger.experiment.log({f'val/val_acc_' + dataset + '_epoch': val_acc[dataset], 'epoch': self.current_epoch})
        # self.log('val_acc_' + dataset + '_epoch', val_acc[dataset], on_epoch=True, prog_bar=True, logger=True)

      # CER
      val_cer[dataset] = self.val_cers[dataset].compute()
      # Check if CER is NaN
      if torch.isnan(val_cer[dataset]):
         [dataset] = 1e5
      self.logger.experiment.log({f'val/val_cer_' + dataset + '_epoch': val_cer[dataset], 'epoch': self.current_epoch})
      
      # Log metric for saving checkpoint to optimise only for one dataset. I.e: training for IAM and saving the best model in Rimes
      self.logger.experiment.log({f'val/optim_target_{dataset}': val_cer[dataset], 'epoch': self.current_epoch})

      # WER
      val_wer[dataset] = self.val_wers[dataset].compute()
      # Check if WER is NaN
      if torch.isnan(val_wer[dataset]):
        val_wer[dataset] = 1e5
      self.logger.experiment.log({f'val/val_wer_' + dataset + '_epoch': val_wer[dataset], 'epoch': self.current_epoch})
      
      # Confidence
      if len(self.val_confidences[dataset]) != 0:
        val_conf[dataset] = torch.tensor(self.val_confidences[dataset]).mean()
      
        # Check if confidence is NaN
        if torch.isnan(val_conf[dataset]):
          val_conf[dataset] = 1e5
          
        self.logger.experiment.log({f'val/val_confidence_' + dataset: val_conf[dataset], 'epoch': self.current_epoch})
        
      # Calibration
      if len(self.val_calibrations[dataset]) != 0:
        val_cal = torch.tensor(self.val_calibrations[dataset]).mean()
        # Check if calibration is NaN
        if torch.isnan(val_cal):
          val_cal = 1e5
        self.logger.experiment.log({f'val/val_calibration_' + dataset: val_cal, 'epoch': self.current_epoch})
        
      # Perplexities (internal and external)
      if len(self.val_int_perplexities[dataset]) != 0:
        val_int_perplexity = torch.tensor(self.val_int_perplexities[dataset]).mean()
        # Check if perplexity is NaN
        if torch.isnan(val_int_perplexity):
          val_int_perplexity = 1e5
        self.logger.experiment.log({f'val/val_int_perplexity_' + dataset: val_int_perplexity, 'epoch': self.current_epoch})
        
      val_ext_perplexity = {}
      val_ext_perplexity[dataset] = self.val_ext_perplexities[dataset].compute()      
      # Check if perplexity is NaN
      if torch.isnan(val_ext_perplexity[dataset]):
        val_ext_perplexity[dataset] = 1e5
      self.logger.experiment.log({f'val/val_ext_perplexity_' + dataset: val_ext_perplexity[dataset], 'epoch': self.current_epoch})


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


    print(f'IN-DOMAIN CER = {in_domain_cer}')
    print(f'OUT-OF-DOMAIN CER = {out_of_domain_cer}')
    print(f'IN-DOMAIN WER = {in_domain_wer}')
    print(f'OUT-OF-DOMAIN WER = {out_of_domain_wer}')


    # self.log('in_domain_cer_epoch', in_domain_cer, on_epoch=True, prog_bar=True, logger=True)
    self.logger.experiment.log({f'val/in_domain_val_cer_epoch': in_domain_cer, 'epoch': self.current_epoch})
    self.logger.experiment.log({f'val/in_domain_val_wer_epoch': in_domain_wer, 'epoch': self.current_epoch})
    self.logger.experiment.log({f'val/out_of_domain_val_cer_epoch': out_of_domain_cer, 'epoch': self.current_epoch})
    self.logger.experiment.log({f'val/out_of_domain_val_wer_epoch': out_of_domain_wer, 'epoch': self.current_epoch})
    
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
    self.val_confidences = {val_dataset: [] for val_dataset in self.val_datasets}
    self.val_calibrations = {val_dataset: [] for val_dataset in self.val_datasets}
    self.val_int_perplexities = {val_dataset: [] for val_dataset in self.val_datasets}
    self.val_ext_perplexities = {val_dataset: torchmetrics.Perplexity(ignore_index=2).to(device) for val_dataset in self.val_datasets}

    return mean_val_cer, in_domain_cer, out_of_domain_cer, heldout_domain_cers, val_cer


  def log_test_metrics(self):
    """Log test metrics and return test_cers, test_wers, test_confidences, test_perplexities, test_calibrations."""
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
      
      # Confidence
      test_conf = {}
      if len(self.test_confidences[dataset]) != 0:
        test_conf[dataset] = torch.tensor(self.test_confidences[dataset]).mean()
      
        # Check if confidence is NaN
        if torch.isnan(test_conf[dataset]):
          test_conf[dataset] = 1e5
          
        self.logger.experiment.log({f'test/test_confidence_' + dataset: test_conf[dataset], 'epoch': self.current_epoch})
        
      # Calibration
      if len(self.test_calibrations[dataset]) != 0:
        test_cal = torch.tensor(self.test_calibrations[dataset]).mean()
        # Check if calibration is NaN
        if torch.isnan(test_cal):
          test_cal = 1e5
        self.logger.experiment.log({f'test/test_calibration_' + dataset: test_cal, 'epoch': self.current_epoch})
        
      # Perplexities (internal and external)
      if len(self.test_int_perplexities[dataset]) != 0:
        test_int_perplexity = torch.tensor(self.test_int_perplexities[dataset]).mean()
        # Check if perplexity is NaN
        if torch.isnan(test_int_perplexity):
          test_int_perplexity = 1e5
        self.logger.experiment.log({f'test/test_int_perplexity_' + dataset: test_int_perplexity, 'epoch': self.current_epoch})
        
      test_ext_perplexity = {}
      test_ext_perplexity[dataset] = self.test_ext_perplexities[dataset].compute()      
      # Check if perplexity is NaN
      if torch.isnan(test_ext_perplexity[dataset]):
        test_ext_perplexity[dataset] = 1e5
      self.logger.experiment.log({f'test/test_ext_perplexity_' + dataset: test_ext_perplexity[dataset], 'epoch': self.current_epoch})
      

    dataset = self.test_datasets[0]

    return test_cer, test_wer
      
  # def log(self, message):
  #     self.log_file.write(message + '\n')
  #     self.log_file.flush()

  def close(self):
      self.log_file.close()