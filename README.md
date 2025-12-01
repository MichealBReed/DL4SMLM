# Deep Learning for Single Molecule Localization Microscopy (DL4SMLM)

## Overview
This codebase contains various deep learning models and helper functions developed for the purpose of analyzing single molecule localization microscopy datasets.

## System Requirements
- The training and validation processes were successfully performed on the following GPU devices on the UNC Chapel Hill Longleaf software:
  - A100 MIG
  - GTX 1080
- Inference was performed using the cpu with the following hardware specifications: 24 x Intel(R) Xeon(R) CPU E5-2643 v3 @ 3.40GHz

## Usage Example

## Read in the data using the custom ImageDataset class.
The function assumes that there are two directories, X and Y, that contain paths to the diffraction limited and localized emitter images respecfully.
 
```python
training_dataset = DL4SMLM.ImageDataset(X_path = "path_to_training_X_data",
                                    Y_path = "path_to_training_Y_data",
                                    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean,std)]))



valid_dataset = DL4SMLM.ImageDataset(X_path = "path_to_validation_X_data",
                                    Y_path = "path_to_validation_Y_data",
                                    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean,std)]))

test_dataset = DL4SMLM.ImageDataset(X_path = "path_to_test_X_data",
                                    Y_path = "path_to_test_Y_data",
                                    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean,std)]))
```



## Load the Dataset into PyTorch's DataLoader class (called through the DL4SMLM library).

```python
training_data_loader = DL4SMLM.DataLoader(training_dataset, batch_size=batch_size)

valid_data_loader = DL4SMLM.DataLoader(valid_dataset, batch_size=batch_size)

test_data_loader = DL4SMLM.DataLoader(test_dataset, batch_size=batch_size)

```

## Instantiate Model, Loss Function, Optimizer and Learning Rate Scheduler
```python
l1l2 = DL4SMLM.L1L2Loss()
srcnn = DL4SMLM.SRCNN()
srcnn_optimizer = torch.optim.Adam(srcnn.parameters(),lr=1e-4)
srcnn_scheduler = lr_scheduler.ReduceLROnPlateau(srcnn_optimizer,'min',patience=5)
```

## Declare a path to save the model weights after training is complete
```python
srcnn_save_path = 'trained_model_weights path'
```


## Train and Validate Model
```python
srcnn_train_loss, srcnn_valid_loss = DL4SMLM.TrainValid(model = srcnn,train_dataloader = training_data_loader,valid_dataloader = valid_data_loader,
                                                                    optimizer = srcnn_optimizer,scheduler = srcnn_scheduler,device = device,
                                                                    save_model_path = srcnn_save_path,epochs=50,lmda = lmda,loss_function = l1l2)

```

## Test Model
```python
ssim, nmse = DL4SMLM.TestModel(srcnn,test_data_loader)
```
