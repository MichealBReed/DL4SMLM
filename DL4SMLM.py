from torch import nn
import torch,os,tifffile,torchmetrics
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F



torch.set_default_dtype(torch.float32)


# Helper Dataset Function

class ImageDataset(Dataset):
    """
    Dataset class dedicated to SMLM data bulit using the Dataset class in the Pytorch library.
    """
    
    def __init__(self,X_path,Y_path,transform=None):

        """
        Parameters
        ------------
        X_path : str
            String containing the path toward the diffraction limited images.

        Y_path : str
            String containing the path toward the super resolved emitter images.

        transform : Pytorch Compose object (torchvision.transforms.Compose) containing any combination of torchvision.transforms methods / 
                    to be applied to the diffraction limited image data.
        """

        self.transform = transform
        #load data with images
        
        #load diffraction limited data
        self.X_names = os.listdir(X_path)
        self.X = [tifffile.imread(X_path +"/" + file_name).astype(np.float32) for file_name in self.X_names]
        self.n_images = len(self.X_names)
        
        #load super resolution data
        self.Y_names = os.listdir(Y_path)
        self.Y = [tifffile.imread(Y_path +"/" + file_name).astype(np.float32) for file_name in self.Y_names]
        
        
    def __getitem__(self,index):


        if self.transform:

            X = self.transform(self.X[index])

            Y = self.Y[index]

            return X,Y

        else:
            X = self.X[index]

            Y = self.Y[index]

            return X,Y
    
    def __len__(self):
        return self.n_images

# Performance Metrics
    
class WeightedMSE(nn.Module):
    """
    An implementation of the Weighted Mean Squared Error loss function constructed using the nn module of the Pytorch library.  
    """
    def __init__(self):
        super(WeightedMSE, self).__init__()

    def forward(self, prediction, target,lmda):

        """
        Parameters
        ------------
        prediction : tensor
            Predicted super resolved image of emitters. 

        target : tensor
            Ground truth super resolved image of emitters. 

        weight : int
            Weight value which all non-zero pixel values will be scaled by.

         lmda : int
            lambda parameter for the L1 regularization term.
        """
        
        loss = ((lmda / 2) * ((prediction - target)**2)).mean().requires_grad_() + prediction.abs().mean().requires_grad_()
        
        return loss
    
class WeightedMAE(nn.Module):
    """
    An implementation of the Weighted Mean Absilute Error loss function constructed using the nn module of the Pytorch library.  
    """
    def __init__(self):
        super(WeightedMAE, self).__init__()

    def forward(self, prediction, target,lmda):

        """
        Parameters
        ------------
        prediction : tensor
            Predicted super resolved image of emitters. 

        target : tensor
            Ground truth super resolved image of emitters. 

        weight : int
            Weight value which all non-zero pixel values will be scaled by.

         lmda : int
            lambda parameter for the L1 regularization term.
        """
        
        loss = ((lmda / 2) * torch.abs(prediction - target)).mean().requires_grad_() + prediction.abs().mean().requires_grad_()
        
        return loss
    

class L1L2Loss(nn.Module):
    """
    An implementation of the custom loss function introduced in the original Deep-STORM paper written by Nehme et al 2018.
    It involves Gaussian convolutions using a 3x3 kernel over the predicted and ground truth images and a subsequent calculation of the /
    Mean Squared Error. Additionally, an L1 regularization term consisting of the predicted image is incorporated to control/
    for sparsity. In theory, decreasing the lambda parameter aggresively promotes sparsity while increasing it coerces stricter /
    alignment to the data effectively preserving more of the data. 
    
    """
    def __init__(self):
        super(L1L2Loss,self).__init__()
        self.n = np.zeros((3,3),dtype=np.float32)
        self.n[2,2] = 1
        self.gaussian_filter = gaussian_filter(self.n,sigma = 1)
        self.gaussian_kernel = torch.from_numpy(self.gaussian_filter)
        self.gk = self.gaussian_kernel.reshape(1,1,3,3)
        self.register_buffer('gaussian_weights', self.gk)

    def forward(self,prediction,target,lmda):
        """
        Parameters
        ------------
        prediction : tensor
            Predicted super resolved image of emitters. 

        target : tensor
            Ground truth super resolved image of emitters. 

        lmda : int
            lambda parameter for the L1 regularization term.
        """

        
        predicted_convolution = F.conv2d(prediction,self.gaussian_weights
                                         )

        truth_convolution = F.conv2d(target,self.gaussian_weights
                                         )

        # Compute the Loss

        loss = ((lmda / 2) * (predicted_convolution - truth_convolution)**2).mean().requires_grad_() + (predicted_convolution.abs().mean().requires_grad_())

        return loss
    
class AttentionImitationLoss(nn.Module):
    """
    An implementation of a loss function from Risqi et al 2019 paper titled "Distilling Knowledge from a Deep Pose Regressor Network".
    This function is designed to transfer single molecule localization ability from a teacher network onto a student network. While the 
    original work uses Mean Square Error, this implementation relies on the L1L2loss from the DL4SMLM library. 
    
    """
    def __init__(self):
        super(AttentionImitationLoss,self).__init__()
        self.l1l2loss = L1L2Loss()

    def forward(self,student_prediction,teacher_prediction,ground_truth,
                teacher_training_loss_vector,alpha,lmda):
        
        """
        Parameters
        ------------
        student_prediction : tensor
            Predicted super resolved image of emitters from the student model that is being trained. 

        teacher_prediction : tensor
            Predicted super resolved image of emitters from a teacher model 

        ground_truth : tensor
            Ground truth super resolved image of emitters. 

        teacher_training_loss_vector : vector of floats
            Vector containing the teacher loss values during its training phase.

        alpha : float [0-1]
            Hyperparameter controlling if the student network should focus more on learning more from the ground truth or from the teacher network. 
            A larger alpha will be more attentive to the ground truth while a smaller will focus more on the teacher model.

        lmda : int
            Hyperparameter for the L1 regularization term controlling sparsity of the L1L2loss function

        
        """
        
        teacher_l1l2 = self.l1l2loss
        student_l1l2 = self.l1l2loss
        teacher2student_l1l2 = self.l1l2loss
        
        max_error = np.max(teacher_training_loss_vector)
        min_error = np.min(teacher_training_loss_vector)

        normalization_factor = max_error - min_error

        teacher_error = teacher_l1l2(teacher_prediction,ground_truth,lmda)
        phi = 1 - (teacher_error / normalization_factor)

        student_error = student_l1l2(student_prediction,ground_truth,lmda)

        teacher2student_error = teacher2student_l1l2(student_prediction,teacher_prediction,lmda)

        loss = (alpha * student_error) + ((1 - alpha) * phi * teacher2student_error)

        return loss.mean().requires_grad_()






def NMSE(prediction,ground_truth):
    """
    An implementation of the Normalized Mean Square Error as described in Nehme et al 2018 and other related works./
    Briefly, it is the square of the difference between the predicted image and the ground truth divided by the sum of the ground truth.
    This value is then multiplied by 100. 

    Parameters
        ------------
        prediction : tensor
            Predicted super resolved image of emitters. 

        ground_truth : tensor
            Ground truth super resolved image of emitters. 
    """


    sum_of_difference = torch.sum(prediction - ground_truth)

    top = torch.square(sum_of_difference)

    bottom = torch.square(torch.sum(ground_truth))

    nmse = (top / bottom ) * 100

    return nmse.item()


# Model Architectures

class SRCNN(nn.Module):
    """
    Super Resolution Convolutional Nerual Network:
    Published by Dong et al 2014 in the paper titled 'Learning a Deep Convolutional Network for Image Super Resolution.
    This architecture was mainly developed to optimize traditional image processing workflows, not necessarily bioimaging.
    """


    def __init__(self):
        super().__init__()

        self.feature_representation = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64, kernel_size=3,padding="same"),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.non_linear_mapping = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1,padding="same"),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )

        self.reconstruction = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=1,kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=1)
        )
        
    def forward(self,image_input):

        """
        Parameters
        ------------
        image_input : tensor
            Diffraction limited image of emitters

        Outputs
        ------------
        x : tensor 
            Super resolved image of localized emitters

        """
        #Patch Extraction and representation
        x = self.feature_representation(image_input)
        #Non-linear mapping
        x = self.non_linear_mapping(x)
        #Reconstruction
        x = self.reconstruction(x)
        return x



class USTORM(nn.Module):
    """
    U-Net for Stochastic Optical Reconstruction Microscopy:
    This architecture is inspired by the original U-Net model developed for biomedical image segmentation by Ronneberger et al in 2015 in a paper titled/
    'U-net: Convolutional networks for biomedical image segmentation'. However, this architecture is not as deep as it also inspired by Yao et al 2018 who /
    applied a U-Net-esque architecture to solve the pansharpening problem in remote sensing in a paper titled 'Pixel-wise regression using U-net and its application/
    on pan-sharpening'.
    """

    def __init__(self):
        super().__init__()

        self.feature_representation = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )

        self.first_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.second_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )

        self.third_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=32,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )

        self.fourth_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=96,out_channels=16,kernel_size=3,padding="same"),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU()
        )


        self.reconstruction = nn.Sequential(
            nn.Conv2d(in_channels=48,out_channels=1,kernel_size=1,padding="same"),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.up = nn.Upsample(scale_factor = 2)

    def forward(self,input_image):

        """
        Parameters
        ------------
        input_image : tensor
            Diffraction limited image of emitters

        Outputs
        ------------
        reconstructed_emitters : tensor 
            Super resolved image of localized emitters

        """

        initial_features = self.feature_representation(input_image)

        first_convolution_output = self.first_convolution_block(initial_features)
        first_convolution_pool_output = self.pool(first_convolution_output)

        second_convolution_output = self.second_convolution_block(first_convolution_pool_output)
        second_convolution_pool_output = self.pool(second_convolution_output)

        third_convolution_output = self.third_convolution_block(second_convolution_pool_output)
        third_convolution_upsample_output = self.up(third_convolution_output)

        fourth_convolution_input = torch.cat((first_convolution_pool_output,third_convolution_upsample_output),1)
        fourth_convolution_output = self.fourth_convolution_block(fourth_convolution_input)
        fourth_convolution_upsample_output = self.up(fourth_convolution_output)

        reconstruction_input = torch.cat((initial_features,fourth_convolution_upsample_output),1)
        reconstructed_emitters = self.reconstruction(reconstruction_input)

        return reconstructed_emitters


class DeepSTORM(nn.Module):
    """
    Deep Convolutional Neural Network for Stochastic Optical Reconstruction Microscopy :
    Published by Nehme et al 2018 in the paper titled 'Deep-STORM: super resolution single-molecule microscopy by deep learning'. To our knowledge/
    it is the first Convolutional Neural Net dedicated towards single molecule microscopy.
    """
    def __init__(self):
        super().__init__()

        self.first_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.second_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.third_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fourth_convolution_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128,out_channels=512,kernel_size=3,padding="same"),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )

        self.fifth_convolution_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=128,kernel_size=3,padding="same"),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )

        self.sixth_convolution_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128,out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )

        self.seventh_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )

        self.reconstruction = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1,padding="same"),
            nn.BatchNorm2d(num_features=1)
        )
        


    def forward(self,image_input):

        """
        Parameters
        ------------
        image_input : tensor
            Diffraction limited image of emitters

        Outputs
        ------------
        reconstructed_image : tensor 
            Super resolved image of localized emitters

        """

        ##Encoder Phase
        first_conv_output = self.first_convolution_block(image_input)

        

        second_conv_output = self.second_convolution_block(first_conv_output)
       

        third_conv_output = self.third_convolution_block(second_conv_output)
        

        ##Decoder Phase
        fourth_conv_output = self.fourth_convolution_block(third_conv_output)
        

        fifth_conv_output = self.fifth_convolution_block(fourth_conv_output)
        

        sixth_conv_output = self.sixth_convolution_block(fifth_conv_output)
        

        seventh_conv_output = self.seventh_convolution_block(sixth_conv_output)
        

        #Non-Linear Mapping
        reconstructed_image = self.reconstruction(seventh_conv_output)
        
        return reconstructed_image



class SkipSTORM(nn.Module):
    """
    Deep Convolutional Neural Network for Stochastic Optical Reconstruction Microscopy with a Skip Connection :
    Similar architecture as Deep-STORM except there is a skip connection between the input image and the reconstruction layer to provide/
    spatial context from the original image.
    """
    def __init__(self):
        super().__init__()

        self.first_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.second_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.third_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fourth_convolution_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128,out_channels=512,kernel_size=3,padding="same"),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )

        self.fifth_convolution_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=128,kernel_size=3,padding="same"),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )

        self.sixth_convolution_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128,out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )

        self.seventh_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )

        self.reconstruction = nn.Sequential(
            nn.Conv2d(in_channels=33, out_channels=1, kernel_size=3,padding="same"),
            nn.BatchNorm2d(num_features=1)
        )
        


    def forward(self,image_input):
        """
        Parameters
        ------------
        image_input : tensor
            Diffraction limited image of emitters

        Outputs
        ------------
        reconstructed_image : tensor 
            Super resolved image of localized emitters

        """

        ##Encoder Phase
        first_conv_output = self.first_convolution_block(image_input)

        

        second_conv_output = self.second_convolution_block(first_conv_output)

        

        third_conv_output = self.third_convolution_block(second_conv_output)
        

        ##Decoder Phase
        fourth_conv_output = self.fourth_convolution_block(third_conv_output)
        

        fifth_conv_output = self.fifth_convolution_block(fourth_conv_output)
        

        sixth_conv_output = self.sixth_convolution_block(fifth_conv_output)
        

        seventh_conv_output = self.seventh_convolution_block(sixth_conv_output)
        

        reconstruction_convolution_input = torch.cat((seventh_conv_output,image_input),1)
        
        #Non-Linear Mapping
        reconstructed_image = self.reconstruction(reconstruction_convolution_input)
        
        return reconstructed_image
    


class DRLSTORM(nn.Module):
    """
    Image reconstruction with a deep convolutional neural network in high-density super-resolution microscopy :
    Published by Qu et al 2020 in the paper titled 'Image Reconstruction with a deep /
    convolutional neural network in high density super-resolution microscopy'. This work aimed to develop an architecture
    that could facilitate live STORM imaging with a high density of emitters.
    """

    def __init__(self):
        super().__init__()

        self.n = np.zeros((3,3),dtype=np.float32)
        self.n[2,2] = 1
        self.gaussian_filter = gaussian_filter(self.n,sigma = 1)
        self.gaussian_kernel = torch.from_numpy(self.gaussian_filter)
        self.gk = self.gaussian_kernel.reshape(1,1,3,3)
        self.gaussian_kernels = self.gk.repeat(32,32,1,1)
        self.register_buffer('gaussian_weights', self.gaussian_kernels)

        self.first_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )

        self.second_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.third_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fourth_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=592, kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=592),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fifth_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=592,out_channels=592,padding='same', kernel_size=3),
            nn.BatchNorm2d(num_features=592),
            nn.ReLU(),
        )

        self.sixth_convolution_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=592,out_channels=128,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
            
        )

        self.seventh_convolution_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.eigth_convolution_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )

        self.ninth_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.tenth_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.eleventh_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=592, kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=592),
            nn.MaxPool2d(kernel_size=2)
        )


        self.twelfth_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=592,out_channels=592,padding='same', kernel_size=3),
            nn.BatchNorm2d(num_features=592),
            nn.ReLU(),
        )

        self.thirtheenth_convolution_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=592,out_channels=128,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )

        self.fourteenth_convolution_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )


        self.fifteenth_convolution_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )

        self.reconstruction = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=1,kernel_size=3,padding='same'),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
        )


    def forward(self,image_input):
        """
        Parameters
        ------------
        input_image : tensor
            Diffraction limited image of emitters

        Outputs
        ------------
        resolved_emitters : tensor 
            Super resolved image of localized emitters

        """

        ##First Half

        x = self.first_convolution_block(image_input)

        x = self.second_convolution_block(x)

        x = self.third_convolution_block(x)

        x = self.fourth_convolution_block(x)

        x = self.fifth_convolution_block(x)

        x = self.sixth_convolution_block(x)

        x = self.seventh_convolution_block(x)

        x = self.eigth_convolution_block(x)

        

        ## Residual Block
        
        gauss_conv = F.conv2d(x,self.gaussian_weights,padding='same')
        
                                         
        residual = image_input - (gauss_conv)
        

        ##Second Half
        x = self.ninth_convolution_block(residual)

        x = self.tenth_convolution_block(x)

        x = self.eleventh_convolution_block(x)

        x = self.twelfth_convolution_block(x)

        x = self.thirtheenth_convolution_block(x)

        x = self.fourteenth_convolution_block(x)

        x = self.fifteenth_convolution_block(x)

        resolved_emitters = self.reconstruction(x)

        return resolved_emitters
    



class FIDSTORM(nn.Module):
    """
    Fast Dense Image Reconstruction based Deep Learning in STORM
    Published by Zhou et al 2023 in the paper titled 'Deep learning using a residual deconvolutional network enables real-time/
    high desnity single-molecule localization microscopy'. This model aims to circumvent the data processing bottlenecks of Deep-STORM/
    and DRL-STORM by learning directrly from the low resolution raw images.
    """

    def __init__(self):
        super().__init__()

        self.input_convolution = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding='same')


        self.first_residual_block = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, kernel_size = 3, padding = 'same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.second_residual_block = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, kernel_size = 3, padding = 'same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.third_residual_block = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, kernel_size = 3, padding = 'same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.fourth_residual_block = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, kernel_size = 3, padding = 'same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.fifth_residual_block = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, kernel_size = 3, padding = 'same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.first_deconvolution = nn.Conv2d(in_channels = 128,out_channels = 32, kernel_size = 3, padding = 'same')
        
        self.second_deconvolution = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = 'same')

        self.third_deconvolution = nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 3, padding = 'same')


    def forward(self,input_image):
        """
        Parameters
        ------------
        input_image : tensor
            Diffraction limited image of emitters

        Outputs
        ------------
        reconstructed_emitters : x 
            Super resolved image of localized emitters

        """

        input_convolution_output = self.input_convolution(input_image)
        
        x = self.first_residual_block(input_convolution_output)

        x = self.second_residual_block(x)

        x = self.third_residual_block(x)

        x = self.fourth_residual_block(x)

        x = self.fifth_residual_block(x)

        x = torch.cat((input_convolution_output,x),1)

        x = self.first_deconvolution(x)

        x = self.second_deconvolution(x)

        x = self.third_deconvolution(x)

        return x

# Training, Validation and Test Model Helper Functions

def TrainValid(model = torch.nn.Module,train_dataloader = torch.utils.data.DataLoader,
               valid_dataloader = torch.utils.data.DataLoader,
               optimizer = torch.optim,device="cpu", 
               epochs = 30, scheduler = None, 
               batch_size=5,save_model_path=".",
               lmda=1,loss_function = None):
    

    """
    Function that automates the training and validation process of a model.

    Parameters
        ------------
        model : torch.nn.Module
            Model to be trained

        train_dataloader : torch.utils.data.DataLoader
            Dataloader object containing the X,Y pairs of the training data.

        valid_dataloader : torch.utils.data.DataLoader
            Dataloader object containing the X,Y pairs of the validation data.

        optimizer: torch.optim.
            Optimizer algorithm to be used during training.

        device : str 
            Device that training and validation will be performed on.

        epochs : int
            Number of training epochs

        scheduler : torch.optim.lr_scheduler
            Learning rate scheduler algorithm to be used during learning,
            

        batch_size : int


        save_model_path : str
            Path to save the model weights after trained for the specified number of epochs

        lmda : int
            Lambda parameter to be used during the L1L2 loss computation.

        loss_function : torch.nn.Module
            Loss function to be used during training and validation.

    Outputs
        ------------
        1. Training Loss Vector : list
            A vector of floats that has a length equal to the training_epochs parameter containing the average loss per batch after each training epoch.

        2. Validation Loss Vector : list
            A vector of floats that has a length equal to the training_epochs parameter containing the average loss per batch after each validation epoch.

        3. Model Weights (optional)
            Parameters of the trained model that can be uploaded into another instance of the architecture which was trained.

    """
    
    model.to(device)
    loss_function.to(device)

    train_loss_vector = []

    valid_loss_vector = []
    
    
    
    for epoch in range(0,epochs):
        
        model.train(True)
        training_loss = 0

        for batch, (X,y) in enumerate(train_dataloader):

            N,C,H,W = X.shape

            X,y = X.to(device), y.to(device)

            y_pred = model(X)
         
            
            #Compute the L1L2 bump loss metric
            loss = loss_function(y_pred,y.unsqueeze(1),lmda)
            

            training_loss+=loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            

        
        avg_training_loss_per_batch = training_loss / len(train_dataloader)
       
        train_loss_vector.append(avg_training_loss_per_batch)

        valid_loss = 0

        with torch.inference_mode():

            for batch, (X,y) in enumerate(valid_dataloader):

                N,C,H,W = X.shape

                X,y = X.to(device), y.to(device)

                y_pred = model(X)
                
                #Compute the L1L2 bump loss metric
                loss = loss_function(y_pred,y.unsqueeze(1),lmda)
                
                valid_loss+=loss.item()

        avg_valid_loss_per_batch = valid_loss / len(valid_dataloader)
        
        scheduler.step(avg_valid_loss_per_batch)
        

        valid_loss_vector.append(avg_valid_loss_per_batch)
        
    if save_model_path == None:
        return train_loss_vector, valid_loss_vector
    else:
        optimized_weights = torch.save(model.state_dict(),save_model_path)
        return train_loss_vector, valid_loss_vector

def HintTraining(teacher_model = torch.nn.Module,student_model = torch.nn.Module,
                 train_dataloader = torch.utils.data.DataLoader,
               valid_dataloader = torch.utils.data.DataLoader,
               optimizer = torch.optim,device="cpu", 
               epochs = 30, scheduler = None, 
               batch_size=5,save_model_path=".",loss_function = None):
    

    """
    Function that automates the training and validation process of a model.

    Parameters
        ------------
        model : torch.nn.Module
            Model to be trained

        train_dataloader : torch.utils.data.DataLoader
            Dataloader object containing the X,Y pairs of the training data.

        valid_dataloader : torch.utils.data.DataLoader
            Dataloader object containing the X,Y pairs of the validation data.

        optimizer: torch.optim.
            Optimizer algorithm to be used during training.

        device : str 
            Device that training and validation will be performed on.

        epochs : int
            Number of training epochs

        scheduler : torch.optim.lr_scheduler
            Learning rate scheduler algorithm to be used during learning,
            
        batch_size : int

        save_model_path : str
            Path to save the model weights after trained for the specified number of epochs

        loss_function : torch.nn.Module
            Loss function to be used during training and validation.

    Outputs
        ------------
        1. Training Loss Vector : list
            A vector of floats that has a length equal to the training_epochs parameter containing the average loss per batch after each training epoch.

        2. Validation Loss Vector : list
            A vector of floats that has a length equal to the training_epochs parameter containing the average loss per batch after each validation epoch.

        3. Model Weights (optional)
            Parameters of the trained model that can be uploaded into another instance of the architecture which was trained.

    """
    
    teacher_model.to(device)
    teacher_model.eval()
    student_model.to(device)
    loss_function.to(device)

    train_loss_vector = []

    valid_loss_vector = []
    
    
    
    for epoch in range(0,epochs):
        
        student_model.train(True)
        training_loss = 0

        for batch, (X,y) in enumerate(train_dataloader):

            X,y = X.to(device), y.to(device)

            student_predict = student_model(X)
            teacher_predict = teacher_model(X)
         
            loss = loss_function(student_predict,teacher_predict)
            
            training_loss+=loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            

        
        avg_training_loss_per_batch = training_loss / len(train_dataloader)
       
        train_loss_vector.append(avg_training_loss_per_batch)

        valid_loss = 0

        with torch.inference_mode():

            for batch, (X,y) in enumerate(valid_dataloader):

                N,C,H,W = X.shape

                X,y = X.to(device), y.to(device)

                student_predict = student_model(X)
                teacher_predict = teacher_model(X)
                
                
                loss = loss_function(student_predict,teacher_predict)
                
                valid_loss+=loss.item()

        avg_valid_loss_per_batch = valid_loss / len(valid_dataloader)
        
        scheduler.step(avg_valid_loss_per_batch)
        

        valid_loss_vector.append(avg_valid_loss_per_batch)
        
    if save_model_path == None:
        return train_loss_vector, valid_loss_vector
    else:
        optimized_weights = torch.save(student_model.state_dict(),save_model_path)
        return train_loss_vector, valid_loss_vector

def KnowledgeTransfer(student_model = torch.nn.Module,teacher_model = torch.nn.Module,
                train_dataloader = torch.utils.data.DataLoader,
                valid_dataloader = torch.utils.data.DataLoader,
               optimizer = torch.optim,device="cpu", 
               epochs = 30, scheduler = None, 
               batch_size=5,save_model_path=".",
               lmda=1,loss_function = None,
               alpha=0.5,teacher_training_loss_vector=None,
               teacher_valid_loss_vector=None):
    

    """
    Function that automates the training and validation process of a model.

    Parameters
        ------------
        student_model : torch.nn.Module
            Model to be trained

        teacher_model : torch.nn.Module
            Trained teacher model.

        train_dataloader : torch.utils.data.DataLoader
            Dataloader object containing the X,Y pairs of the training data.

        valid_dataloader : torch.utils.data.DataLoader
            Dataloader object containing the X,Y pairs of the validation data.

        optimizer: torch.optim.
            Optimizer algorithm to be used during training.

        device : str 
            Device that training and validation will be performed on.

        epochs : int
            Number of training epochs

        alpha : float
            Float controlling the attentiveness of the student model to the ground truth or to the teacher model.
            See the help message for the AttentiveImitationLoss function for more information.

        scheduler : torch.optim.lr_scheduler
            Learning rate scheduler algorithm to be used during learning,
            

        batch_size : int

        save_model_path : str
            Path to save the model weights after trained for the specified number of epochs

        lmda : int
            Lambda parameter to be used during the L1L2 loss computation.

        loss_function : torch.nn.Module
            Loss function to be used during training and validation.

        teacher_training_loss_vector : list
            Vector containing the loss values of the teacher model during the training phase.

        teacher_valid_loss_vector : list
            Vector containing the loss values of the teacher model during the validation phase.

        

    Outputs
        ------------
        1. Training Loss Vector : list
            A vector of floats that has a length equal to the training_epochs parameter containing the average loss per batch after each training epoch.

        2. Validation Loss Vector : list
            A vector of floats that has a length equal to the training_epochs parameter containing the average loss per batch after each validation epoch.

        3. Model Weights (optional)
            Parameters of the trained model that can be uploaded into another instance of the architecture which was trained.

    """
    
    student_model.to(device)
    teacher_model.to(device)
    teacher_model.eval()

    loss_function.to(device)

    train_loss_vector = []

    valid_loss_vector = []
    
    
    
    for epoch in range(0,epochs):
        
        student_model.train(True)
        training_loss = 0

        for batch, (X,y) in enumerate(train_dataloader):

            X,y = X.to(device), y.to(device)

            student_pred = student_model(X)
            teacher_pred = teacher_model(X)
         
            
            loss = loss_function(student_pred,teacher_pred,y.unsqueeze(1),
                                 teacher_training_loss_vector,alpha,lmda)
            

            training_loss+=loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            

        
        avg_training_loss_per_batch = training_loss / len(train_dataloader)
       
        train_loss_vector.append(avg_training_loss_per_batch)

        valid_loss = 0

        with torch.inference_mode():

            for batch, (X,y) in enumerate(valid_dataloader):

                X,y = X.to(device), y.to(device)

                student_pred = student_model(X)
                teacher_pred = teacher_model(X)
                
                
                loss = loss_function(student_pred,teacher_pred,y.unsqueeze(1),
                                 teacher_valid_loss_vector,alpha,lmda)
                
                valid_loss+=loss.item()

        avg_valid_loss_per_batch = valid_loss / len(valid_dataloader)
        
        scheduler.step(avg_valid_loss_per_batch)
        

        valid_loss_vector.append(avg_valid_loss_per_batch)
        
    if save_model_path == None:
        return train_loss_vector, valid_loss_vector
    else:
        optimized_weights = torch.save(student_model.state_dict(),save_model_path)
        return train_loss_vector, valid_loss_vector

    return train_loss_vector, valid_loss_vector


def TestModel(model = torch.nn.Module,test_dataloader = torch.utils.data.DataLoader,inference_device="cpu"):

    """
    Function that automates the testing of a model. It will iterate through each X,Y data pair in the test_dataloader variable,/
    perform an inference on X, threshold any negative values to 0, and then compute the Structured Similarity Image Metric /
    and the Normalized Mean Square Error between the inferred image and the ground truth Y. 

    Parameters
        ------------
        model : torch.nn.Module
            A trained SMLM model.

        test_dataloader : torch.utils.data.DataLoader
            String containing the path toward the super resolved emitter images.

        inference_device : str
            String containing the path toward the super resolved emitter images.

    Outputs
        ------------
        1. Structured Similarity Image Metric Vector : list
            A vector of Structured Similarity Image Metric that has a length equal /
            to the number of data pairs in the test_dataloader 

        2. Normalized Mean Square Error : list
            A vector of Normalized Mean Square Error that has a length equal /
            to the number of data pairs in the test_dataloader
    """
 
    ssim_vector = []
    nmse_vector = []
    

    model.eval()

    ssim = torchmetrics.image.StructuralSimilarityIndexMeasure()
    

    with torch.inference_mode():

        model.to(inference_device)

        for batch, (X,y) in enumerate(test_dataloader):

            X,y = X.to(inference_device), y.to(inference_device)

            
            y_pred = model(X)
            

            thresh = torch.nn.Threshold(0,0)

            y_pred = thresh(y_pred)
            
            y = y.unsqueeze(1)
            
            ssim_metric = ssim(y_pred,y).item()
            

            nmse_metric = NMSE(y_pred,y)

            ssim_vector.append(ssim_metric)
            
            nmse_vector.append(nmse_metric)

            

    return ssim_vector,nmse_vector


def VisualizeInference(model = torch.nn.Module,test_dataset = None ,inference_device="cpu",
                       index=0,save_name=None,model_name=None):
    """
    Function that visualizes the diffraction limited, ground truth super resolved, and the predicted super resolved images/
    respectively.

    Parameters
        ------------
        model : torch.nn.Module
            A trained SMLM model.

        test_dataset : DL4SMLM.ImageDataset object
            ImageDataset object from the DL4SMLM library 

        inference_device : str
            String containing the device where inference will occur.

        index : int
            Integer bound between 0 and length(test_dataset) - 1 inclusive. Designates which X,Y data pair will be visualized.

        save_name : str (optional)
            If the image is to be saved then provide a name to save the resulting image else the image will be displayed to the screen.

        model_name : str
            The name of the model that will be displayed over its infered image in the last column of the resulting plot.

    Outputs
        ------------
        1. A 1 x 3 image containing the diffraction limited, ground truth super resolved, and the predicted super resolved images/
        respectively. If a save name is provided then the image is written to a file compatible with the matplotlib.pylplot library.
    """
    
    x,y = test_dataset[index]
    
    x = x.unsqueeze(0)
    
    
    model.eval()
    
    with torch.inference_mode(True):
        
        model_predict = model(x)
        
        threshold = torch.nn.Threshold(0,0)
        
        model_predict_threshold = threshold(model_predict)
        
    fig,ax = plt.subplots(1,3,figsize=(20,16))
    
    model_predict_array = model_predict_threshold.squeeze().numpy()
    
    
    if save_name == None:
        x = x.squeeze().numpy()
        ax[0].imshow(x)
        ax[0].set_title("Diffraction_Limted")
        im_ratio = x.shape[1]/x.shape[0]
        plt.colorbar(ax[0].imshow(x), fraction=0.047*im_ratio)

        ax[1].imshow(y)
        ax[1].set_title("Ground_Truth")
        im_ratio = y.shape[1]/y.shape[0]
        plt.colorbar(ax[1].imshow(y), fraction=0.047*im_ratio)

        ax[2].imshow(model_predict_array)
        ax[2].set_title(model_name)
        im_ratio = model_predict_array.shape[1]/model_predict_array.shape[0]
        plt.colorbar(ax[2].imshow(model_predict_array),fraction=0.047*im_ratio)



        plt.tight_layout()
        
        plt.show()
    
    else:
        x = x.squeeze().numpy()
        ax[0].imshow(x)
        ax[0].set_title("Diffraction_Limted")
        im_ratio = x.shape[1]/x.shape[0]
        plt.colorbar(ax[0].imshow(x), fraction=0.047*im_ratio)

        ax[1].imshow(y)
        ax[1].set_title("Ground_Truth")
        im_ratio = y.shape[1]/y.shape[0]
        plt.colorbar(ax[1].imshow(y), fraction=0.047*im_ratio)

        ax[2].imshow(model_predict_array)
        ax[2].set_title(model_name)
        im_ratio = model_predict_array.shape[1]/model_predict_array.shape[0]
        plt.colorbar(ax[2].imshow(model_predict_array),fraction=0.047*im_ratio)

        
        plt.tight_layout()
        
        plt.savefig(save_name)
        
        plt.show()


def PixelHistogram(model = torch.nn.Module,test_dataset = None ,inference_device="cpu",
                       index=0,save_name=None,model_name=None):
    """
    Function that visualizes the diffraction limited, ground truth super resolved, and the predicted super resolved images/
    respectively.

    Parameters
        ------------
        model : torch.nn.Module
            A trained SMLM model.

        test_dataset : DL4SMLM.ImageDataset object
            ImageDataset object from the DL4SMLM library 

        inference_device : str
            String containing the device where inference will occur.

        index : int
            Integer bound between 0 and length(test_dataset) - 1 inclusive. Designates which X,Y data pair will be visualized.

        save_name : str (optional)
            If the image is to be saved then provide a name to save the resulting image else the image will be displayed to the screen.

        model_name : str
            The name of the model that will be displayed over its infered image in the last column of the resulting plot.

    Outputs
        ------------
        1. A 1 x 3 image containing the diffraction limited, ground truth super resolved, and the predicted super resolved images/
        respectively. If a save name is provided then the image is written to a file compatible with the matplotlib.pylplot library.
    """
    
    x,y = test_dataset[index]
    
    x = x.unsqueeze(0)
    
    
    model.eval()
    
    with torch.inference_mode(True):
        
        model_predict = model(x)
        
        threshold = torch.nn.Threshold(0,0)
        
        model_predict_threshold = threshold(model_predict)
        
    model_predict_array = model_predict_threshold.squeeze().numpy()
    
    
    if save_name == None:
        plt.hist(model_predict_array.flatten())
        plt.ylabel('Frequency')
        plt.xlabel('Intensity')
        plt.tight_layout()
        plt.show()
    
    else:
       plt.hist(model_predict_array.flatten())
       plt.ylabel('Frequency')
       plt.xlabel('Intensity')
       plt.tight_layout()
       plt.savefig(save_name)

def InferImage(model = torch.nn.Module,image_stack_dir = torch.utils.data.DataLoader,
               inference_device="cpu", image_stack_mean=None, image_stack_std=None,
               ground_truth_image_path=None):

    """
    Form the super resolved image using the provided model to perform single molecule localization. The function will iterate 
    over all the images provided in the dataloader, infer the position of the single emitter, and then summate the result to developing
    final image.

    Parameters
        ------------
        model : torch.nn.Module
            A trained SMLM model.

        image_stack_dataloader : torch.utils.data.DataLoader
            String containing the path of the recorded image_stack

        inference_device : str
            String containing the path toward the super resolved emitter images.

    Outputs
        ------------
        Super Resolved Image : png
            A png of the inferred image
    """

    model.eval()

    image_file_names = os.listdir(image_stack_dir)

    _dummy_image = tifffile.imread(os.path.join(image_stack_dir,image_file_names[0]))

    H,W = _dummy_image.shape

    current_image = np.zeros((H,W))


    with torch.inference_mode():

        model.to(inference_device)

        for filename in image_file_names:

            whole_image_path = os.path.join(image_stack_dir,filename)

            image = tifffile.imread(whole_image_path).astype(np.float32)

            image_tensor = torch.from_numpy(image)

            normalized_image_tensor = (image_tensor -image_stack_mean) / image_stack_std

            normalized_image_tensor = normalized_image_tensor.unsqueeze(0)

            normalized_image_tensor = normalized_image_tensor.unsqueeze(0)
            
            X = normalized_image_tensor.to(inference_device)
            
            y_pred = model(X)

            thresh = torch.nn.Threshold(0,0)

            model_predict_threshold = thresh(y_pred)

            emitters = model_predict_threshold.squeeze().numpy()

            current_image = current_image + emitters
            
    
    return current_image
