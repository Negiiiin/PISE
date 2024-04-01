import torch.nn as nn
import torch
from torch.nn.utils.spectral_norm import spectral_norm
import torchvision
import os
import torch.nn.functional as F
import loss
import numpy as np
from PIL import Image
import ntpath
from torch.nn import init
import warnings
from torch.optim import lr_scheduler

from VGG19 import VGG19

from basic_blocks import *
from generator import *
from discriminator import *

class Final_Model(nn.Module):
    def __init__(self, opt):
        super(Final_Model, self).__init__()
        self.device = opt.device
        self.save_dir = opt.checkpoints_dir

        # Define the generator
        self.generator = Generator()
        self.discriminator = Discriminator(ndf=32, img_f=128, layers=4)

        # Initialize loss functions and optimizers if training
        if opt.isTrain:
            self.init_losses_and_optimizers(opt)

        if opt.continue_train:
            print('Loading weights from iteration %s' % opt.which_iter)
            self.load_networks(self.which_iter)

    def init_weights(self, gain=0.02):
        # Iterate over all modules in the model
        for m in self.modules():
            class_name = m.__class__.__name__
            # Initialize weights for BatchNorm2d layers
            if class_name.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            # Initialize weights for Conv and Linear layers using orthogonal initialization
            elif hasattr(m, 'weight') and (class_name.find('Conv') != -1 or class_name.find('Linear') != -1):
                init.kaiming_normal_(m.weight.data)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def init_losses_and_optimizers(self, opt):
        self.GAN_loss = loss.Adversarial_Loss().to(self.device)
        self.L1_loss = nn.L1Loss()
        self.VGG_loss = loss.VGG_Loss().to(self.device)
        self.cross_entropy_2d_loss = loss.Cross_Entropy_Loss2d()

        # Optimizer for the generator
        self.optimizer_G = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.generator.parameters()),
            lr=opt.lr, betas=(0.9, 0.999)
        )
        self.scheduler_G = lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda epoch: max(0, 1.0 - (
                epoch + 1 + opt.iter_count - opt.niter) / float(opt.niter_decay + 1)))

        # Optimizer for the discriminator
        self.optimizer_D = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.discriminator.parameters()),
            lr=opt.lr * opt.ratio_g2d, betas=(0.9, 0.999)
        )

        self.scheduler_D = lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lambda epoch: max(0, 1.0 - (
                    epoch + 1 + opt.iter_count - opt.niter) / float(opt.niter_decay + 1)))

    def update_lr(self):
        """Update learning rate."""
        self.scheduler_D.step()
        self.scheduler_G.step()

        lr = self.optimizer_D.param_groups[0]['lr']

        print(f"Current Learning Rate = {lr:.7f}")

    def set_input(self, input):
        self.input = input
        self.image_paths = []

        # Define keys for easier readability and modification
        keys = ['P1', 'BP1', 'SPL1', 'P2', 'BP2', 'SPL2']
        # Automatically transfer all tensors to the specified device
        for key in keys:
            if key in input:
                setattr(self, 'input_' + key, input[key].to(self.device))

        self.label_P2 = input['label_P2'].to(self.device)

        # Handle image paths separately as they are not tensors
        for i in range(input['P1'].size(0)):  # Assuming 'P1' exists and has a batch dimension
            path = '_'.join([os.path.splitext(input[path_key][i])[0] for path_key in ['P1_path', 'P2_path']])
            self.image_paths.append(path)

    def forward(self):
        self.generated_img, self.loss_reg, self.parsav = self.generator(self.input_P1, self.input_P2, self.input_BP1, self.input_BP2, self.input_SPL1, self.input_SPL2) #TODO

    def backward_D(self, real_input, fake_input, unfreeze_netD=True):
        """Calculate the GAN loss for the discriminator, including handling WGAN-GP if specified."""
        if unfreeze_netD:
            # Unfreeze the discriminator if it was frozen
            for param in self.discriminator.parameters():
                param.requires_grad = True

        # Real input loss
        D_real = self.discriminator(real_input)
        D_real_loss = self.GAN_loss(D_real, True)

        # Fake input loss (detach to avoid backproping into the generator)
        D_fake = self.discriminator(fake_input.detach())
        D_fake_loss = self.GAN_loss(D_fake, False)

        # Combined discriminator loss
        self.D_loss = (D_real_loss + D_fake_loss) * 0.5

        # Backpropagate the discriminator loss
        self.D_loss.backward()

        return self.D_loss

    def backward_G(self, lambda_regularization=30.0, lambda_rec=5.0, lambda_pl=100.0, lambda_a=2.0, lambda_style=200.0, lambda_content=0.5):
        """Calculate training loss for the generator."""
        # Initialize total loss
        total_loss = 0

        # Parsing Generator two losses ---------------------------
        label_P2 = self.label_P2.squeeze(1).long()
        self.parsing_gen_cross = self.cross_entropy_2d_loss(self.parsav, label_P2)
        self.parsing_gen_l1 = self.L1_loss(self.parsav,
                                           self.input_SPL2) * lambda_pl  # l1 distance loss between the generated parsing map and the target parsing map
        total_loss += self.parsing_gen_cross + self.parsing_gen_l1

        # Image generator losses ---------------------------------
        # Regularization loss
        self.L_cor = self.loss_reg * lambda_regularization # self.loss_reg is an output of the generator - Lcor TODO
        total_loss += self.L_cor

        # L1 loss (Appearance loss)
        self.L_l1 = self.L1_loss(self.generated_img, self.input_P2) * lambda_rec # Ll1 - self.generated_img is an output of the generator - TODO
        total_loss += self.L_l1

        # Freeze the discriminator for the generator's backward pass
        for param in self.discriminator.parameters():
            param.requires_grad = False

        # GAN loss (Adversarial loss)
        D_fake = self.discriminator(self.generated_img) #TODO
        self.loss_adv = self.GAN_loss(D_fake, True) * lambda_a
        total_loss += self.loss_adv

        # Perceptual loss (Content and Style)
        self.loss_content_gen, self.loss_style_gen = self.VGG_loss(self.generated_img, self.input_P2) #TODO
        self.loss_style_gen *= lambda_style
        self.loss_content_gen *= lambda_content
        total_loss += self.loss_content_gen + self.loss_style_gen

        # Backpropagate the total loss
        total_loss.backward()

    def optimize_parameters(self):
        """
            Update network weights by performing a forward pass, then optimizing the discriminator and generator.
        """
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D(self.input_P2, self.generated_img)
        self.optimizer_D.step()

        # Optimize the generator
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def save_results(self, save_data, results_dir="eval_results", data_name='none', data_ext='jpg'):
        """
            Save the training or testing results to disk.
        """
        for i, img_path in enumerate(self.image_paths):
            print(f'Processing image: {img_path}')
            # Extract the base name without the directory path and extension
            base_name = os.path.splitext(ntpath.basename(img_path))[0]
            img_name = f"{base_name}_{data_name}_{i}.{data_ext}"

            # Construct the full path for saving the image
            full_img_path = os.path.join(results_dir, img_name)

            # Convert the tensor to a numpy image and save
            img_numpy = tensor2im(save_data[i])
            save_image(img_numpy, full_img_path)

    def test(self, subset=20):
        img_gen, _, _ = self.generator(
            self.input_P1[:subset, :, :, :], self.input_P2[:subset, :, :, :],
            self.input_BP1[:subset, :, :, :], self.input_BP2[:subset, :, :, :],
            self.input_SPL1[:subset, :, :, :], self.input_SPL2[:subset, :, :, :]
        )

        result = torch.cat([self.input_P1[:subset, :, :, :], img_gen, self.input_P2[:subset, :, :, :]], dim=3)
        self.save_results(result, data_name='all')

    def save_networks(self, epoch):
        """
            Save all the networks to the disk.
        """
        save_filename = f"{epoch}_generator.pth"
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(self.generator.cpu().state_dict(), save_path)
        self.generator.to(self.device)

        save_filename = f"{epoch}_discriminator.pth"
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(self.discriminator.cpu().state_dict(), save_path)
        self.discriminator.to(self.device)

    def load_networks(self, epoch): # Check if we can give which epoch to opt TODO
        """
            Load all the networks from the disk.
        """
        for name, net in zip(["generator", "discriminator"], [self.generator, self.discriminator]):
            filename = f"{epoch}_{name}.pth"
            path = os.path.join(self.save_dir, filename)

            if not os.path.isfile(path):
                warnings.warn(f"Checkpoint not found for network {name} at {path}", RuntimeWarning)
                continue

            state_dict = torch.load(path, map_location=self.device)
            model_dict = net.state_dict()

            # Filter out unnecessary keys
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            # Update current model state dict
            model_dict.update(state_dict)
            net.load_state_dict(model_dict)

            print(f"Loaded {name} from {filename}")

            if not self.is_train:
                net.eval()

    def get_loss_results(self):
        return {"D_loss": self.D_loss,
                "loss_content_gen": self.loss_content_gen,
                "loss_style_gen": self.loss_style_gen,
                "loss_adv": self.loss_adv,
                "L_l1": self.L_l1,
                "parsing_gen_cross": self.parsing_gen_cross,
                "parsing_gen_l1": self.parsing_gen_l1}


# From resource code -----------------------------------------------------------------------

def tensor2im(image_tensor, bytes=255.0, need_dec=False, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().detach().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().detach().numpy()

    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if need_dec:
        image_numpy = decode_labels(image_numpy.astype(int))
    else:
        image_numpy = (image_numpy + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)

# label color
label_colours = [(0, 0, 0)
    , (128, 0, 0), (255, 0, 0), (0, 85, 0), (170, 0, 51), (255, 85, 0), (0, 0, 85), (0, 119, 221), (85, 85, 0),
                 (0, 85, 85), (85, 51, 0), (52, 86, 128), (0, 128, 0)
    , (0, 0, 255), (51, 170, 221), (0, 255, 255), (85, 255, 170), (170, 255, 85), (255, 255, 0), (255, 170, 0)]

def decode_labels(mask, num_images=1, num_classes=20):
    """
        Decode batch of segmentation masks.

        Args:
          mask: result of inference after taking argmax.
          num_images: number of images to decode from the batch.
          num_classes: number of classes to predict (including background).

        Returns:
          A batch with num_images RGB images of the same size as the input.
    """
    h, w, c = mask.shape
    # assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((h, w, 3), dtype=np.uint8)

    img = Image.new('RGB', (len(mask[0]), len(mask)))
    pixels = img.load()
    tmp = []
    tmp1 = []
    for j_, j in enumerate(mask[:, :, 0]):
        for k_, k in enumerate(j):
            # tmp1.append(k)
            # tmp.append(k)
            if k < num_classes:
                pixels[k_, j_] = label_colours[k]
    # np.save('tmp1.npy', tmp1)
    # np.save('tmp.npy',tmp)
    outputs = np.array(img)
    # print(outputs[144,:,0])
    return outputs

def save_image(image_numpy, image_path):
    # Handle grayscale images (with a single channel)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.squeeze(axis=2)  # Remove the channel dimension

    # Handle images with 8 channels by converting them to a label map
    elif image_numpy.shape[2] == 8:
        image_numpy = np.argmax(image_numpy, axis=2)  # Convert channel dimension to label map
        image_numpy = decode_labels(image_numpy)  # Assume decode_labels returns an RGB image

    # Save the image
    image = Image.fromarray(image_numpy)
    image.save(image_path)

# From resource code -----------------------------------------------------------------------