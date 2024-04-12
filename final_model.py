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
import itertools
import torch
import torch.nn as nn
from collections import OrderedDict


from basic_blocks import *
from generator import *
from discriminator import *
from eval_metric import *

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

# I removed baseModel class
class Final_Model(nn.Module):

    # Maybe move this to options?
    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--netG', type=str, default='pose', help='The name of net Generator')
        parser.add_argument('--netD', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        # parser.add_argument('--lambda_correct', type=float, default=5.0, help='weight for the Sampling Correctness loss')
        parser.add_argument('--lambda_style', type=float, default=200.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')
        parser.add_argument('--lambda_regularization', type=float, default=30.0,
                            help='weight for the affine regularization loss')

        parser.add_argument('--use_spect_g', action='store_false',
                            help="whether use spectral normalization in generator")
        parser.add_argument('--use_spect_d', action='store_false',
                            help="whether use spectral normalization in discriminator")
        parser.add_argument('--save_input', action='store_false', help="whether save the input images when testing")

        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        parser.set_defaults(save_input=False)

        return parser

    def __init__(self, opt):
        super(Final_Model, self).__init__()

        self.opt = opt
        self.device = opt.device

        self.generator = Generator().to(self.device)

        # define the discriminator
        # if self.opt.dataset_mode == 'fashion':
        self.discriminator = Discriminator(ndf=32, img_f=128, layers=4).to(self.device)

        # define the loss functions
        self.GAN_loss = loss.Adversarial_Loss().to(self.device)
        self.L1_loss = torch.nn.L1Loss()
        self.VGG_loss = loss.VGG_Loss().to(opt.device)
        self.cross_entropy_2d_loss = loss.Cross_Entropy_Loss2d()

        if opt.isTrain is False:
            self.eval()
            self.generator.eval()
            print("Is in eval mode")
            return None
        # Optimizer for the generator
        self.optimizer_G = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.generator.parameters()),
            lr=opt.lr, betas=(0.9, 0.999)
        )
        # self.scheduler_G = lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda epoch: max(0, 1.0 - (
        #         epoch + 1 + opt.iter_count - opt.niter) / float(opt.niter_decay + 1)))

        # Optimizer for the discriminator
        self.optimizer_D = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.discriminator.parameters()),
            lr=opt.lr * opt.ratio_g2d, betas=(0.9, 0.999)
        )

    def save_networks(self, epoch, iteration):
        """
            Save all the networks to the disk.
        """
        save_filename = f"{epoch}_{iteration}_generator.pth"
        save_path = os.path.join(self.opt.save_dir, save_filename)

        torch.save(self.generator.cpu().state_dict(), save_path)
        self.generator.to(self.device)

        save_filename = f"{epoch}_{iteration}_discriminator.pth"
        save_path = os.path.join(self.opt.save_dir, save_filename)

        torch.save(self.discriminator.cpu().state_dict(), save_path)
        self.discriminator.to(self.device)

    def save_results(self, save_data, iteration, epoch, results_dir="fashion_data/eval_results", data_name='none', data_ext='jpg'):
        """
            Save the training or testing results to disk.
        """
        # print("----------------", result_psnr)

        for i, img_path in enumerate(self.image_paths):
            print(f'Processing image: {img_path}')
            # Extract the base name without the directory path and extension
            base_name = os.path.splitext(ntpath.basename(img_path))[0]
            img_name = f"{epoch}:{iteration}_{base_name}_{data_name}_{i}.{data_ext}"

            # Construct the full path for saving the image
            full_img_path = os.path.join(results_dir, img_name)

            # Convert the tensor to a numpy image and save
            img_numpy = tensor2im(save_data[i])
            save_image(img_numpy, full_img_path)


    def load_networks(self, generator_path, discriminator_path):
        """
            Load all the networks from the disk.
        """
        self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device))


    def print_gradients(self, model,epoch , iteration, modelname='generator'):
        data = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                data[name] = parameter.grad

            #     print(f"Gradients of {name}: {parameter.grad}")
            # else:
            #     print(f"{name} has no gradients")
        self.save_dictionary(f"logs/grads{modelname}{epoch}:{iteration}.txt", data)

    def save_debug_files(self, path, epoch, iteration):
        self.save_dictionary(os.path.join(path, f'parsing_generator_debug_epoch_{epoch}_iteration_{iteration}.txt'), self.generator.parsing_generator.debugger)
        self.save_dictionary(os.path.join(path, f'encoder_3_debug_epoch_{epoch}_iteration_{iteration}.txt'), self.generator.encoder_3.debugger)
        self.save_dictionary(os.path.join(path, f'per_region_encoding_debug_epoch_{epoch}_iteration_{iteration}.txt'), self.generator.per_region_encoding.debugger)
        self.save_dictionary(os.path.join(path, f'enoder_2_debug_epoch_{epoch}_iteration_{iteration}.txt'), self.generator.encoder_2.debugger)
        self.save_dictionary(os.path.join(path, f'decoder_2_debug_epoch_{epoch}_iteration_{iteration}.txt'), self.generator.decoder_2.debugger)
        self.save_dictionary(os.path.join(path, f'per_region_normalization_debug_epoch_{epoch}_iteration_{iteration}.txt'), self.generator.per_region_normalization.debugger)
        self.save_dictionary(os.path.join(path, f'Generator_debug_epoch_{epoch}_iteration_{iteration}.txt'), self.generator.debugger)
        self.save_dictionary(os.path.join(path, f'Discriminator_debug_epoch_{epoch}_iteration_{iteration}.txt'), self.discriminator.debugger)

    def clear_or_create_directory(self, dir_path):
        """
        Deletes all files and folders inside the given directory recursively,
        and creates the directory if it does not exist.

        :param dir_path: Path to the directory to clear or create.
        """
        # Check if the directory exists
        if os.path.exists(dir_path):
            # Delete all files and subdirectories in the directory
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                # Check if it's a file or a directory
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or link
        else:
            # Create the directory if it does not exist
            os.makedirs(dir_path)
            print(f"Directory '{dir_path}' was created.")

    def save_dictionary(self, file_path, my_dict):
        # Writing the dictionary to a file
        with open(file_path, 'w') as file:
            for key, value in my_dict.items():
                file.write(f'{key}:\n {value}\n-----------------------------------\n')

    def test2(self, iteration, epoch,  subset=20):
        torch.set_printoptions(threshold=10_000)
        # self.print_gradients(self.generator, epoch, iteration, modelname='generator')
        # self.print_gradients(self.discriminator, epoch, iteration, modelname='discriminator')
        generated_img, _, _ = self.generator(
            self.input_P1[:subset, :, :, :], self.input_P2[:subset, :, :, :],
            self.input_BP1[:subset, :, :, :], self.input_BP2[:subset, :, :, :],
            self.input_SPL1[:subset, :, :, :], self.input_SPL2[:subset, :, :, :], debug=True
        )
        # self.save_debug_files(f"logs", epoch, iteration)
        # self.generator.train()
        result = torch.cat([self.input_P1[:subset, :, :, :], generated_img, self.input_P2[:subset, :, :, :]], dim=3)
        result_psnr = psnr(self.input_P2[:subset, :, :, :], generated_img)
        lpips_result = calc_lpips(self.input_P2[:subset, :, :, :], generated_img, self.opt)
        # fid_result = calc_fid(self.input_P2[:subset, :, :, :], generated_img)
        print(f'Whole batch: PSNR: {result_psnr} - LPIPS: {lpips_result[[0, 0, 0]]}')
        self.save_results(result, iteration, epoch, data_name='all')


        # result = torch.cat([self.input_BP1[:subset, :, :, :], self.input_SPL1[:subset, :, :, :]], dim=3)
        # self.save_results(result, iteration, epoch, data_name='all')


    def test_phase(self,index,  subset=32):
        generated_img, _, _ = self.generator(
            self.input_P1[:subset, :, :, :], self.input_P2[:subset, :, :, :],
            self.input_BP1[:subset, :, :, :], self.input_BP2[:subset, :, :, :],
            self.input_SPL1[:subset, :, :, :], self.input_SPL2[:subset, :, :, :], debug=True
        )
        # result = torch.cat([self.input_P1[:subset, :, :, :], generated_img, self.input_P2[:subset, :, :, :]], dim=3)
        result_psnr = psnr(self.input_P2[:subset, :, :, :], generated_img)
        lpips_result = calc_lpips(self.input_P2[:subset, :, :, :], generated_img, self.opt)
        # fid_result = calc_fid(self.input_P2[:subset, :, :, :], generated_img)
        print(f'Whole batch: PSNR: {result_psnr} - LPIPS: {lpips_result[[0, 0, 0]]}')
        # self.save_results(result, "", "Test", data_name='all', results_dir='fashion_data/eval_test_results')
        # self.save_results(generated_img, index, "Test", data_name='ref', results_dir='fashion_data/test_output')
        return generated_img, result_psnr, lpips_result



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

    def get_loss_results(self):
        return {"D_loss": self.D_loss,
                "loss_content_gen": self.loss_content_gen,
                "loss_style_gen": self.loss_style_gen,
                "L_L1": self.L_l1,
                "loss_adv": self.loss_adv,
                "parsing_gen_cross": self.parsing_gen_cross,
                "parsing_gen_l1": self.parsing_gen_l1
                }

    def forward(self):
        """Run forward processing to get the inputs"""
        self.generated_img, self.loss_reg, self.parsav = self.generator(self.input_P1, self.input_P2, self.input_BP1,
                                                              self.input_BP2, self.input_SPL1, self.input_SPL2)

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


    def backward_G(self):
        total_loss = 0

        # Parsing Generator two losses ---------------------------
        label_P2 = self.label_P2.squeeze(1).long()
        self.parsing_gen_cross = self.cross_entropy_2d_loss(self.parsav, label_P2)

        self.parsing_gen_l1 = self.L1_loss(self.parsav,
                                           self.input_SPL2) * 100  # l1 distance loss between the generated parsing map and the target_image parsing map

        total_loss += self.parsing_gen_cross + self.parsing_gen_l1


        # Image generator losses ---------------------------------
        # Regularization loss
        self.L_cor = self.loss_reg * self.opt.lambda_regularization # self.loss_reg is an output of the generator - Lcor
        total_loss += self.L_cor

        # L1 loss (Appearance loss)
        self.L_l1 = self.L1_loss(self.generated_img, self.input_P2) * self.opt.lambda_rec # Ll1 - self.generated_img is an output of the generator
        total_loss += self.L_l1



        # Freeze the discriminator for the generator's backward pass
        for param in self.discriminator.parameters():
            param.requires_grad = False

        # GAN loss (Adversarial loss)

        D_fake = self.discriminator(self.generated_img)
        self.loss_adv = self.GAN_loss(D_fake, True) * self.opt.lambda_g
        total_loss += self.loss_adv

        for param in self.discriminator.parameters():
            param.requires_grad = True

        # total_loss.backward()
        # return

        # Perceptual loss (Content and Style)
        self.loss_content_gen, self.loss_style_gen = self.VGG_loss(self.generated_img, self.input_P2)
        self.loss_style_gen *= self.opt.lambda_style
        self.loss_content_gen *= self.opt.lambda_content
        total_loss += (self.loss_content_gen + self.loss_style_gen)


        # Backpropagate the total loss
        total_loss.backward()

    def optimize_parameters(self):
        """update network weights"""
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D(self.input_P2, self.generated_img)
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
