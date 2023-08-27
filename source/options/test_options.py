###############################################################################
# This file contains Train Options (arguments, input params) should be defined 
# by user.
###############################################################################

from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        # Data loader argument
        parser.add_argument('--serial_batches', action='store_true', help='shuffle options: if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--validDataset_split', type=float, default=0.0, help='ratio for splitting valid dataset from main dataset')
        
        # Data transform argument
        parser.add_argument('--resize_or_crop', type=str, default='resize', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip', action='store_true', default=True, help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--use_skinmask', action='store_true', default=False, help='if specified, use YCrCb or RGB')
        
        # Model training configuration
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        
        self.isTrain = False
        return parser
