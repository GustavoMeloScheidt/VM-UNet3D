from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.vmunet3d import VMUNet_3D
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler    
import torch    

class nnUNetTrainerVMUNet3D(nnUNetTrainer):
    
    #Modif by Gustavo Scheidt 
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_epochs = 400

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), 
                                           self.initial_lr, 
                                           weight_decay=self.weight_decay,
                                           eps=1e-4)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
    #End of Modif 
    
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module: #Add the deep supervision in the future 
        print("We're using the class VM_UNet_3D")
        # Ignorando os argumentos de arquitetura do nnU-Net (plans.json) e para usar o modelo personalizado
        return VMUNet_3D(
            input_channels=num_input_channels,
            num_classes=plans_manager.num_output_channels,
            depths=[2, 2, 9, 2],
            depths_decoder=[2, 9, 2, 2],
            drop_path_rate=0.2,
            #deep_supervision=enable_deep_supervision 
        )
    