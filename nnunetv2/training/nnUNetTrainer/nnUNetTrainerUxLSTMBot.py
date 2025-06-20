from typing import Union
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UxLSTMBot_3d import get_uxlstm_bot_3d_from_plans
from nnunetv2.nets.UxLSTMBot_2d import get_uxlstm_bot_2d_from_plans


class nnUNetTrainerUxLSTMBot(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_uxlstm_bot_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_uxlstm_bot_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")
        
        print("UxLSTMBot: {}".format(model))

        return model


    # @staticmethod
    # def build_network_architecture(architecture_class_name: str,
    #                                 arch_init_kwargs: dict,
    #                                 arch_init_kwargs_req_import:Union[List[str], Tuple[str, ...]], 
    #                                 num_input_channels: int,
    #                                 num_output_channels: int,
    #                                 enable_deep_supervision: bool = True) -> nn.Module:
        
    #     plans_manager = PlansManager(dict)
    #     configuration_manager = ConfigurationManager(dict)
    #     num_input_channels = 1
    #     enable_deep_supervision = True

    #     model = get_uxlstm_bot_2d_from_plans(plans_manager, dataset_json, configuration_manager,
    #                                       num_input_channels, deep_supervision=enable_deep_supervision)
        
    #     return model
