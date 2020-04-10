#from library.models2 import vae_gaussian2, 
from library.models2.vae_gaussian2 import *
from library.architectures import *
import pdb 

vae_models = {'GaussianVae': VaeGaussian}


vae_architectures = {'ConvEncoder28x28': ConvEncoder28x28,
                'ConvDecoder28x28': ConvDecoder28x28}




def parse_model_config(config):

    model_params = config.get('model_params')
    model_instance = vae_models[model_params['name']]
    model_dict = parse_architecture_config(config)
    
    model = model_instance(**model_dict)

    return model


def parse_architecture_config(config):
    architecture = config.get('architecture')
    arch_params = config.get('architecture_params')
    
    arch_dict = {'img_encoder': vae_architectures.get(architecture.get('img_encoder', None), None), 
                'img_decoder': vae_architectures.get(architecture.get('img_decoder', None), None),
                'attr_encoder': vae_architectures.get(architecture.get('attr_encoder', None), None),
                'attr_decoder': vae_architectures.get(architecture.get('attr_decoder', None), None)}
    
    model_dict = {}
    for instance in arch_dict.items():
        if instance[1] is not None:
            model_dict[instance[0]] = instance[1](**arch_params)

    return model_dict
