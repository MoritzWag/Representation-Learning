from library.models2.vae_gaussian2 import *
from library.models2.base2 import *
from library.architectures import *
import pdb 

base_models = {'VaeBase': VaeBase,
            'MMVaeBase': MMVaeBase}

vae_models = {'GaussianVae': VaeGaussian}


vae_architectures = {'ConvEncoder28x28': ConvEncoder28x28,
                'ConvDecoder28x28': ConvDecoder28x28,
                'AttrEncoder': AttributeEncoder,
                'AttrDecoder': AttributeDecoder,
                'expert': ProductOfExperts}


def createClass(vae_model, base_model):
    #class vae_model(base_model): pass
    model = type('vae_model' , (vae_model, base_model), {})

    return model

def makeWithMixins(cls, mixins):
    for mixin in mixins:
        if mixin not in cls.__bases__:
            cls.__bases__ = (mixin, ) + cls.__bases__
        else:
            print("whatsoever")
    return cls

def parse_model_config(config):
    model_params = config.get('model_params')
    vae_instance = vae_models[model_params['name']]
    base_instance = base_models[model_params['base']]
    vae_instance.__bases__ = (base_instance, )
    model_dict = parse_architecture_config(config)
    model = vae_instance(**model_dict)
    return model

def parse_architecture_config(config):
    architecture = config.get('architecture')
    img_arch_params = config.get('img_arch_params')
    text_arch_params = config.get('text_arch_params')

    arch_dict = {'img_encoder': vae_architectures.get(architecture.get('img_encoder', None), None), 
                'img_decoder': vae_architectures.get(architecture.get('img_decoder', None), None),
                'text_encoder': vae_architectures.get(architecture.get('text_encoder', None), None),
                'text_decoder': vae_architectures.get(architecture.get('text_decoder', None), None),
                'expert': vae_architectures.get(architecture.get('expert', None), None)}
    
    model_dict = {}
    for instance in arch_dict.items():
        if instance[1] is not None:
            try: 
                model_dict[instance[0]] = instance[1](**img_arch_params)
            except:
                try:
                    model_dict[instance[0]] = instance[1](**text_arch_params)
                except: 
                    model_dict[instance[0]] = instance[1]()
    return model_dict
