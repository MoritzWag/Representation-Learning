from library.models2.vae_gaussian2 import *
from library.models2.vae_info2 import *
from library.models2.cat_vae import *
from library.models2.base2 import *
from library.architectures import *
from library.models2.vae_beta import *
from library.models2.autoencoder import *
from library.models2.vae_gaussmix import *
from library.models2.vae_joint import *
from library.models2.vae_dip import *
from library.models2.lin_autoencoder import * 
import pdb 

base_models = {'VaeBase': VaeBase,
            'MMVaeBase': MMVaeBase,
            'AutoencoderBase': AutoencoderBase}

vae_models = {'GaussianVae': VaeGaussian,
            'BetaVae': BetaVae,
            'InfoVae': InfoVae,
            'CatVae': CatVae, 
            'Autoencoder': Autoencoder,
            'GaussmixVae': GaussmixVae,
            'JointVae': JointVae,
            'DIPVae': DIPVae,
            'LinearAutoencoder': LinearAutoencoder}

vae_architectures = {'ConvEncoder28x28': ConvEncoder28x28,
                'ConvDecoder28x28': ConvDecoder28x28,
                'ConvEncoder64x64': ConvEncoder64x64,
                'ConvDecoder64x64': ConvDecoder64x64,
                'ConvEncoder224x224': ConvEncoder224x224,
                'ConvDecoder224x224': ConvDecoder224x224,
                'AttrEncoder': AttributeEncoder,
                'AttrDecoder': AttributeDecoder,
                'expert': ProductOfExperts,
                'CustomizedResNet101': CustomizedResNet101,
                'ConvEncoder': ConvEncoder,
                'ConvDecoder': ConvDecoder,
                'LinearEncoder': LinearEncoder,
                'LinearDecoder': LinearDecoder}

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

def parse_model_config(config, trial=None):
    model_params = config.get('model_params')
    try:
        hyper_params = config.get('model_hyperparams')
    except:
        pass
    vae_instance = vae_models[model_params['name']]
    base_instance = base_models[model_params['base']]
    vae_instance.__bases__ = (base_instance, )
    model_dict = parse_architecture_config(config)
    if hyper_params is not None:
        model_dict.update(hyper_params)
    model = vae_instance(trial=trial, **model_dict)
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


def update_config(config, args):
    """
    """
    for name, value in vars(args).items():
        if value is None:
            continue
        
        for key in config.keys():
            if config[key].__contains__(name):
                config[key][name] = value
    
    return config

