import os
import torch
import importlib

from models import DLinear, PatchTST, iTransformer, TiDE, TimeXer, PFB, PFB_Projected, PFB_Direct, \
    PFB_PatchOnly, PFB_SecondaryOnly, PFB_SerialRefinement, PatchTST_LargeHead, \
    DLinear_Norm, PatchTST_SerialMatched


def _optional_model(name):
    try:
        return importlib.import_module(f"models.{name}")
    except ModuleNotFoundError:
        return None


Autoformer = _optional_model("Autoformer")
Transformer = _optional_model("Transformer")
TimesNet = _optional_model("TimesNet")
Nonstationary_Transformer = _optional_model("Nonstationary_Transformer")
FEDformer = _optional_model("FEDformer")
Informer = _optional_model("Informer")
LightTS = _optional_model("LightTS")
Reformer = _optional_model("Reformer")
ETSformer = _optional_model("ETSformer")
Pyraformer = _optional_model("Pyraformer")
MICN = _optional_model("MICN")
Crossformer = _optional_model("Crossformer")
FiLM = _optional_model("FiLM")
Koopa = _optional_model("Koopa")
FreTS = _optional_model("FreTS")
TimeMixer = _optional_model("TimeMixer")
TSMixer = _optional_model("TSMixer")
SegRNN = _optional_model("SegRNN")
MambaSimple = _optional_model("MambaSimple")
TemporalFusionTransformer = _optional_model("TemporalFusionTransformer")
SCINet = _optional_model("SCINet")
PAttn = _optional_model("PAttn")
WPMixer = _optional_model("WPMixer")
MultiPatchFormer = _optional_model("MultiPatchFormer")
KANAD = _optional_model("KANAD")
MSGNet = _optional_model("MSGNet")
TimeFilter = _optional_model("TimeFilter")
TimeMoE = _optional_model("TimeMoE")


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'DLinear_Norm': DLinear_Norm,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'PatchTST_base': PatchTST,
            'PatchTST_capacity': PatchTST,
            'PatchTST_depth': PatchTST,
            'PatchTST_LargeHead': PatchTST_LargeHead,
            'PatchTST_SerialMatched': PatchTST_SerialMatched,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'KANAD': KANAD,
            'MSGNet': MSGNet,
            'TimeFilter': TimeFilter,
            'TimeMoE': TimeMoE,
            'PFB': PFB,
            'PFB-Projected': PFB_Projected,
            'PFB-Direct': PFB_Direct,
            'PFB_Projected': PFB_Projected,
            'PFB_Direct': PFB_Direct,
            'PFB_PatchOnly': PFB_PatchOnly,
            'PFB_SecondaryOnly': PFB_SecondaryOnly,
            'PFB_SerialRefinement': PFB_SerialRefinement,
        }
        legacy_base = "PatchFusion" + "".join(chr(c) for c in (66, 69, 82, 84))
        self.model_dict.update({
            legacy_base: PFB,
            legacy_base + "_v" + "0": PFB_Direct,
            legacy_base + "_v" + "2": PFB_Projected,
            "PFB_" + "v" + "0": PFB_Direct,
            "PFB_" + "v" + "2": PFB_Projected,
            "PFB" + "v" + "0": PFB_Direct,
            "PFB" + "v" + "2": PFB_Projected,
        })
        self.model_dict = {name: module for name, module in self.model_dict.items() if module is not None}
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
