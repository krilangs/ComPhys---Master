from ROOT import TColor
from ROOT import kBlack,kWhite,kGray,kRed,kPink,kMagenta,kViolet,kBlue,kAzure,kCyan,kTeal,kGreen,kSpring,kYellow,kOrange

#____________________________________________________________________________
def configure_samples():

  ### Colors

  red      = TColor.GetColor('#e6194B')
  orange   = TColor.GetColor('#f58231')  
  yellow   = TColor.GetColor('#ffe119')
  lime     = TColor.GetColor('#bfef45')
  green    = TColor.GetColor('#3cb44b')
  cyan     = TColor.GetColor('#42d4f4')
  blue     = TColor.GetColor('#4363d8')
  purple   = TColor.GetColor('#911eb4')
  magenta  = TColor.GetColor('#f032e6')
  pink     = TColor.GetColor('#fabebe') 
  apricot  = TColor.GetColor('#ffd8b1')
  beige    = TColor.GetColor('#fffac8')
  mint     = TColor.GetColor('#aaffc3')
  lavender = TColor.GetColor('#e6beff')
  teal     = TColor.GetColor('#469990')
  maroon   = TColor.GetColor('#800000')
  navy     = TColor.GetColor('#000075')
  brown    = TColor.GetColor('#9A6324')
  olive    = TColor.GetColor('#808000')
  grey     = TColor.GetColor('#a9a9a9')

  ### Sample path suffices
  data_suffix = '_SusySkimEWK3L_v2.3b_SUSY2orSUSY16_tree_NoSys.root'
  bkg_suffix  = '_SusySkimEWK3L_v2.3b_SUSY2orSUSY16_tree_fullJES.root'
  sig_suffix  = '_WZOffshell_SusySkimEWK3L_v2.3b_SUSY2orSUSY16_tree_fullJES.root'

  ### Dictionary of samples
  d_samp = {

    # Data
    'data'     :{'type':'data', 'leg':'Data',        'f_color':0,'l_color':0,  'path': 'data'+data_suffix},

    # Background
    'other'      :{'type':'bkg',  'leg':'other',       'f_color':kMagenta-10, 'path':'other'+bkg_suffix},
    'diboson2L'  :{'type':'bkg', 'path':'diboson2L'+bkg_suffix},
    'singletop'  :{'type':'bkg',  'path':'singletop'+bkg_suffix},
    'fakes'      :{'type':'bkg',  'leg':'fakes',       'f_color':kGray,       'path':'fakes'+bkg_suffix}, 
    'ttbar'      :{'type':'bkg',  'leg':'ttbar',       'f_color':90,          'path':'ttbar'+bkg_suffix},
    'ttbar+X'    :{'type':'bkg',  'leg':'ttbar+x',     'f_color':kViolet-8,          'path':'ttbar'+bkg_suffix},
    'topOther'   :{'type':'bkg',  'path':'topOther'+bkg_suffix},
    'higgs'      :{'type':'bkg',  'path':'higgs'+bkg_suffix},
    'triboson'   :{'type':'bkg',  'leg':'triboson',    'f_color':kOrange-3,  'path':'triboson'+bkg_suffix},
    'diboson3L'  :{'type':'bkg',  'leg':'WZ',          'f_color':kAzure+1,    'path':'diboson3L'+bkg_suffix},
    'diboson4L'  :{'type':'bkg',  'leg':'ZZ',          'f_color':kAzure-1,    'path':'diboson4L'+bkg_suffix},

    # Signal
    'MGPy8EG_A14N23LO_C1N2_WZ_100_10_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_100_20_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_100_30_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_100_40_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_100_60_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_100_80_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_100_90_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},

    'MGPy8EG_A14N23LO_C1N2_WZ_110_100_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_110_20_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_110_30_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_110_40_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_110_50_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_110_70_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_110_85_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_110_95_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},

    'MGPy8EG_A14N23LO_C1N2_WZ_125_100_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_125_110_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_125_115_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_125_35_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_125_45_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_125_55_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_125_65_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_125_85_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},

    'MGPy8EG_A14N23LO_C1N2_WZ_140_100_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_140_115_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_140_125_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_140_130_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_140_50_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_140_60_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_140_70_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_140_80_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},

    'MGPy8EG_A14N23LO_C1N2_WZ_150_100_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_110_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_120_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_130_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_140_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_60_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_70_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_80_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_90_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},

    'MGPy8EG_A14N23LO_C1N2_WZ_200_105_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_108_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_110_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_120_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_130_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_140_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_150_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_160_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_170_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_180_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_190_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},

    'MGPy8EG_A14N23LO_C1N2_WZ_250_155_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_250_158_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_250_160_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_250_170_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_250_190_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_250_200_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_250_210_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_250_220_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_250_230_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_250_240_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},

    'MGPy8EG_A14N23LO_C1N2_WZ_300_205_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_300_208_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_300_220_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_300_240_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_300_260_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_300_280_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_300_290_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},

    'MGPy8EG_A14N23LO_C1N2_WZ_350_270_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_350_290_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_350_310_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_350_330_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_350_340_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},

    'MGPy8EG_A14N23LO_C1N2_WZ_400_320_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_400_340_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_400_360_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_400_380_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_400_390_3L_3L3_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},


    'MGPy8EG_A14N23LO_C1N2_WZ_100_95_3L2MET75_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (100,95)','l_color':red,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_100_97_3L2MET75_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (100,97)','l_color':orange,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_110_105_3L2MET75_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_110_107_3L2MET75_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_125_120_3L2MET75_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_125_122_3L2MET75_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_145_3L2MET75_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_147_3L2MET75_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_175_170_3L2MET75_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_175_172_3L2MET75_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_195_3L2MET75_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (200,195)','l_color':purple,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_197_3L2MET75_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (100,197)','l_color':blue,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_90_85_3L2MET75_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_90_87_3L2MET75_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},


    'MGPy8EG_A14N23LO_C1N2_WZ_100_0_3L_2L7_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_1_3L_2L7_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_50_3L_2L7_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_100_3L_2L7_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_50_3L_2L7_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_250_150_3L_2L7_NoSys':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,'path':'AllSignals'+sig_suffix},
  }

  return d_samp
