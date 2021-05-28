from ROOT import TColor
from ROOT import kBlack,kWhite,kGray,kRed,kPink,kMagenta,kViolet,kBlue,kAzure,kCyan,kTeal,kGreen,kSpring,kYellow,kOrange

#____________________________________________________________________________
def configure_samples(isData15_16, isData17, isData18, isData15_18, Signal):

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
  # 3L  ntuples
  if isData15_16:
    vers = "1516"
  elif isData17:
    vers = "17"
  elif isData18:
    vers = "18"

  # Classified ntuples
  data_suffix = "_classified.root" # Not used 
  bkg_suffix = "_" + Signal + "_classified.root"
  sig_suffix = "_" + Signal + "_classified.root"

  # 3L ntuples before classification
  #data_suffix = vers + '_merged_processed.root'
  #bkg_suffix = '_merged_processed.root'
  #sig_suffix = '_merged_processed.root'

  
  ### Dictionary of samples
  d_samp = {

    # Data
    'data'     :{'type':'data', 'leg':'Data',        'f_color':0,'l_color':0,  'path': 'data'+data_suffix},

    # Background
   # 'other'      :{'type':'bkg',  'leg':'other',       'f_color':kMagenta-10, 'path':'other'+bkg_suffix},
    'diboson2L'  :{'type':'bkg',  'leg':'diboson2L',   'f_color':kMagenta-10,  'path':'diboson2L'+bkg_suffix},
    'singletop'  :{'type':'bkg',  'leg':'singletop',   'f_color':kMagenta+10,  'path':'singletop'+bkg_suffix},
    #'fakes'      :{'type':'bkg',  'leg':'fakes',       'f_color':kGray,        'path':'fakes'+bkg_suffix}, 
    'ttbar'      :{'type':'bkg',  'leg':'ttbar',       'f_color':90,           'path':'ttbar'+bkg_suffix},
    'ttbar+X'    :{'type':'bkg',  'leg':'ttbar+x',     'f_color':kViolet-8,    'path':'ttbar'+bkg_suffix},
    'topOther'   :{'type':'bkg',  'path':'topOther'+bkg_suffix},
    'higgs'      :{'type':'bkg',  'path':'higgs'+bkg_suffix},
    'triboson'   :{'type':'bkg',  'leg':'triboson',    'f_color':kOrange-3,    'path':'triboson'+bkg_suffix},
    'diboson3L'  :{'type':'bkg',  'leg':'WZ',          'f_color':kAzure+1,     'path':'diboson3L'+bkg_suffix},
    'diboson4L'  :{'type':'bkg',  'leg':'ZZ',          'f_color':kAzure-1,     'path':'diboson4L'+bkg_suffix},
    'Zjets'      :{'type':'bkg',  'leg':'Z+jets',      'f_color':grey,       'path':'Zjets'+bkg_suffix },


    # Signal
    'LFCMN1150':{'type':'sig', 'leg':'N_{1}=150 GeV','l_color':kGreen,'path':'LFCMN1150'+sig_suffix},
    'LFCMN1450':{'type':'sig', 'leg':'N_{1}=450 GeV','l_color':kGreen+3,'path':'LFCMN1450'+sig_suffix},
    
#------------------------------------------

    'MGPy8EG_A14N23LO_C1N2_WZ_140_100_3L_3L3':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (140,100)','l_color':lime,    '  path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_140_115_3L_3L3':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (140,115)','l_color':orange,   'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_140_125_3L_3L3':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (140,125)','l_color':yellow,   'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_60_3L_3L3' :{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (150,60)' ,'l_color':magenta,  'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_70_3L_3L3' :{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (150,70)' ,'l_color':green,    'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_80_3L_3L3' :{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (150,80)' ,'l_color':cyan,     'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_150_90_3L_3L3' :{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (150,90)' ,'l_color':blue,     'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_150_3L_3L3':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (200,150)','l_color':red,   'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_110_3L_3L3':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (200,110)','l_color':blue,  'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_120_3L_3L3':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (200,120)','l_color':pink,     'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_130_3L_3L3':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (200,130)','l_color':navy,     'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_140_3L_3L3':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (200,140)','l_color':apricot,  'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_160_3L_3L3':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (200,160)','l_color':mint,     'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_180_3L_3L3':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (200,180)','l_color':teal,     'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_200_190_3L_3L3':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (200,190)','l_color':green, 'path':'AllSignals'+sig_suffix},
    'MGPy8EG_A14N23LO_C1N2_WZ_250_210_3L_3L3':{'type':'sig','leg':'#tilde{#font[12]{\chi}}_{1}^{\pm} #tilde{#font[12]{\chi}}_{2}^{0} (250,210)','l_color':maroon,   'path':'AllSignals'+sig_suffix}, 
  }

  return d_samp
