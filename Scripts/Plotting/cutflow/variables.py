#____________________________________________________________________________
def configure_vars(sig_reg):

  # format for each variable is
  #'var_name_in_ntuple':{TLaTeX axis entry, units, Nbins, xmin, xmax, arrow position (for N-1 plots), arrow direction}
  # to do variable bin widths, place 'var' as value of 'hXNbins' and specify lower bin edges as 'binsLowE':[0,4,5,11,15,20,40,60]
  # e.g.     'lep2Pt':{'tlatex':'p_{T}(#font[12]{l}_{2})','units':'GeV','hXNbins':'var','hXmin':0,'hXmax':60,'binsLowE':[0,4,5,11,15,20,40,60],'cut_pos':200,'cut_dir':'upper'}, 

  # edit for the 3L analysis: the 'variable key' does not have to be the name of the variable as it appears in the TTree, it can also contain plotting information. If so, add key 'ntupVar' with the actual name.
 
  d_met_Et = {}

  if "WZ-3L" in sig_reg:
    d_met_Et = {'tlatex':'E_{T}^{miss}','units':'GeV','hXNbins':10,'hXmin':100,'hXmax':200,'cut_pos':200,'cut_dir':'lower'}
  elif "ZZ-4L" in sig_reg:
    d_met_Et = {'tlatex':'E_{T}^{miss}','units':'GeV','hXNbins':5,'hXmin':0,'hXmax':50,'cut_pos':200,'cut_dir':'lower'}
  elif "SR2-high" in sig_reg:
    d_met_Et = {'tlatex':'E_{T}^{miss}','units':'GeV','hXNbins':30,'hXmin':0,'hXmax':300,'cut_pos':250,'cut_dir':'lower'}
  elif "SR2-int" in sig_reg:
    d_met_Et = {'tlatex':'E_{T}^{miss}','units':'GeV','hXNbins':30,'hXmin':0,'hXmax':300,'cut_pos':150,'cut_dir':'lower'}
  elif "SR2-low-2J" in sig_reg:
    d_met_Et = {'tlatex':'E_{T}^{miss}','units':'GeV','hXNbins':30,'hXmin':0,'hXmax':300,'cut_pos':100,'cut_dir':'lower'}
  else:
    d_met_Et = {'tlatex':'E_{T}^{miss}','units':'GeV','hXNbins':30,'hXmin':0,'hXmax':300,'cut_pos':200,'cut_dir':'lower'}

  d_vars = {

    ### 3L variables

    'jet1Pt':{'tlatex':'p_{T}(j_{1})','units':'GeV','hXNbins':25,'hXmin':0,'hXmax':500,'cut_pos':100,'cut_dir':'lower'},
    'lep3Pt'                  :{'tlatex':'p_{T}^{l3}','units':'GeV','hXNbins':8,'hXmin':0,'hXmax':40,'cut_pos':200,'cut_dir':'lower'},
#    'lep3Pt'                  :{'tlatex':'p_{T}(#font[12]{l}_{3})','units':'GeV','hXNbins':8,'hXmin':0,'hXmax':40,'cut_pos':200,'cut_dir':'lower'},
    'L3_mt_minMll_upper'      :{'tlatex':'m_{T}^{min mll}','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':100,'cut_pos':70,'cut_dir':'upper','ntupVar':'L3_mt_minMll'},
    'L3_mt_minMll_lower'      :{'tlatex':'m_{T}^{min mll}','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':100,'cut_pos':70,'cut_dir':'lower','ntupVar':'L3_mt_minMll'},
    'L3_pT3lOverMet'          :{'tlatex':'p_{T}^{leptons}/E_{T}^{miss}','units':'','hXNbins':8,'hXmin':0,'hXmax':2.0,'cut_pos':10,'cut_dir':'upper'},
    'L3_minDeltaR'            :{'tlatex':'min\DeltaR','units':'','hXNbins':20,'hXmin':0,'hXmax':3.2,'cut_pos':1.6,'cut_dir':'upper'},
    'L3_m3l'                  :{'tlatex':'m_{lll}','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':200,'cut_pos':70,'cut_dir':'lower'},
#    'L3_mt2leplsp_100'        :{'tlatex':'m_{T2}^{100}','units':'GeV','hXNbins':30,'hXmin':0,'hXmax':300,'cut_pos':200,'cut_dir':'upper'},
    'L3_mt2leplsp_100_minMll' :{'tlatex':'m_{T2}^{100}','units':'GeV','hXNbins':30,'hXmin':0,'hXmax':300,'cut_pos':200,'cut_dir':'upper'},
    'met_Signif'             :{'tlatex':'E_{T}^{miss} sign.','units':'','hXNbins':40,'hXmin':0,'hXmax':20,'cut_pos':0.4,'cut_dir':'lower'}, 
    #'L3_maxMll_0_200_upper':{'tlatex':'max m_{ll}','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':200,'cut_pos':70,'cut_dir':'upper','ntupVar':'L3_maxMll'}, 
    #'L3_maxMll_0_400_upper':{'tlatex':'max m_{ll}','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':400,'cut_pos':70,'cut_dir':'upper','ntupVar':'L3_maxMll'}, 
    #'L3_maxMll_0_200_lower':{'tlatex':'max m_{ll}','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':200,'cut_pos':70,'cut_dir':'lower','ntupVar':'L3_maxMll'}, 
    #'L3_maxMll_0_400_lower':{'tlatex':'max m_{ll}','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':400,'cut_pos':70,'cut_dir':'lower','ntupVar':'L3_maxMll'}, 
    #'L3_maxMll'              :{'tlatex':'max m_{ll}','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':500,'cut_pos':70,'cut_dir':'upper'}, 
    #'L3_mt_minMll'           :{'tlatex':'m_{T}^{min mll}','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':200,'cut_pos':70,'cut_dir':'upper'},
    #'L3_minMll'              :{'tlatex':'min m_{ll}','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':200,'cut_pos':70,'cut_dir':'upper'}, 



    ### 2L2J variables (kept for reference)

    #MET related variables
    'met_Et':d_met_Et,
    'met_track_Et':{'tlatex':'Track E_{T}^{miss}','units':'GeV','hXNbins':15,'hXmin':0,'hXmax':300,'cut_pos':900,'cut_dir':'lower'},
    'METRel':{'tlatex':'E_{T}^{miss,rel}','units':'GeV','hXNbins':30,'hXmin':0,'hXmax':600,'cut_pos':900,'cut_dir':'lower'},
    'METTrackRel':{'tlatex':'Track E_{T}^{miss,rel}','units':'GeV','hXNbins':30,'hXmin':0,'hXmax':600,'cut_pos':900,'cut_dir':'lower'},
    'dPhiMetAndMetTrack':{'tlatex':'|#Delta#phi(#bf{p}_{T}^{miss},Track#bf{p}_{T}^{miss})|','units':'','hXNbins':20,'hXmin':0,'hXmax':4,'cut_pos':200,'cut_dir':'upper'},
    'TST_Et':{'tlatex':'TracksofttermE_{T}','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':100,'cut_pos':200,'cut_dir':'lower'},
    'fabs(dPhiMetAndMetTrack)':{'tlatex':'|#Delta#phi(#bf{p}_{T}^{miss}, Track #bf{p}_{T}^{miss})|','units':'','hXNbins':25,'hXmin':0,'hXmax':4,'cut_pos':200,'cut_dir':'upper'},
    
    #Jet variables
    #'jetPt[0]':{'tlatex':'p_{T}(j_{1})','units':'GeV','hXNbins':25,'hXmin':0,'hXmax':500,'cut_pos':100,'cut_dir':'lower'},
    'jetPt[1]':{'tlatex':'p_{T}(j_{2})','units':'GeV','hXNbins':25,'hXmin':0,'hXmax':500,'cut_pos':900,'cut_dir':'lower'},
    'jetPt[2]':{'tlatex':'p_{T}(j_{3})','units':'GeV','hXNbins':10,'hXmin':0,'hXmax':200,'cut_pos':30,'cut_dir':'lower'},
    'jetPt[3]':{'tlatex':'p_{T}(j_{4})','units':'GeV','hXNbins':10,'hXmin':0,'hXmax':200,'cut_pos':30,'cut_dir':'lower'},
    'jetPt[4]':{'tlatex':'p_{T}(j_{5})','units':'GeV','hXNbins':10,'hXmin':0,'hXmax':200,'cut_pos':400,'cut_dir':'upper'},
    'jetEta[0]':{'tlatex':'#eta(j_{1})','units':'','hXNbins':50,'hXmin':-5,'hXmax':5,'cut_pos':10,'cut_dir':'upper'},
    'jetEta[1]':{'tlatex':'#eta(j_{2})','units':'','hXNbins':50,'hXmin':-5,'hXmax':5,'cut_pos':10,'cut_dir':'upper'},

    # Jet multiplicity
    'nTotalJet':{'tlatex':'N_{|#eta|<4.5Jets}^{pT>25GeV}','units':'','hXNbins':10,'hXmin':0,'hXmax':10,'cut_pos':13,'cut_dir':'upper'},
    'nTotalJet20':{'tlatex':'N_{|#eta|<4.5Jets}^{pT>20GeV}','units':'','hXNbins':10,'hXmin':0,'hXmax':10,'cut_pos':13,'cut_dir':'upper'},
    'nJet30':{'tlatex':'N_{|#eta|<2.8Jets}^{pT>30GeV}','units':'','hXNbins':10,'hXmin':0,'hXmax':10,'cut_pos':45,'cut_dir':'upper'},
    'nJet25':{'tlatex':'N_{|#eta|<2.8Jets}^{pT>25GeV}','units':'','hXNbins':10,'hXmin':0,'hXmax':10,'cut_pos':3,'cut_dir':'upper'},
    'nJet20':{'tlatex':'N_{|#eta|<2.8Jets}^{pT>20GeV}','units':'','hXNbins':10,'hXmin':0,'hXmax':10,'cut_pos':3,'cut_dir':'upper'},

    # B tagged multiplicity
    'nBJet20_MV2c10_FixedCutBEff_77':{'tlatex':'N_{b-jet}^{pT>20GeV}, 77% WP','units':'','hXNbins':7,'hXmin':0,'hXmax':7,'cut_pos':1,'cut_dir':'upper'},
    'nBJet30_85':{'tlatex':'N_{b-jet}^{pT>30GeV},85%WP','units':'','hXNbins':7,'hXmin':0,'hXmax':7,'cut_pos':1,'cut_dir':'upper'},
    'nBJet30_77':{'tlatex':'N_{b-jet}^{pT>30GeV},77%WP','units':'','hXNbins':7,'hXmin':0,'hXmax':7,'cut_pos':1,'cut_dir':'upper'},
    'nBJet30_70':{'tlatex':'N_{b-jet}^{pT>30GeV},70%WP','units':'','hXNbins':7,'hXmin':0,'hXmax':7,'cut_pos':1,'cut_dir':'upper'},
    'nBJet30_60':{'tlatex':'N_{b-jet}^{pT>30GeV},60%WP','units':'','hXNbins':7,'hXmin':0,'hXmax':7,'cut_pos':1,'cut_dir':'upper'},

    # Higher level jet variables
    'mjj':{'tlatex':'m(#font[12]{jj})','units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':200,'cut_pos':230,'cut_dir':'upper'},
    'Rjj':{'tlatex':'#DeltaR(#font[12]{jj})','units':'','hXNbins':25,'hXmin':0,'hXmax':5,'cut_pos':2.0,'cut_dir':'upper'},
    'hadronicWPt':{'tlatex':'p_{T}(jj)','units':'GeV','hXNbins':100,'hXmin':0,'hXmax':1000,'cut_pos':2.5,'cut_dir':'lower'},
    'Ht30':{'tlatex':'H_{T}','units':'GeV','hXNbins':50,'hXmin':0,'hXmax':1000,'cut_pos':2.5,'cut_dir':'lower'},
    'vectorSumJetsPt':{'tlatex':'p_{T} of vectorial sum of all jets','units':'GeV','hXNbins':100,'hXmin':0,'hXmax':1000,'cut_pos':2.5,'cut_dir':'lower'},

    # Jets and MET
    'DPhiJ1Met':{'tlatex':'|#Delta#phi(#bf{p}_{T}^{j1}, #bf{p}_{T}^{miss})|','units':'','hXNbins':20,'hXmin':0,'hXmax':4,'cut_pos':2.0,'cut_dir':'lower'},
    'DPhiJ2Met':{'tlatex':'|#Delta#phi(#bf{p}_{T}^{j2}, #bf{p}_{T}^{miss})|','units':'','hXNbins':20,'hXmin':0,'hXmax':4,'cut_pos':2.5,'cut_dir':'lower'},
    'DPhiJ3Met':{'tlatex':'|#Delta#phi(#bf{p}_{T}^{j3}, #bf{p}_{T}^{miss})|','units':'','hXNbins':20,'hXmin':0,'hXmax':4,'cut_pos':2.5,'cut_dir':'lower'},
    'DPhiJ4Met':{'tlatex':'|#Delta#phi(#bf{p}_{T}^{j4}, #bf{p}_{T}^{miss})|','units':'','hXNbins':20,'hXmin':0,'hXmax':4,'cut_pos':2.5,'cut_dir':'lower'},
    'minDPhi4JetsMet':{'tlatex':'min[|#Delta#phi(#bf{p}_{T}^{leading four jets}, #bf{p}_{T}^{miss})|]','units':'','hXNbins':20,'hXmin':0,'hXmax':4,'cut_pos':2.5,'cut_dir':'lower'},
    'minDPhiAllJetsMet':{'tlatex':'min[|#Delta#phi(#bf{p}_{T}^{All jets}, #bf{p}_{T}^{miss})|]','units':'','hXNbins':20,'hXmin':0,'hXmax':4,'cut_pos':0.4,'cut_dir':'lower'},
    'dPhiVectorSumJetsMET':{'tlatex':'|#Delta#phi(#bf{H}_{T}^{jets}, #bf{p}_{T}^{miss})|]','units':'','hXNbins':16,'hXmin':0,'hXmax':4,'cut_pos':2.5,'cut_dir':'lower'},
    'dPhiVectorSumJetsJ1':{'tlatex':'|#Delta#phi(#bf{H}_{T}^{jets}, #bf{p}_{T}^{j1})|]','units':'','hXNbins':16,'hXmin':0,'hXmax':4,'cut_pos':2.5,'cut_dir':'lower'},
    'METOverJ1pT':{'tlatex':'E_{T}^{miss}/p_{T}(j_{1})','units':'','hXNbins':40,'hXmin':0,'hXmax':6.0,'cut_pos':10,'cut_dir':'lower'},
    'METOverHT':{'tlatex':'E_{T}^{miss}/H_{T}^{jets}','units':'','hXNbins':40,'hXmin':0,'hXmax':6.0,'cut_pos':10,'cut_dir':'lower'},
    'METOverPtW':{'tlatex':'E_{T}^{miss}/p_{T}^{W}','units':'','hXNbins':40,'hXmin':0,'hXmax':6.0,'cut_pos':10,'cut_dir':'lower'},
    'dPhiPjjMet':{'tlatex':'|#Delta#phi(#bf{p}_{#font[12]{jj}},#bf{p}_{T}^{miss})|','units':'','hXNbins':25,'hXmin':0,'hXmax':5,'cut_pos':170,'cut_dir':'upper'},

    # Count leptons
    'nLep_base':{'tlatex':'N(baseline leptons)','units':'','hXNbins':10,'hXmin':0,'hXmax':10,'cut_pos':120,'cut_dir':'upper'},
    'nLep_signal':{'tlatex':'N(signal leptons)','units':'','hXNbins':10,'hXmin':0,'hXmax':10,'cut_pos':123,'cut_dir':'upper'},

    # Info for single leptons
    'lepPt':{'tlatex':'p_{T}(#font[12]{l})','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':400,'cut_pos':200,'cut_dir':'upper'},
    'lepPt[0]':{'tlatex':'p_{T}(#font[12]{l}_{1})','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':400,'cut_pos':200,'cut_dir':'upper'},
    'lepPt[1]':{'tlatex':'p_{T}(#font[12]{l}_{2})','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':400,'cut_pos':200,'cut_dir':'upper'},
    'lepPt[2]':{'tlatex':'p_{T}(#font[12]{l}_{3})','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':400,'cut_pos':200,'cut_dir':'upper'},
    'lepPt[3]':{'tlatex':'p_{T}(#font[12]{l}_{4})','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':400,'cut_pos':200,'cut_dir':'upper'},

    'lepFlavor':{'tlatex':'Lepton flavor','units':'','hXNbins':2,'hXmin':0.5,'hXmax':2.5,'cut_pos':0,'cut_dir':'upper'},

    #'lep1Pt':{'tlatex':'p_{T}(#font[12]{l}_{1})','units':'GeV','hXNbins':'var','hXmin':0,'hXmax':60,'binsLowE':[0,4,5,11,15,20,40,60],'cut_pos':200,'cut_dir':'upper'},
    #'lep2Pt':{'tlatex':'p_{T}(#font[12]{l}_{2})','units':'GeV','hXNbins':'var','hXmin':0,'hXmax':60,'binsLowE':[0,4,5,11,15,20,40,60],'cut_pos':200,'cut_dir':'upper'},
    'lep1Eta':{'tlatex':'#eta(#font[12]{l}_{1})','units':'','hXNbins':50,'hXmin':-5,'hXmax':5,'cut_pos':120,'cut_dir':'upper'},
    'lep2Eta':{'tlatex':'#eta(#font[12]{l}_{2})','units':'','hXNbins':50,'hXmin':-5,'hXmax':5,'cut_pos':120,'cut_dir':'upper'},
    'lep1Phi':{'tlatex':'#phi(#font[12]{l}_{1})','units':'','hXNbins':40,'hXmin':-4,'hXmax':4,'cut_pos':120,'cut_dir':'upper'},
    'lep2Phi':{'tlatex':'#phi(#font[12]{l}_{2})','units':'','hXNbins':40,'hXmin':-4,'hXmax':4,'cut_pos':120,'cut_dir':'upper'},

    'lep1Type':{'tlatex':'Lep 1 MC Truth Type','units':'','hXNbins':39,'hXmin':0,'hXmax':39,'cut_pos':50,'cut_dir':'upper'},
    'lep2Type':{'tlatex':'Lep 2 MCTruth Type','units':'','hXNbins':39,'hXmin':0,'hXmax':39,'cut_pos':50,'cut_dir':'upper'},
    'lep1Origin':{'tlatex':'Lep 1 MC Truth Origin','units':'','hXNbins':46,'hXmin':0,'hXmax':46,'cut_pos':70,'cut_dir':'upper'},
    'lep2Origin':{'tlatex':'Lep 2 MC Truth Origin','units':'','hXNbins':46,'hXmin':0,'hXmax':46,'cut_pos':70,'cut_dir':'upper'},
 
    'lep1D0':{'tlatex':'d_{0}(#font[12]{l}_{1})','units':'mm','hXNbins':80,'hXmin':-20,'hXmax':20,'cut_pos':50,'cut_dir':'upper'},
    'lep2D0':{'tlatex':'d_{0}(#font[12]{l}_{2})','units':'mm','hXNbins':80,'hXmin':-20,'hXmax':20,'cut_pos':50,'cut_dir':'upper'},
    'lep1D0Sig':{'tlatex':'d_{0}(#font[12]{l}_{1}) / #sigma(d_{0}(#font[12]{l}_{1}))','units':'','hXNbins':40,'hXmin':0,'hXmax':20,'cut_pos':50,'cut_dir':'upper'},
    'lep2D0Sig':{'tlatex':'d_{0}(#font[12]{l}_{2}) / #sigma(d_{0}(#font[12]{l}_{1}))','units':'','hXNbins':40,'hXmin':0,'hXmax':20,'cut_pos':50,'cut_dir':'upper'},
    'lep1Z0':{'tlatex':'z_{0}(#font[12]{l}_{1})','units':'mm','hXNbins':80,'hXmin':-20,'hXmax':20,'cut_pos':50,'cut_dir':'upper'},
    'lep2Z0':{'tlatex':'z_{0}(#font[12]{l}_{2})','units':'mm','hXNbins':80,'hXmin':-20,'hXmax':20,'cut_pos':50,'cut_dir':'upper'},
    'lep1Z0SinTheta':{'tlatex':'z_{0}(#font[12]{l}_{1})','units':'mm','hXNbins':40,'hXmin':0,'hXmax':20,'cut_pos':50,'cut_dir':'upper'},
    'lep2Z0SinTheta':{'tlatex':'z_{0}(#font[12]{l}_{2})','units':'mm','hXNbins':40,'hXmin':0,'hXmax':20,'cut_pos':50,'cut_dir':'upper'},
 
    'lep1Ptvarcone20':{'tlatex':'p_{T} Var Cone 20','units':'','hXNbins':40,'hXmin':0,'hXmax':20,'cut_pos':999,'cut_dir':'upper'},
    'lep1Ptvarcone30':{'tlatex':'p_{T} Var Cone 30','units':'','hXNbins':40,'hXmin':0,'hXmax':20,'cut_pos':999,'cut_dir':'upper'},
    'lep1Topoetcone20':{'tlatex':'p_{T} Topo E_{T} Cone 30','units':'','hXNbins':40,'hXmin':0,'hXmax':20,'cut_pos':999,'cut_dir':'upper'},
    'lep2Ptvarcone20':{'tlatex':'p_{T} Var Cone 20','units':'','hXNbins':40,'hXmin':0,'hXmax':20,'cut_pos':999,'cut_dir':'upper'},
    'lep2Ptvarcone30':{'tlatex':'p_{T} Var Cone 30','units':'','hXNbins':40,'hXmin':0,'hXmax':20,'cut_pos':999,'cut_dir':'upper'},
    'lep2Topoetcone20':{'tlatex':'p_{T} Topo E_{T} Cone 30','units':'','hXNbins':40,'hXmin':0,'hXmax':20,'cut_pos':999,'cut_dir':'upper'},

    # Higher level lepton variables 
    'LepCosThetaCoM':{'tlatex':'|cos#theta_{#font[12]{ll}}^{Melia}|','units':'','hXNbins':10,'hXmin':0,'hXmax':1,'cut_pos':1.0,'cut_dir':'upper'},
    'LepCosThetaLab':{'tlatex':'|cos#theta_{#font[12]{ll}}^{Barr}|','units':'','hXNbins':10,'hXmin':0,'hXmax':1,'cut_pos':1.0,'cut_dir':'upper'},
    'Rll':{'tlatex':'#DeltaR(#font[12]{ll})','units':'','hXNbins':25,'hXmin':0,'hXmax':5,'cut_pos':2.0,'cut_dir':'upper'},
    #'mll':{'tlatex':'m(#font[12]{ll})','units':'GeV','hXNbins':'var','hXmin':0.0,'hXmax':60,'binsLowE':[0,1,3,5,10,15,20,25,30,40,50,60],'cut_pos':900,'cut_dir':'upper'},
    'mll':{'tlatex':'m(#font[12]{ll})','units':'GeV','hXNbins':15,'hXmin':50,'hXmax':200,'cut_pos':230,'cut_dir':'upper'},
    'lep1PtOverlep2Pt':{'tlatex':'p_{T}^{#font[12]{l}_{1}}/p_{T}^{#font[12]{l}_{2}}','units':'','hXNbins':25,'hXmin':0,'hXmax':10,'cut_pos':120,'cut_dir':'upper'},
    'Ptll':{'tlatex':'p_{T}(#font[12]{ll})','units':'GeV','hXNbins':50,'hXmin':0,'hXmax':1000,'cut_pos':200,'cut_dir':'lower'},

    'lep1PtOverMll':{'tlatex':'p_{T}^{#font[12]{l}_{1}}/m(#font[12]{ll})','units':'','hXNbins':30,'hXmin':0,'hXmax':6,'cut_pos':120,'cut_dir':'lower'},
    'lep2PtOverMll':{'tlatex':'p_{T}^{#font[12]{l}2}/m(#font[12]{ll})','units':'','hXNbins':30,'hXmin':0,'hXmax':6,'cut_pos':120,'cut_dir':'lower'},
    'lep12PtOverMll':{'tlatex':'H_{T}^{leptons}/m(#font[12]{ll})','units':'','hXNbins':50,'hXmin':0,'hXmax':10,'cut_pos':120,'cut_dir':'lower'},
    'lep12PtOverMt2':{'tlatex':'(p_{T}^{#font[12]{l}1}+p_{T}^{#font[12]{l}2})/m_{T2}^{#chi=100GeV}','units':'','hXNbins':20,'hXmin':0,'hXmax':5,'cut_pos':120,'cut_dir':'upper'},
    'RjlOverEl':{'tlatex':'min(R_{j#font[12]{l}})/E_{#font[12]{l}}','units':'GeV^{#minus1}','hXNbins':15,'hXmin':0,'hXmax':0.15,'cut_pos':0.02,'cut_dir':'lower'},
    
    # Lepton variables involving MET
    'mt_lep1':{'tlatex':'m_{T}(#font[12]{l}_{1})','units':'GeV','hXNbins':20,'hXmin':0,'hXmax':200,'cut_pos':70,'cut_dir':'upper'},
    'mt_lep2':{'tlatex':'m_{T}(#font[12]{l}_{2})','units':'GeV','hXNbins':40,'hXmin':0,'hXmax':200,'cut_pos':320,'cut_dir':'upper'},
    'mt_lep1+mt_lep2':{'tlatex':'m_{T}(#font[12]{l}_{1})+m_{T}(#font[12]{l}_{2})','units':'GeV','hXNbins':30,'hXmin':0,'hXmax':300,'cut_pos':820,'cut_dir':'upper'},
    #'METOverHTLep':{'tlatex':'E_{T}^{miss}/H_{T}^{leptons}','units':'','hXNbins':30,'hXmin':0,'hXmax':30.0,'cut_pos':5.0,'cut_dir':'lower'},
    'METOverHTLep':{'tlatex':'E_{T}^{miss}/H_{T}^{leptons}','units':'','hXNbins':'var','hXmin':0,'hXmax':30.0,'binsLowE':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,20,25,30],'cut_pos':5.0,'cut_dir':'lower'},
    'MTauTau':{'tlatex':'m(#tau#tau)','units':'GeV','hXNbins':40,'hXmin':-2000,'hXmax':2000,'cut_pos':0,'cut_dir':'upper','cut_pos2':160,'cut_dir2':'lower'},
    'METOverPtZ':{'tlatex':'E_{T}^{miss}/p_{T}^{Z}','units':'','hXNbins':40,'hXmin':0,'hXmax':6.0,'cut_pos':10,'cut_dir':'lower'},
    'dPhiPllMet':{'tlatex':'|#Delta#phi(#bf{p}_{#font[12]{ll}},#bf{p}_{T}^{miss})|','units':'','hXNbins':25,'hXmin':0,'hXmax':5,'cut_pos':170,'cut_dir':'upper'},

    # mT2 by various trial invisible masses
    'mt2leplsp_0'  :{'tlatex':'m_{T2}^{0}',  'units':'GeV','hXNbins':50,'hXmin':0,  'hXmax':500,'cut_pos':120,'cut_dir':'upper'},
    #'mt2leplsp_0'  :{'tlatex':'m_{T2}^{0}',  'units':'GeV','hXNbins':50,'hXmin':0,  'hXmax':100,'cut_pos':120,'cut_dir':'upper'},
    'mt2leplsp_25' :{'tlatex':'m_{T2}^{25}', 'units':'GeV','hXNbins':50,'hXmin':15, 'hXmax':115,'cut_pos':120,'cut_dir':'upper'},
    'mt2leplsp_50' :{'tlatex':'m_{T2}^{50}', 'units':'GeV','hXNbins':50,'hXmin':40, 'hXmax':140,'cut_pos':120,'cut_dir':'upper'},
    'mt2leplsp_75' :{'tlatex':'m_{T2}^{75}', 'units':'GeV','hXNbins':50,'hXmin':65, 'hXmax':165,'cut_pos':120,'cut_dir':'upper'},
    'mt2leplsp_90' :{'tlatex':'m_{T2}^{90}', 'units':'GeV','hXNbins':50,'hXmin':80, 'hXmax':180,'cut_pos':120,'cut_dir':'upper'},
    'mt2leplsp_100':{'tlatex':'m_{T2}^{100}','units':'GeV','hXNbins':'var','hXmin':90, 'hXmax':190,'binsLowE':[90,100,102,105,110,120,130,190],'cut_pos':920,'cut_dir':'upper'},
    #'mt2leplsp_100':{'tlatex':'m_{T2}^{100}','units':'GeV','hXNbins':50,'hXmin':90, 'hXmax':190,'cut_pos':1120,'cut_dir':'upper'},
    'mt2leplsp_110':{'tlatex':'m_{T2}^{110}','units':'GeV','hXNbins':50,'hXmin':100,'hXmax':200,'cut_pos':120,'cut_dir':'upper'},
    'mt2leplsp_120':{'tlatex':'m_{T2}^{120}','units':'GeV','hXNbins':50,'hXmin':110,'hXmax':210,'cut_pos':120,'cut_dir':'upper'},
    'mt2leplsp_150':{'tlatex':'m_{T2}^{150}','units':'GeV','hXNbins':50,'hXmin':140,'hXmax':240,'cut_pos':120,'cut_dir':'upper'},
    'mt2leplsp_175':{'tlatex':'m_{T2}^{175}','units':'GeV','hXNbins':50,'hXmin':165,'hXmax':265,'cut_pos':120,'cut_dir':'upper'},
    'mt2leplsp_200':{'tlatex':'m_{T2}^{200}','units':'GeV','hXNbins':50,'hXmin':190,'hXmax':290,'cut_pos':120,'cut_dir':'upper'},
    'mt2leplsp_250':{'tlatex':'m_{T2}^{250}','units':'GeV','hXNbins':50,'hXmin':240,'hXmax':340,'cut_pos':120,'cut_dir':'upper'},
    'mt2leplsp_300':{'tlatex':'m_{T2}^{300}','units':'GeV','hXNbins':50,'hXmin':290,'hXmax':390,'cut_pos':120,'cut_dir':'upper'},
    
    'nVtx':{'tlatex':'Number of vertices','units':'','hXNbins':60,'hXmin':0,'hXmax':60,'cut_pos':170,'cut_dir':'upper'},
    'mu':{'tlatex':'Average interactions per crossing #langle #mu #rangle','units':'','hXNbins':70,'hXmin':0,'hXmax':70,'cut_pos':170,'cut_dir':'upper'},
    'actual_mu':{'tlatex':'Actual interactions per crossing #mu','units':'','hXNbins':70,'hXmin':0,'hXmax':70,'cut_pos':170,'cut_dir':'upper'},


    #'var':{'tlatex':'name','units':'','hXNbins':00,'hXmin':00,'hXmax':00,'cut_pos','cut_dir','leftorright'},
    #generic variables used in RJ
    'mTl3'               :{'tlatex':'mTl3'     ,'units':'GeV','hXNbins':20,'hXmin':0.0  ,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
    'mll_RJ'             :{'tlatex':'m_{ll}'   ,'units':'GeV','hXNbins':20,'hXmin':0.0  ,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
    'minDphi'            :{'tlatex':'minDphi'  ,'units':'','hXNbins':20,'hXmin':-3.14,'hXmax':3.14,'cut_pos':3.14,'cut_dir':'upper'},
    'mTW'                :{'tlatex':'m_{T}^{W}','units':'GeV','hXNbins':20,'hXmin':0.0  ,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},


    #categorisation variables

    'is2Lep2Jet'         :{'tlatex':'is2Lep2Jet','units':'','hXNbins':2,'hXmin':0,'hXmax':1,'cut_pos':2,'cut_dir':'upper'},
    'is2L2JInt'          :{'tlatex':'is2L2JInt' ,'units':'','hXNbins':2,'hXmin':0,'hXmax':1,'cut_pos':2,'cut_dir':'upper'},
    'is3Lep'             :{'tlatex':'is3Lep'    ,'units':'','hXNbins':2,'hXmin':0,'hXmax':1,'cut_pos':2,'cut_dir':'upper'},
    'is3LInt'            :{'tlatex':'is3Lint'   ,'units':'','hXNbins':2,'hXmin':0,'hXmax':1,'cut_pos':2,'cut_dir':'upper'},
    'is3Lep2Jet'         :{'tlatex':'is3Lep2Jet','units':'','hXNbins':2,'hXmin':0,'hXmax':1,'cut_pos':2,'cut_dir':'upper'},
    'is3Lep3Jet'         :{'tlatex':'is3Lep3Jet','units':'','hXNbins':2,'hXmin':0,'hXmax':1,'cut_pos':2,'cut_dir':'upper'},
    'is4Lep2Jet'         :{'tlatex':'is4Lep2Jet','units':'','hXNbins':2,'hXmin':0,'hXmax':1,'cut_pos':2,'cut_dir':'upper'},
    'is4Lep3Jet'         :{'tlatex':'is4Lep3Jet','units':'','hXNbins':2,'hXmin':0,'hXmax':1,'cut_pos':2,'cut_dir':'upper'},


    # default hemisphere type variables

    'H2PP'               :{'tlatex':'H_{1,1}^{PP}','units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
    'H4PP'               :{'tlatex':'H_{3,1}^{PP}','units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
    'H5PP'               :{'tlatex':'H_{4,1}^{PP}','units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},

    
    # ratio type rj variables 

    'R_H2PP_H5PP'        :{'tlatex':'H_{1,1}^{PP}/H_{4,1}^{PP}','units':'','hXNbins':10,'hXmin':0.0,'hXmax':1.0,'cut_pos':1.0,'cut_dir':'upper'},
    'RPT_HT4PP'          :{'tlatex':'RPT_HT4PP'                ,'units':'','hXNbins':10,'hXmin':0.0,'hXmax':1.0,'cut_pos':1.0,'cut_dir':'upper'},
    'RPT_HT5PP'          :{'tlatex':'RPT_HT5PP'                ,'units':'','hXNbins':10,'hXmin':0.0,'hXmax':1.0,'cut_pos':1.0,'cut_dir':'upper'},
    'R_HT4PP_H4PP'       :{'tlatex':'R_HT4PP_H4PP'             ,'units':'','hXNbins':10,'hXmin':0.0,'hXmax':1.0,'cut_pos':1.0,'cut_dir':'upper'},
    'R_minH2P_minH3P'    :{'tlatex':'R_minH2P_minH3P'          ,'units':'','hXNbins':10,'hXmin':0.0,'hXmax':1.0,'cut_pos':1.0,'cut_dir':'upper'},
     
     
     #ISR type variables
    
     'NjS'               :{'tlatex':'N_{J}^{S}'     ,'units':'','hXNbins':6,'hXmin':0,'hXmax':6,'cut_pos':6,'cut_dir':'upper'},
     'NjISR'             :{'tlatex':'N_{J}^{ISR}'   ,'units':'','hXNbins':6,'hXmin':0,'hXmax':6,'cut_pos':6,'cut_dir':'upper'},
     'MJ'                :{'tlatex':'M_{J}'         ,'units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'MZ'                :{'tlatex':'M_{Z}'         ,'units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'PTI'               :{'tlatex':'P_{T, I}^{CM}' ,'units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'PTISR'             :{'tlatex':'P_{T,ISR}^{CM}','units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'PTCM'              :{'tlatex':'P_{T,CM}'      ,'units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'RISR'              :{'tlatex':'R_{ISR}'       ,'units':'','hXNbins':10,'hXmin':0.0,'hXmax':1.0,'cut_pos':1.0,'cut_dir':'upper'},
     'dphiVP'            :{'tlatex':'dphiVP'        ,'units':'','hXNbins':16,'hXmin':0.0,'hXmax':3.2,'cut_pos':3.2,'cut_dir':'upper'},
     'dphiISRI'          :{'tlatex':'dphiISRI'      ,'units':'','hXNbins':16,'hXmin':0.0,'hXmax':3.2,'cut_pos':3.2,'cut_dir':'upper'},


     #RJ variables for a decay tree which requires 2 leptons, but has 3  or 4 leptons in the CR. 

     'lept1Pt_VR'        :{'tlatex':'lept1Pt_VR'        ,'units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'lept2Pt_VR'        :{'tlatex':'lept1Pt_VR'        ,'units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'lept1sign_VR'      :{'tlatex':'lept1sign_VR'      ,'units':'','hXNbins':10,'hXmin':0,'hXmax':10,'cut_pos':10,'cut_dir':'upper'},
     'lept1sign_VR'      :{'tlatex':'lept2sign_VR'      ,'units':'','hXNbins':10,'hXmin':0,'hXmax':10,'cut_pos':10,'cut_dir':'upper'},
     'mll_RJ_VR'         :{'tlatex':'m_{ll}'            ,'units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'H2PP_VR'           :{'tlatex':'H2PP_VR'           ,'units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'H5PP_VR'           :{'tlatex':'H5PP_VR'           ,'units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'R_H2PP_H5PP_VR'    :{'tlatex':'R_H2PP_H5PP_VR'    ,'units':'','hXNbins':10,'hXmin':0.0,'hXmax':1,'cut_pos':1,'cut_dir':'upper'},
     'RPT_HT5PP_VR'      :{'tlatex':'RPT_HT5PP_VR'      ,'units':'','hXNbins':10,'hXmin':0.0,'hXmax':1,'cut_pos':1,'cut_dir':'upper'},
     'R_minH2P_minH3P_VR':{'tlatex':'R_minH2P_minH3P_VR','units':'','hXNbins':10,'hXmin':0.0,'hXmax':1,'cut_pos':1,'cut_dir':'upper'},
     

     'NjS_VR'            :{'tlatex':'N_{J}^{S}'     ,'units':'','hXNbins':6,'hXmin':0,'hXmax':6,'cut_pos':6,'cut_dir':'upper'},
     'NjISR_VR'          :{'tlatex':'N_{J}^{ISR}'   ,'units':'','hXNbins':6,'hXmin':0,'hXmax':6,'cut_pos':6,'cut_dir':'upper'},
     'MJ_VR'             :{'tlatex':'M_{J}'         ,'units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'MZ_VR'             :{'tlatex':'M_{Z}'         ,'units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'PTI_VR'            :{'tlatex':'P_{T, I}^{CM}' ,'units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'PTISR_VR'          :{'tlatex':'P_{T,ISR}^{CM}','units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'PTCM_VR'           :{'tlatex':'P_{T,CM}'      ,'units':'GeV','hXNbins':20,'hXmin':0.0,'hXmax':400,'cut_pos':400,'cut_dir':'upper'},
     'RISR_VR'           :{'tlatex':'R_{ISR}'       ,'units':'','hXNbins':10,'hXmin':0.0,'hXmax':1.0,'cut_pos':1.0,'cut_dir':'upper'},
     'dphiVP_VR'         :{'tlatex':'dphiVP'        ,'units':'','hXNbins':16,'hXmin':0.0,'hXmax':3.2,'cut_pos':3.2,'cut_dir':'upper'},
     'dphiISRI_VR'       :{'tlatex':'dphiISRI'      ,'units':'','hXNbins':16,'hXmin':0.0,'hXmax':3.2,'cut_pos':3.2,'cut_dir':'upper'},

  }

  return d_vars

