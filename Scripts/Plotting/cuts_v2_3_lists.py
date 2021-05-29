## 3L EWK REGIONS
## 
## This file defines all the regions used in the analysis.
## This file is read by HistFitter and is also the reference
## used to determine the reg_flag variable in the ntuples.
##

cutsDict = {}

####################
## COMMON CUTS    ##
####################

#Simple nLep_base=3
cuts_TriL = ['nLep_base==3', 'nLep_signal==3']

cuts_3L = ['nLep_base==3', 'nLep_signal==3']#, 'L3_SFOS', 'nLep_base_comb==3']
PLT_cut = ['lep1Signal', 'lep2Signal', 'lep3Signal', 'lep3IsoPLVTight']
TRG_cut = ['((met_Et>200. && combinedTrigResultMET==1) || (segTrigMatchLEP==1 && lep2Pt>10.))']

#To be applied in all SRs, CRs, VRs
cuts_3L_PLT_TRG_dR = cuts_3L + PLT_cut + TRG_cut + ['L3_minDeltaR3L>0.4']

####################
## SIGNAL REGIONS ##
####################

## 3L Analysis - Trilepton ##
# Sign
SS_12 = ['lep1Charge == lep2Charge']  # Same sign
OS_12 = ['lep1Charge != lep2Charge']  # Opposite sign
SS_13 = ['lep1Charge == lep3Charge']  # Same sign
OS_13 = ['lep1Charge != lep3Charge']  # Opposite sign
SS_23 = ['lep2Charge == lep3Charge']  # Same sign
OS_23 = ['lep2Charge != lep3Charge']  # Opposite sign
# Flavor
SF_12 = ['lep1Flavor == lep2Flavor']  # Same flavor
DF_12 = ['lep1Flavor != lep2Flavor']  # Different flavor
SF_13 = ['lep1Flavor == lep3Flavor']  # Same flavor
DF_13 = ['lep1Flavor != lep3Flavor']  # Different flavor
SF_23 = ['lep2Flavor == lep3Flavor']  # Same flavor
DF_23 = ['lep2Flavor != lep3Flavor']  # Different flavor

#cutsDict['3L'] = cuts_3L
cutsDict['3L'] = cuts_TriL
cutsDict['SR_3L'] = cuts_TriL

cutsDict['3L_SS_12'] = cuts_3L + SS_12
cutsDict['3L_OS_12'] = cuts_3L + OS_12
cutsDict['3L_SS_13'] = cuts_3L + SS_13
cutsDict['3L_OS_13'] = cuts_3L + OS_13
cutsDict['3L_SS_23'] = cuts_3L + SS_23
cutsDict['3L_OS_23'] = cuts_3L + OS_23

cutsDict['3L_SF_12'] = cuts_3L + SF_12
cutsDict['3L_DF_12'] = cuts_3L + DF_12
cutsDict['3L_SF_13'] = cuts_3L + SF_13
cutsDict['3L_DF_13'] = cuts_3L + DF_13
cutsDict['3L_SF_23'] = cuts_3L + SF_23
cutsDict['3L_DF_23'] = cuts_3L + DF_23

cutsDict['3L_SS_SF_12'] = cuts_3L + SS_12 + SF_12
cutsDict['3L_SS_DF_12'] = cuts_3L + SS_12 + DF_12
cutsDict['3L_OS_SF_12'] = cuts_3L + OS_12 + SF_12
cutsDict['3L_OS_DF_12'] = cuts_3L + OS_12 + DF_12
cutsDict['3L_SS_SF_13'] = cuts_3L + SS_13 + SF_13
cutsDict['3L_SS_DF_13'] = cuts_3L + SS_13 + DF_13
cutsDict['3L_OS_SF_13'] = cuts_3L + OS_13 + SF_13
cutsDict['3L_OS_DF_13'] = cuts_3L + OS_13 + DF_13
cutsDict['3L_SS_SF_23'] = cuts_3L + SS_23 + SF_23
cutsDict['3L_SS_DF_23'] = cuts_3L + SS_23 + DF_23
cutsDict['3L_OS_SF_23'] = cuts_3L + OS_23 + SF_23
cutsDict['3L_OS_DF_23'] = cuts_3L + OS_23 + DF_23

# Classified 3L analysis
vtx123 = ['pred_class == 123']
vtx132 = ['pred_class == 132']
vtx213 = ['pred_class == 213']

cutsDict['3LClass_SF_OS'] =  cuts_TriL + SF_12 + OS_12
cutsDict['3LClass_DF_OS'] =  cuts_TriL + DF_12 + OS_12
cutsDict['SR_3LClass_SF_OS_vtx123'] = cuts_TriL + SF_12 + OS_12 + vtx123
cutsDict['SR_3LClass_SF_OS_vtx132'] = cuts_TriL + SF_13 + OS_13 + vtx132
cutsDict['SR_3LClass_SF_OS_vtx213'] = cuts_TriL + SF_12 + OS_12 + vtx213

cutsDict['SR_3LClass_DF_OS_vtx123'] = cuts_TriL + DF_12 + OS_12 + vtx123
cutsDict['SR_3LClass_DF_OS_vtx132'] = cuts_TriL + DF_13 + OS_13 + vtx132
cutsDict['SR_3LClass_DF_OS_vtx213'] = cuts_TriL + DF_12 + OS_12 + vtx213

cutsDict['3LClass_SF_OS_vtx123'] = cuts_TriL + SF_12 + OS_12 + vtx123
cutsDict['3LClass_SF_OS_vtx132'] = cuts_TriL + SF_13 + OS_13 + vtx132
cutsDict['3LClass_SF_OS_vtx213'] = cuts_TriL + SF_12 + OS_12 + vtx213

cutsDict['3LClass_DF_OS_vtx123'] = cuts_TriL + DF_12 + OS_12 + vtx123
cutsDict['3LClass_DF_OS_vtx132'] = cuts_TriL + DF_13 + OS_13 + vtx132
cutsDict['3LClass_DF_OS_vtx213'] = cuts_TriL + DF_12 + OS_12 + vtx213


# Benchmark analysis
mll = ['mll_12>10', 'mll_13>10', 'mll_23>10', 'm_3l>80']
pT = ['lep1_pt>55', 'lep2_pt>15']
mll_MZ = ['abs(mll_12-91.2)>15', 'abs(mll_13-91.2)>15', 'abs(mll_23-91.2)>15', 'abs(m_3l-91.2)>15']

cutsDict['SR_Benchmark'] = mll + pT + mll_MZ
cutsDict['Benchmark'] = mll +pT + mll_MZ


#--------------------------------------------------------

common_SR = ['L3_mll < 75', 'nBJet20_MV2c10==0']
resonances_veto = ['((L3_minMll<3 || L3_minMll>3.2) && (L3_minMll<9. || L3_minMll>12.))']

## LOW MET ##

#0J
lowMET_0J    = ['nJet30==0', 'met_Et<50.', 'met_Signif>1.5']
fakes_conv   = ['(((L3_isEEE || L3_isMME) && abs(L3_m3l-91.2)>20 && L3_minDeltaR<2.4 && L3_minDeltaR>0.6) || (L3_isMMM || L3_isEEM))']

lowMLLb_0J =  ['L3_maxMll<60.', 'L3_minMll>12.', 'L3_minMll<15.', 'lep3Pt>10.', 'L3_mt_minMll<50.', 'L3_pT3lOverMet<1.1', 'L3_mt2leplsp_100_minMll<115', 'L3_minDeltaR<1.6']
lowMLLc_0J =  ['L3_maxMll<60.', 'L3_minMll>15.', 'L3_minMll<20.', 'lep3Pt>10.', 'L3_mt_minMll<50.', 'L3_pT3lOverMet<1.1', 'L3_mt2leplsp_100_minMll<120', 'L3_minDeltaR<1.6']
lowMLLd_0J =  ['L3_maxMll<60.', 'L3_minMll>20.', 'L3_minMll<30.', 'lep3Pt>10.', 'L3_mt_minMll<50.', 'L3_pT3lOverMet<1.1', 'L3_mt2leplsp_100_minMll<130', 'L3_minDeltaR<1.6']
lowMLLe_0J =  ['L3_maxMll<60.', 'L3_minMll>30.', 'L3_minMll<40.', 'lep3Pt>10.', 'L3_mt_minMll<60.', 'L3_pT3lOverMet<1.3']
lowMLLf1_0J = ['L3_maxMll<75.', 'L3_minMll>40.', 'L3_minMll<60.', 'lep3Pt>15.', 'L3_mt_minMll<60.', 'L3_pT3lOverMet<1.4', 'L3_m3l>100.']
lowMLLf2_0J = ['L3_maxMll<75.', 'L3_minMll>40.', 'L3_minMll<60.', 'lep3Pt>15.', 'L3_mt_minMll>90.', 'L3_pT3lOverMet<1.4', 'L3_m3l>100.']
lowMLLg1_0J = ['L3_maxMll<75.', 'L3_minMll>60.', 'L3_minMll<75.', 'lep3Pt>15.', 'L3_mt_minMll<60.', 'L3_pT3lOverMet<1.4', 'L3_m3l>100.']
lowMLLg2_0J = ['L3_maxMll<75.', 'L3_minMll>60.', 'L3_minMll<75.', 'lep3Pt>15.', 'L3_mt_minMll>90.', 'L3_pT3lOverMet<1.4', 'L3_m3l>100.']

#incl f and g
lowMLLf_0J = ['L3_maxMll<75.', 'L3_minMll>40.', 'L3_minMll<60.', 'lep3Pt>15.', '(L3_mt_minMll<60. || L3_mt_minMll>90)', 'L3_pT3lOverMet<1.4', 'L3_m3l>100.']
lowMLLg_0J = ['L3_maxMll<75.', 'L3_minMll>60.', 'L3_minMll<75.', 'lep3Pt>15.', '(L3_mt_minMll<60. || L3_mt_minMll>90)', 'L3_pT3lOverMet<1.4', 'L3_m3l>100.']

SRlow_0Jb = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_0J + lowMLLb_0J
SRlow_0Jc = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_0J + lowMLLc_0J
SRlow_0Jd = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_0J + lowMLLd_0J
SRlow_0Je = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_0J + lowMLLe_0J
SRlow_0Jf1 = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_0J + lowMLLf1_0J
SRlow_0Jf2 = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_0J + lowMLLf2_0J
SRlow_0Jg1 = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_0J + lowMLLg1_0J
SRlow_0Jg2 = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_0J + lowMLLg2_0J

cutsDict['SRlow_0Jb']  = SRlow_0Jb
cutsDict['SRlow_0Jc']  = SRlow_0Jc 
cutsDict['SRlow_0Jd']  = SRlow_0Jd
cutsDict['SRlow_0Je']  = SRlow_0Je
cutsDict['SRlow_0Jf1'] = SRlow_0Jf1
cutsDict['SRlow_0Jf2'] = SRlow_0Jf2
cutsDict['SRlow_0Jg1'] = SRlow_0Jg1
cutsDict['SRlow_0Jg2'] = SRlow_0Jg2

#nJ
lowMET_nJ    = ['nJet30>0', 'met_Et<200.', 'met_Signif>3.0']

lowMLLb_nJ =  ['L3_maxMll<60.', 'L3_minMll>12.', 'L3_minMll<15.', 'lep3Pt>10.', 'L3_mt_minMll<50', 'L3_pT3lOverMet<1.0', 'L3_mt2leplsp_100_minMll<115', 'L3_minDeltaR<1.6']
lowMLLc_nJ =  ['L3_maxMll<60.', 'L3_minMll>15.', 'L3_minMll<20.', 'lep3Pt>10.', 'L3_mt_minMll<50', 'L3_pT3lOverMet<1.0', 'L3_mt2leplsp_100_minMll<120', 'L3_minDeltaR<1.6']
lowMLLd_nJ =  ['L3_maxMll<60.', 'L3_minMll>20.', 'L3_minMll<30.', 'lep3Pt>10.', 'L3_mt_minMll<50', 'L3_pT3lOverMet<1.0', 'L3_mt2leplsp_100_minMll<130', 'L3_minDeltaR<1.6']
lowMLLe_nJ =  ['L3_maxMll<60.', 'L3_minMll>30.', 'L3_minMll<40.', 'lep3Pt>10.', 'L3_mt_minMll<60', 'L3_pT3lOverMet<1.0']
lowMLLf1_nJ = ['L3_maxMll<75.', 'L3_minMll>40.', 'L3_minMll<60.', 'lep3Pt>15.', 'L3_mt_minMll<60', 'L3_pT3lOverMet<1.2']
lowMLLf2_nJ = ['L3_maxMll<75.', 'L3_minMll>40.', 'L3_minMll<60.', 'lep3Pt>15.', 'L3_mt_minMll>90', 'L3_pT3lOverMet<1.2']
lowMLLg1_nJ = ['L3_maxMll<75.', 'L3_minMll>60.', 'L3_minMll<75.', 'lep3Pt>15.', 'L3_mt_minMll<60', 'L3_pT3lOverMet<1.2']
lowMLLg2_nJ = ['L3_maxMll<75.', 'L3_minMll>60.', 'L3_minMll<75.', 'lep3Pt>15.', 'L3_mt_minMll>90', 'L3_pT3lOverMet<1.2']

#incl f and g
lowMLLf_nJ = ['L3_maxMll<75.', 'L3_minMll>40.', 'L3_minMll<60.', 'lep3Pt>15.', '(L3_mt_minMll<60 || L3_mt_minMll>90)', 'L3_pT3lOverMet<1.2']
lowMLLg_nJ = ['L3_maxMll<75.', 'L3_minMll>60.', 'L3_minMll<75.', 'lep3Pt>15.', '(L3_mt_minMll<60 || L3_mt_minMll>90)', 'L3_pT3lOverMet<1.2']
#

SRlow_nJb  = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_nJ + lowMLLb_nJ
SRlow_nJc  = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_nJ + lowMLLc_nJ
SRlow_nJd  = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_nJ + lowMLLd_nJ
SRlow_nJe  = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_nJ + lowMLLe_nJ
SRlow_nJf1 = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_nJ + lowMLLf1_nJ
SRlow_nJf2 = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_nJ + lowMLLf2_nJ
SRlow_nJg1 = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_nJ + lowMLLg1_nJ
SRlow_nJg2 = cuts_3L_PLT_TRG_dR + common_SR + fakes_conv + lowMET_nJ + lowMLLg2_nJ

cutsDict['SRlow_nJb']  = SRlow_nJb
cutsDict['SRlow_nJc']  = SRlow_nJc
cutsDict['SRlow_nJd']  = SRlow_nJd
cutsDict['SRlow_nJe']  = SRlow_nJe
cutsDict['SRlow_nJf1'] = SRlow_nJf1
cutsDict['SRlow_nJf2'] = SRlow_nJf2
cutsDict['SRlow_nJg1'] = SRlow_nJg1
cutsDict['SRlow_nJg2'] = SRlow_nJg2

 
## HIGH MET ## 

common_highMET = ['L3_maxMll<75', 'met_Signif>3']

#0J
highMET_0J  = ['nJet30==0', 'met_Et>50']

highMLLb_0J  = ['L3_minMll>12.', 'L3_minMll<15.', 'L3_mt2leplsp_100_minMll<115.', 'lep1Pt>25.', 'lep2Pt>15.', 'lep3Pt>10.', 'L3_mt_minMll<50']
highMLLc_0J  = ['L3_minMll>15.', 'L3_minMll<20.', 'L3_mt2leplsp_100_minMll<120.', 'lep1Pt>25.', 'lep2Pt>15.', 'lep3Pt>10.', 'L3_mt_minMll<50']
highMLLd_0J  = ['L3_minMll>20.', 'L3_minMll<30.', 'L3_mt2leplsp_100_minMll<130.', 'lep1Pt>25.', 'lep2Pt>15.', 'lep3Pt>10.', 'L3_mt_minMll<60']
highMLLe_0J  = ['L3_minMll>30.', 'L3_minMll<40.', 'L3_mt2leplsp_100_minMll<140.', 'lep1Pt>25.', 'lep2Pt>15.', 'lep3Pt>10.', 'L3_mt_minMll<60']
highMLLf1_0J = ['L3_minMll>40.', 'L3_minMll<60.', 'L3_mt2leplsp_100_minMll<160.', 'lep1Pt>25.', 'lep2Pt>15.', 'lep3Pt>10.', 'L3_mt_minMll<70']
highMLLf2_0J = ['L3_minMll>40.', 'L3_minMll<60.', 'L3_mt2leplsp_100_minMll<160.', 'lep1Pt>25.', 'lep2Pt>15.', 'lep3Pt>10.', 'L3_mt_minMll>90']
highMLLg1_0J = ['L3_minMll>60.', 'L3_minMll<75.', 'L3_mt2leplsp_100_minMll<175.', 'lep1Pt>25.', 'lep2Pt>15.', 'lep3Pt>10.', 'L3_mt_minMll<70']
highMLLg2_0J = ['L3_minMll>60.', 'L3_minMll<75.', 'L3_mt2leplsp_100_minMll<175.', 'lep1Pt>25.', 'lep2Pt>15.', 'lep3Pt>10.', 'L3_mt_minMll>90']

#incl f and g
highMLLf_0J = ['L3_minMll>40.', 'L3_minMll<60.', 'L3_mt2leplsp_100_minMll<160.', 'lep1Pt>25.', 'lep2Pt>15.', 'lep3Pt>10.', '(L3_mt_minMll<70 || L3_mt_minMll>90)']
highMLLg_0J = ['L3_minMll>60.', 'L3_minMll<75.', 'L3_mt2leplsp_100_minMll<175.', 'lep1Pt>25.', 'lep2Pt>15.', 'lep3Pt>10.', '(L3_mt_minMll<70 || L3_mt_minMll>90)']
#

SRhigh_0Jb = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_0J + highMLLb_0J
SRhigh_0Jc = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_0J + highMLLc_0J
SRhigh_0Jd = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_0J + highMLLd_0J
SRhigh_0Je = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_0J + highMLLe_0J
SRhigh_0Jf1 = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_0J + highMLLf1_0J
SRhigh_0Jf2 = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_0J + highMLLf2_0J
SRhigh_0Jg1 = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_0J + highMLLg1_0J
SRhigh_0Jg2 = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_0J + highMLLg2_0J

cutsDict['SRhigh_0Jb']  = SRhigh_0Jb
cutsDict['SRhigh_0Jc']  = SRhigh_0Jc
cutsDict['SRhigh_0Jd']  = SRhigh_0Jd
cutsDict['SRhigh_0Je']  = SRhigh_0Je
cutsDict['SRhigh_0Jf1'] = SRhigh_0Jf1
cutsDict['SRhigh_0Jf2'] = SRhigh_0Jf2
cutsDict['SRhigh_0Jg1'] = SRhigh_0Jg1
cutsDict['SRhigh_0Jg2'] = SRhigh_0Jg2

 
#nJ
highMET_nJ  = ['nJet30>0', 'met_Et>200']
soft_lep   = ['(lep1Flavor==1 && lep1Pt>4.5 || lep1Flavor==2 && lep1Pt>3.) && (lep2Flavor==1 && lep2Pt>4.5 || lep2Flavor==2 && lep2Pt>3.) && (lep3Flavor==1 && lep3Pt>4.5 || lep3Flavor==2 && lep3Pt>3.)']

highMLLa_nJ  = ['L3_minMll>1.',  'L3_minMll<12.', 'L3_mt2leplsp_100_minMll<112.', 'L3_pT3lOverMet<0.2']
highMLLb_nJ  = ['L3_minMll>12.', 'L3_minMll<15.', 'L3_mt2leplsp_100_minMll<115.', 'L3_pT3lOverMet<0.2']
highMLLc_nJ  = ['L3_minMll>15.', 'L3_minMll<20.', 'L3_mt2leplsp_100_minMll<120.', 'L3_pT3lOverMet<0.3']
highMLLd_nJ  = ['L3_minMll>20.', 'L3_minMll<30.', 'L3_mt2leplsp_100_minMll<130.', 'L3_pT3lOverMet<0.3']
highMLLe_nJ  = ['L3_minMll>30.', 'L3_minMll<40.', 'L3_mt2leplsp_100_minMll<140.', 'L3_pT3lOverMet<0.3']
highMLLf_nJ  = ['L3_minMll>40.', 'L3_minMll<60.', 'L3_mt2leplsp_100_minMll<160.', 'L3_pT3lOverMet<1.0']
highMLLg_nJ  = ['L3_minMll>60.', 'L3_minMll<75.', 'L3_mt2leplsp_100_minMll<175.', 'L3_pT3lOverMet<1.0']

SRhigh_nJa = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_nJ + soft_lep + highMLLa_nJ
SRhigh_nJb = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_nJ + soft_lep + highMLLb_nJ
SRhigh_nJc = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_nJ + soft_lep + highMLLc_nJ
SRhigh_nJd = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_nJ + soft_lep + highMLLd_nJ
SRhigh_nJe = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_nJ + soft_lep + highMLLe_nJ
SRhigh_nJf = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_nJ + soft_lep + highMLLf_nJ
SRhigh_nJg = cuts_3L_PLT_TRG_dR + common_SR + common_highMET + highMET_nJ + soft_lep + highMLLg_nJ

cutsDict['SRhigh_nJa']  = SRhigh_nJa 
cutsDict['SRhigh_nJb']  = SRhigh_nJb
cutsDict['SRhigh_nJc']  = SRhigh_nJc
cutsDict['SRhigh_nJd']  = SRhigh_nJd
cutsDict['SRhigh_nJe']  = SRhigh_nJe
cutsDict['SRhigh_nJf']  = SRhigh_nJf
cutsDict['SRhigh_nJg']  = SRhigh_nJg




# #####################
# ## WZ REGIONS      ##
# #####################

# ## WZ CRs ##
# WZCR_0J      = 'L3_mll>81. && L3_mll<101. && nJet30==0 && nBJet20_MV2c10==0 && met_Et<50. && lep3Pt>10. && L3_mt>50.'
# WZCR_nJ      = 'L3_mll>81. && L3_mll<101. && nJet30>0  && nBJet20_MV2c10==0 && met_Et<50. && lep3Pt>10. && L3_mt>50.'

# configMgr.cutsDict['CR_0J_WZ']   = cuts_3L_PLT_TRG_dR + WZCR_0J
# configMgr.cutsDict['CR_nJ_WZ']   = cuts_3L_PLT_TRG_dR + WZCR_nJ


# ##  WZ VRs ## 
# common_WZ_VR = 'L3_maxMll<75. && L3_mll < 75. && nBJet20_MV2c10==0 && met_Signif >1.5 &&'

# WZVR_0J        = 'L3_minMll>12. && L3_minMll<75. && nJet30==0 && met_Et<50. && lep3Pt>10 && L3_mt_minMll>60. && L3_mt_minMll<90. && L3_mW_PZB_minMll>75. && L3_RWLepMet_minMll>2.6' 
# WZVR_nJ        = 'L3_minMll>12. && L3_minMll<75. && nJet30>0  && met_Et<80. && lep3Pt>10 && L3_mt_minMll>60. && L3_mt_minMll<90. && L3_pT3lOverMet>0.5'
# WZVR_nJ_lowmll = 'L3_minMll>1. && L3_minMll<9. && nJet30>0  && met_Et>80. && L3_mt_minMll>30 && L3_pT3lOverMet>0.3 && '+resonances_veto

# configMgr.cutsDict['VR_0J_WZ']      = cuts_3L_PLT_TRG_dR + common_WZ_VR +  fakes_conv + WZVR_0J 
# configMgr.cutsDict['VR_nJ_WZ']      = cuts_3L_PLT_TRG_dR + common_WZ_VR +  fakes_conv + WZVR_nJ 
# configMgr.cutsDict['VR_nJ_WZ_low_mll']  = cuts_3L_PLT_TRG_dR + common_WZ_VR + WZVR_nJ_lowmll



# Copy and rename dictionary to be used for plotting     
cutsDict_plotting = cutsDict.copy()

# Fill separate dict with the cuts as one single string to be read by Histfitter          
for sig_reg, cuts in cutsDict.iteritems():
    cutsDict[sig_reg] = ' && '.join(cuts)
