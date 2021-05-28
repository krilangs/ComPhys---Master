 # 3L EWK SIGNAL REGION 
 
cutsDict = {}

## COMMON CUTS ##
cuts_3L = ['nLep_base==3 && nLep_signal==3 && L3_SFOS && nLep_base_comb==3 && lep1Signal && lep2Signal && lep3Signal && lep3IsoPLVTight']
bjet_veto = ['nBJet20_MV2c10==0']
trigger = ['((met_Et>200. && combinedTrigResultMET==1) || (segTrigMatchLEP==1 && lep2Pt>10.))']

cleaning = ['L3_minDeltaR3L>0.4']
mll_roof = ['L3_mll < 75']


## LOW MET ##
fakes_conv   = ['(((L3_isEEE || L3_isMME) && abs(L3_m3l-91.2)>20 && L3_minDeltaR<2.4 && L3_minDeltaR>0.6) || (L3_isMMM || L3_isEEM))']

# 0J
lowMLLb_0J =  ['L3_maxMll<60.', 'L3_minMll>12. && L3_minMll<15.', 'nJet30==0', 'met_Et<50', 'L3_minDeltaR<1.6',                     'met_Signif>1.5', 'L3_mt2leplsp_100_minMll<115.', 'L3_mt_minMll<50.', 'L3_pT3lOverMet<1.1']
lowMLLc_0J =  ['L3_maxMll<60.', 'L3_minMll>15. && L3_minMll<20.', 'nJet30==0', 'met_Et<50', 'L3_minDeltaR<1.6',                     'met_Signif>1.5', 'L3_mt2leplsp_100_minMll<120.', 'L3_mt_minMll<50.', 'L3_pT3lOverMet<1.1']
lowMLLd_0J =  ['L3_maxMll<60.', 'L3_minMll>20. && L3_minMll<30.', 'nJet30==0', 'met_Et<50', 'L3_minDeltaR<1.6',                     'met_Signif>1.5', 'L3_mt2leplsp_100_minMll<130.', 'L3_mt_minMll<50.', 'L3_pT3lOverMet<1.1']
lowMLLe_0J =  ['L3_maxMll<60.', 'L3_minMll>30. && L3_minMll<40.', 'nJet30==0', 'met_Et<50',                                         'met_Signif>1.5',                                 'L3_mt_minMll<60.', 'L3_pT3lOverMet<1.3']
lowMLLf1_0J = ['L3_maxMll<75.', 'L3_minMll>40. && L3_minMll<60.', 'nJet30==0', 'met_Et<50',                     'L3_m3l>100.',      'met_Signif>1.5',                                 'L3_mt_minMll<60.', 'L3_pT3lOverMet<1.4']
lowMLLf2_0J = ['L3_maxMll<75.', 'L3_minMll>40. && L3_minMll<60.', 'nJet30==0', 'met_Et<50',                     'L3_m3l>100.',      'met_Signif>1.5',                                 'L3_mt_minMll>90.', 'L3_pT3lOverMet<1.4']
lowMLLg1_0J = ['L3_maxMll<75.', 'L3_minMll>60. && L3_minMll<75.', 'nJet30==0', 'met_Et<50',                     'L3_m3l>100.',      'met_Signif>1.5',                                 'L3_mt_minMll<60.', 'L3_pT3lOverMet<1.4']
lowMLLg2_0J = ['L3_maxMll<75.', 'L3_minMll>60. && L3_minMll<75.', 'nJet30==0', 'met_Et<50',                     'L3_m3l>100.',      'met_Signif>1.5',                                 'L3_mt_minMll>90.', 'L3_pT3lOverMet<1.4']

SRlow_0Jb  = cuts_3L + bjet_veto + trigger + ['lep3Pt>10.'] + cleaning + fakes_conv + mll_roof + lowMLLb_0J
SRlow_0Jc  = cuts_3L + bjet_veto + trigger + ['lep3Pt>10.'] + cleaning + fakes_conv + mll_roof + lowMLLc_0J  
SRlow_0Jd  = cuts_3L + bjet_veto + trigger + ['lep3Pt>10.'] + cleaning + fakes_conv + mll_roof + lowMLLd_0J  
SRlow_0Je  = cuts_3L + bjet_veto + trigger + ['lep3Pt>10.'] + cleaning + fakes_conv + mll_roof + lowMLLe_0J  
SRlow_0Jf1 = cuts_3L + bjet_veto + trigger + ['lep3Pt>15.'] + cleaning + fakes_conv + mll_roof + lowMLLf1_0J 
SRlow_0Jf2 = cuts_3L + bjet_veto + trigger + ['lep3Pt>15.'] + cleaning + fakes_conv + mll_roof + lowMLLf2_0J 
SRlow_0Jg1 = cuts_3L + bjet_veto + trigger + ['lep3Pt>15.'] + cleaning + fakes_conv + mll_roof + lowMLLg1_0J 
SRlow_0Jg2 = cuts_3L + bjet_veto + trigger + ['lep3Pt>15.'] + cleaning + fakes_conv + mll_roof + lowMLLg2_0J 

cutsDict['SRlow_0Jb']  = SRlow_0Jb
cutsDict['SRlow_0Jc']  = SRlow_0Jc 
cutsDict['SRlow_0Jd']  = SRlow_0Jd
cutsDict['SRlow_0Je']  = SRlow_0Je
cutsDict['SRlow_0Jf1'] = SRlow_0Jf1
cutsDict['SRlow_0Jf2'] = SRlow_0Jf2
cutsDict['SRlow_0Jg1'] = SRlow_0Jg1
cutsDict['SRlow_0Jg2'] = SRlow_0Jg2


# nJ
lowMLLb_nJ =  ['L3_maxMll<60.', 'L3_minMll>12. && L3_minMll<15.', 'nJet30>0', 'met_Et<200', 'L3_minDeltaR<1.6', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<115.', 'L3_mt_minMll<50.', 'L3_pT3lOverMet<1.0']
lowMLLc_nJ =  ['L3_maxMll<60.', 'L3_minMll>15. && L3_minMll<20.', 'nJet30>0', 'met_Et<200', 'L3_minDeltaR<1.6', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<120.', 'L3_mt_minMll<50.', 'L3_pT3lOverMet<1.0']
lowMLLd_nJ =  ['L3_maxMll<60.', 'L3_minMll>20. && L3_minMll<30.', 'nJet30>0', 'met_Et<200', 'L3_minDeltaR<1.6', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<130.', 'L3_mt_minMll<50.', 'L3_pT3lOverMet<1.0']
lowMLLe_nJ =  ['L3_maxMll<60.', 'L3_minMll>30. && L3_minMll<40.', 'nJet30>0', 'met_Et<200',                     'met_Signif>3',                                 'L3_mt_minMll<60.', 'L3_pT3lOverMet<1.0']
lowMLLf1_nJ = ['L3_maxMll<75.', 'L3_minMll>40. && L3_minMll<60.', 'nJet30>0', 'met_Et<200',                     'met_Signif>3',                                 'L3_mt_minMll<60.', 'L3_pT3lOverMet<1.2']
lowMLLf2_nJ = ['L3_maxMll<75.', 'L3_minMll>40. && L3_minMll<60.', 'nJet30>0', 'met_Et<200',                     'met_Signif>3',                                 'L3_mt_minMll>90.', 'L3_pT3lOverMet<1.2']
lowMLLg1_nJ = ['L3_maxMll<75.', 'L3_minMll>60. && L3_minMll<75.', 'nJet30>0', 'met_Et<200',                     'met_Signif>3',                                 'L3_mt_minMll<60.', 'L3_pT3lOverMet<1.2']
lowMLLg2_nJ = ['L3_maxMll<75.', 'L3_minMll>60. && L3_minMll<75.', 'nJet30>0', 'met_Et<200',                     'met_Signif>3',                                 'L3_mt_minMll>90.', 'L3_pT3lOverMet<1.2']

SRlow_nJb  = cuts_3L + bjet_veto + trigger + ['lep3Pt>10.'] + cleaning + fakes_conv + mll_roof + lowMLLb_nJ
SRlow_nJc  = cuts_3L + bjet_veto + trigger + ['lep3Pt>10.'] + cleaning + fakes_conv + mll_roof + lowMLLc_nJ
SRlow_nJd  = cuts_3L + bjet_veto + trigger + ['lep3Pt>10.'] + cleaning + fakes_conv + mll_roof + lowMLLd_nJ
SRlow_nJe  = cuts_3L + bjet_veto + trigger + ['lep3Pt>10.'] + cleaning + fakes_conv + mll_roof + lowMLLe_nJ
SRlow_nJf1 = cuts_3L + bjet_veto + trigger + ['lep3Pt>15.'] + cleaning + fakes_conv + mll_roof + lowMLLf1_nJ
SRlow_nJf2 = cuts_3L + bjet_veto + trigger + ['lep3Pt>15.'] + cleaning + fakes_conv + mll_roof + lowMLLf2_nJ
SRlow_nJg1 = cuts_3L + bjet_veto + trigger + ['lep3Pt>15.'] + cleaning + fakes_conv + mll_roof + lowMLLg1_nJ
SRlow_nJg2 = cuts_3L + bjet_veto + trigger + ['lep3Pt>15.'] + cleaning + fakes_conv + mll_roof + lowMLLg2_nJ

cutsDict['SRlow_nJb']  = SRlow_nJb
cutsDict['SRlow_nJc']  = SRlow_nJc
cutsDict['SRlow_nJd']  = SRlow_nJd
cutsDict['SRlow_nJe']  = SRlow_nJe
cutsDict['SRlow_nJf1'] = SRlow_nJf1
cutsDict['SRlow_nJf2'] = SRlow_nJf2
cutsDict['SRlow_nJg1'] = SRlow_nJg1
cutsDict['SRlow_nJg2'] = SRlow_nJg2


## HIGH MET ## 
maxMll_roof = ['L3_maxMll<75.']

# 0J
lep_pt = ['lep1Pt>25. && lep2Pt>15. && lep3Pt>10.']

highMLLb_0J  = ['L3_minMll>12. && L3_minMll<15.', 'nJet30==0', 'met_Et>50', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<115.', 'L3_mt_minMll<50.']
highMLLc_0J  = ['L3_minMll>15. && L3_minMll<20.', 'nJet30==0', 'met_Et>50', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<120.', 'L3_mt_minMll<50.']
highMLLd_0J  = ['L3_minMll>20. && L3_minMll<30.', 'nJet30==0', 'met_Et>50', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<130.', 'L3_mt_minMll<60.']
highMLLe_0J  = ['L3_minMll>30. && L3_minMll<40.', 'nJet30==0', 'met_Et>50', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<140.', 'L3_mt_minMll<60.']
highMLLf1_0J = ['L3_minMll>40. && L3_minMll<60.', 'nJet30==0', 'met_Et>50', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<160.', 'L3_mt_minMll<70.']
highMLLf2_0J = ['L3_minMll>40. && L3_minMll<60.', 'nJet30==0', 'met_Et>50', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<160.', 'L3_mt_minMll>90.']
highMLLg1_0J = ['L3_minMll>60. && L3_minMll<75.', 'nJet30==0', 'met_Et>50', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<175.', 'L3_mt_minMll<70.']
highMLLg2_0J = ['L3_minMll>60. && L3_minMll<75.', 'nJet30==0', 'met_Et>50', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<175.', 'L3_mt_minMll>90.']

SRhigh_0Jb  = cuts_3L + bjet_veto + trigger + lep_pt + cleaning + mll_roof + maxMll_roof + highMLLb_0J
SRhigh_0Jc  = cuts_3L + bjet_veto + trigger + lep_pt + cleaning + mll_roof + maxMll_roof + highMLLc_0J
SRhigh_0Jd  = cuts_3L + bjet_veto + trigger + lep_pt + cleaning + mll_roof + maxMll_roof + highMLLd_0J
SRhigh_0Je  = cuts_3L + bjet_veto + trigger + lep_pt + cleaning + mll_roof + maxMll_roof + highMLLe_0J
SRhigh_0Jf1 = cuts_3L + bjet_veto + trigger + lep_pt + cleaning + mll_roof + maxMll_roof + highMLLf1_0J
SRhigh_0Jf2 = cuts_3L + bjet_veto + trigger + lep_pt + cleaning + mll_roof + maxMll_roof + highMLLf2_0J
SRhigh_0Jg1 = cuts_3L + bjet_veto + trigger + lep_pt + cleaning + mll_roof + maxMll_roof + highMLLg1_0J
SRhigh_0Jg2 = cuts_3L + bjet_veto + trigger + lep_pt + cleaning + mll_roof + maxMll_roof + highMLLg2_0J
 
cutsDict['SRhigh_0Jb']  = SRhigh_0Jb
cutsDict['SRhigh_0Jc']  = SRhigh_0Jc
cutsDict['SRhigh_0Jd']  = SRhigh_0Jd
cutsDict['SRhigh_0Je']  = SRhigh_0Je
cutsDict['SRhigh_0Jf1'] = SRhigh_0Jf1
cutsDict['SRhigh_0Jf2'] = SRhigh_0Jf2
cutsDict['SRhigh_0Jg1'] = SRhigh_0Jg1
cutsDict['SRhigh_0Jg2'] = SRhigh_0Jg2
 

# nJ
soft_lep   = ['(lep1Flavor==1 && lep1Pt>4.5 || lep1Flavor==2 && lep1Pt>3.) && (lep2Flavor==1 && lep2Pt>4.5 || lep2Flavor==2 && lep2Pt>3.) && (lep3Flavor==1 && lep3Pt>4.5 || lep3Flavor==2 && lep3Pt>3.)']

highMLLa_nJ  = ['L3_minMll>1.  && L3_minMll<12.', 'nJet30>0', 'met_Et>200', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<112.', 'L3_pT3lOverMet<0.2']
highMLLb_nJ  = ['L3_minMll>12. && L3_minMll<15.', 'nJet30>0', 'met_Et>200', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<115.', 'L3_pT3lOverMet<0.2']
highMLLc_nJ  = ['L3_minMll>15. && L3_minMll<20.', 'nJet30>0', 'met_Et>200', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<120.', 'L3_pT3lOverMet<0.3']
highMLLd_nJ  = ['L3_minMll>20. && L3_minMll<30.', 'nJet30>0', 'met_Et>200', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<130.', 'L3_pT3lOverMet<0.3']
highMLLe_nJ  = ['L3_minMll>30. && L3_minMll<40.', 'nJet30>0', 'met_Et>200', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<140.', 'L3_pT3lOverMet<0.3']
highMLLf_nJ  = ['L3_minMll>40. && L3_minMll<60.', 'nJet30>0', 'met_Et>200', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<160.', 'L3_pT3lOverMet<1.0']
highMLLg_nJ  = ['L3_minMll>60. && L3_minMll<75.', 'nJet30>0', 'met_Et>200', 'met_Signif>3', 'L3_mt2leplsp_100_minMll<175.', 'L3_pT3lOverMet<1.0']

SRhigh_nJa = cuts_3L + bjet_veto + trigger + soft_lep + cleaning + mll_roof + maxMll_roof + highMLLa_nJ
SRhigh_nJb = cuts_3L + bjet_veto + trigger + soft_lep + cleaning + mll_roof + maxMll_roof + highMLLb_nJ
SRhigh_nJc = cuts_3L + bjet_veto + trigger + soft_lep + cleaning + mll_roof + maxMll_roof + highMLLc_nJ
SRhigh_nJd = cuts_3L + bjet_veto + trigger + soft_lep + cleaning + mll_roof + maxMll_roof + highMLLd_nJ
SRhigh_nJe = cuts_3L + bjet_veto + trigger + soft_lep + cleaning + mll_roof + maxMll_roof + highMLLe_nJ
SRhigh_nJf = cuts_3L + bjet_veto + trigger + soft_lep + cleaning + mll_roof + maxMll_roof + highMLLf_nJ
SRhigh_nJg = cuts_3L + bjet_veto + trigger + soft_lep + cleaning + mll_roof + maxMll_roof + highMLLg_nJ

cutsDict['SRhigh_nJa']  = SRhigh_nJa 
cutsDict['SRhigh_nJb']  = SRhigh_nJb
cutsDict['SRhigh_nJc']  = SRhigh_nJc
cutsDict['SRhigh_nJd']  = SRhigh_nJd
cutsDict['SRhigh_nJe']  = SRhigh_nJe
cutsDict['SRhigh_nJf']  = SRhigh_nJf
cutsDict['SRhigh_nJg']  = SRhigh_nJg


