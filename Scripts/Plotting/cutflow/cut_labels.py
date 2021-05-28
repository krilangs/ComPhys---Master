
labels = {
    
    ### COMMON CUTS
    'nLep_base==3 && nLep_signal==3 && L3_SFOS && nLep_base_comb==3 && lep1Signal && lep2Signal && lep3Signal && lep3IsoPLVTight': '3 signal leptons',
    'nBJet20_MV2c10==0': '$n_{b'+r'\text{-jets}} = 0$',
    'L3_minDeltaR3L>0.4': 'min $\Delta R_{3\ell} > 0.4$',
    '((met_Et>200. && combinedTrigResultMET==1) || (segTrigMatchLEP==1 && lep2Pt>10.))': 'trigger',
    'L3_mll < 75': '$m_{\ell\ell} < 75$ GeV',

    ### OTHER CUTS
    'L3_maxMll<75.': '$m_{\ell\ell}^'+r'\text{max} < 75$ GeV',
    'L3_maxMll<60.': '$m_{\ell\ell}^'+r'\text{max} < 60$ GeV',

    'L3_minMll>1.  && L3_minMll<12.': '$m_{\ell\ell}^'+r'\text{min} \in [1, 12]$ GeV',
    'L3_minMll>12. && L3_minMll<15.': '$m_{\ell\ell}^'+r'\text{min} \in [12, 15]$ GeV',
    'L3_minMll>15. && L3_minMll<20.': '$m_{\ell\ell}^'+r'\text{min} \in [15, 20]$ GeV',
    'L3_minMll>20. && L3_minMll<30.': '$m_{\ell\ell}^'+r'\text{min} \in [20, 30]$ GeV',
    'L3_minMll>30. && L3_minMll<40.': '$m_{\ell\ell}^'+r'\text{min} \in [30, 40]$ GeV',
    'L3_minMll>40. && L3_minMll<60.': '$m_{\ell\ell}^'+r'\text{min} \in [40, 60]$ GeV',
    'L3_minMll>60. && L3_minMll<75.': '$m_{\ell\ell}^'+r'\text{min} \in [60, 75]$ GeV',

    'nJet30==0': '$n_'+r'\text{jets}^{30 '+r'\text{ GeV}} = 0$',
    'nJet30>0' : '$n_'+r'\text{jets}^{30 '+r'\text{ GeV}} \ge 1$',

    'met_Et>50' : '$E_'+r'\text{T}^'+r'\text{miss} > 50$ GeV',
    'met_Et>200': '$E_'+r'\text{T}^'+r'\text{miss} > 200$ GeV',
    'met_Et<50' : '$E_'+r'\text{T}^'+r'\text{miss} < 50$ GeV',
    'met_Et<200': '$E_'+r'\text{T}^'+r'\text{miss} < 200$ GeV',

    'lep1Pt>25. && lep2Pt>15. && lep3Pt>10.': '$p_'+r'\text{T}^{\ell_1}, p_'+r'\text{T}^{\ell_2}, p_'+r'\text{T}^{\ell_3} > 25, 15, 10$ GeV',

    'L3_mt2leplsp_100_minMll<112.': '$m_'+r'\text{T2}^{100} < 112$ GeV',
    'L3_mt2leplsp_100_minMll<115.': '$m_'+r'\text{T2}^{100} < 115$ GeV',
    'L3_mt2leplsp_100_minMll<120.': '$m_'+r'\text{T2}^{100} < 120$ GeV',
    'L3_mt2leplsp_100_minMll<130.': '$m_'+r'\text{T2}^{100} < 130$ GeV',
    'L3_mt2leplsp_100_minMll<140.': '$m_'+r'\text{T2}^{100} < 140$ GeV',
    'L3_mt2leplsp_100_minMll<160.': '$m_'+r'\text{T2}^{100} < 160$ GeV',
    'L3_mt2leplsp_100_minMll<175.': '$m_'+r'\text{T2}^{100} < 175$ GeV',

    'met_Signif>1.5': '$E_'+r'\text{T}^'+r'\text{miss}$ significance $> 1.5$',
    'met_Signif>3'  : '$E_'+r'\text{T}^'+r'\text{miss}$ significance $> 3.0$',

    'L3_mt_minMll<50.': '$m_'+r'\text{T}^'+r'\text{mllmin} < 50$ GeV',
    'L3_mt_minMll<60.': '$m_'+r'\text{T}^'+r'\text{mllmin} < 60$ GeV',
    'L3_mt_minMll<70.': '$m_'+r'\text{T}^'+r'\text{mllmin} < 70$ GeV',
    'L3_mt_minMll>90.': '$m_'+r'\text{T}^'+r'\text{mllmin} > 90$ GeV',

#    '(lep1Flavor==1 && lep1Pt>4.5 || lep1Flavor==2 && lep1Pt>3.) && (lep2Flavor==1 && lep2Pt>4.5 || lep2Flavor==2 && lep2Pt>3.) && (lep3Flavor==1 && lep3Pt>4.5 || lep3Flavor==2 && lep3Pt>3.)': 'soft leptons',
    '(lep1Flavor==1 && lep1Pt>4.5 || lep1Flavor==2 && lep1Pt>3.) && (lep2Flavor==1 && lep2Pt>4.5 || lep2Flavor==2 && lep2Pt>3.) && (lep3Flavor==1 && lep3Pt>4.5 || lep3Flavor==2 && lep3Pt>3.)': 'electron $p_'+r'\text{T}$, muon $p_'+r'\text{T} > 4.5, 3$ GeV',

    'L3_pT3lOverMet<0.2': '$|\mathbf{p}_'+r'\text{T}^'+r'\text{lep}| / E_'+r'\text{T}^'+r'\text{miss} < 0.2$',
    'L3_pT3lOverMet<0.3': '$|\mathbf{p}_'+r'\text{T}^'+r'\text{lep}| / E_'+r'\text{T}^'+r'\text{miss} < 0.3$',
    'L3_pT3lOverMet<1.0': '$|\mathbf{p}_'+r'\text{T}^'+r'\text{lep}| / E_'+r'\text{T}^'+r'\text{miss} < 1.0$',
    'L3_pT3lOverMet<1.1': '$|\mathbf{p}_'+r'\text{T}^'+r'\text{lep}| / E_'+r'\text{T}^'+r'\text{miss} < 1.1$',
    'L3_pT3lOverMet<1.2': '$|\mathbf{p}_'+r'\text{T}^'+r'\text{lep}| / E_'+r'\text{T}^'+r'\text{miss} < 1.2$',
    'L3_pT3lOverMet<1.3': '$|\mathbf{p}_'+r'\text{T}^'+r'\text{lep}| / E_'+r'\text{T}^'+r'\text{miss} < 1.3$',
    'L3_pT3lOverMet<1.4': '$|\mathbf{p}_'+r'\text{T}^'+r'\text{lep}| / E_'+r'\text{T}^'+r'\text{miss} < 1.4$',

    '(((L3_isEEE || L3_isMME) && abs(L3_m3l-91.2)>20 && L3_minDeltaR<2.4 && L3_minDeltaR>0.6) || (L3_isMMM || L3_isEEM))': 'fakes conversion',

    'lep3Pt>10.': '$p_'+r'\text{T}^{\ell_1}, p_'+r'\text{T}^{\ell_2}, p_'+r'\text{T}^{\ell_3} > 10$ GeV',
    'lep3Pt>15.': '$p_'+r'\text{T}^{\ell_1}, p_'+r'\text{T}^{\ell_2}, p_'+r'\text{T}^{\ell_3} > 15$ GeV',

    'L3_minDeltaR<1.6': 'min $\Delta R_'+r'\text{SFOS} < 1.6$',

    'L3_m3l>100.': '$m_{3 \ell} > 100$ GeV',

}




