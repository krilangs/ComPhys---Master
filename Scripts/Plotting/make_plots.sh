#!/bin/bash

#----- List with (some of) the available periods, regions and variables

### Periods

#per='data15-16 data17 data18' #data15-18'
per='data18'


### Regions

# SUSY leptons
Lep12='3L_SS_SF_12 3L_SS_DF_12 3L_OS_SF_12 3L_OS_DF_12 '
Lep13='3L_SS_SF_13 3L_SS_DF_13 3L_OS_SF_13 3L_OS_DF_13 '
Lep23='3L_SS_SF_23 3L_SS_DF_23 3L_OS_SF_23 3L_OS_DF_23 '

#reg=$Lep12
#reg=$Lep13
#reg=$Lep23
#reg=$Lep12$Lep13$Lep23

Lep='SR_3L '
Lepvtx='SR_3LClass_SF_OS_vtx123 SR_3LClass_SF_OS_vtx132 SR_3LClass_SF_OS_vtx213 SR_3LClass_DF_OS_vtx123 SR_3LClass_DF_OS_vtx132 SR_3LClass_DF_OS_vtx213 '
Bench='SR_Benchmark '

#reg=$Lep
#reg=$Lepvtx
reg=$Bench

### Variables

#var='met_Et'

# Trileptons
#var='fabs(lep2Phi-lep1Phi)'  # dPhi_12
#var='fabs(lep3Phi-lep1Phi)'  # dPhi_13
#var='fabs(lep3Phi-lep2Phi)'  # dPhi_23

#var='sqrt((lep2Phi-lep1Phi)*(lep2Phi-lep1Phi)+(lep2Eta-lep1Eta)*(lep2Eta-lep1Eta))' 
#var='sqrt((lep3Phi-lep1Phi)*(lep3Phi-lep1Phi)+(lep3Eta-lep1Eta)*(lep3Eta-lep1Eta))'
#var='sqrt((lep3Phi-lep2Phi)*(lep3Phi-lep2Phi)+(lep3Eta-lep2Eta)*(lep3Eta-lep2Eta))'

# Original features
Flavor='lep1Flavor lep2Flavor lep3Flavor '
Charge='lep1Charge lep2Charge lep3Charge '
Pt='lep1Pt lep2Pt lep3Pt '
Eta='lep1Eta lep2Eta lep3Eta '
Phi='lep1Phi lep2Phi lep3Phi '
met='met_Et met_Phi '

# Added features
pt='lep1_pt lep2_pt lep3_pt lep4_pt '
E='lep1_E lep2_E lep3_E lep4_E '
phi='lep1_phi lep2_phi lep3_phi lep4_phi '
eta='lep1_eta lep2_eta lep3_eta lep4_eta '
theta='lep1_theta lep2_theta lep3_theta lep4_theta '
px='lep1_px lep2_px lep3_px lep4_px '
py='lep1_py lep2_py lep3_py lep4_py '
pz='lep1_pz lep2_pz lep3_pz lep4_pz '
dphi='dphi_12 dphi_13 dphi_14 dphi_23 dphi_24 dphi_34 '
dR='dR_12 dR_13 dR_14 dR_23 dR_24 dR_34 '
mll='mll_12 mll_13 mll_14 mll_23 mll_24 mll_34 '
m3l='m_3l '

# Features with class-cuts and benchmark
feat_bench='m_3l met_Et '

#var=$Flavor$Charge$Pt$Eta$Phi$met  # Original features
#var=$pt$E$phi$eta$theta$px$py$pz$dphi$dR$mll$m3l   # Added features
var=$feat_bench  # Features with cuts


### Signals

sig='150 450 '
#sig='150 '


#----- Choose which of the above period(s), region(s) and variable(s)  to plot

regions=$reg
variables=$var
periods=$per
signals=$sig


#----- Plot 

plot () {
# All cuts applied
  #./plot.py -s $region -v $variable -p $periode -N -Tri  # Original Ntuples
  ./plot.py -s $region -v $variable -p $periode -S $signal -N -Tri   # Classified Ntuples
}


for periode in $periods  
do
    for signal in $signals
    do
	for region in $regions
	do
	    for variable in $variables
	    do
		plot $signal $periode $region $variable
	    done
	done
    done
done

