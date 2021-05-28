#!/usr/bin/python

"""
Test for checking if final value of cutflow is identical to acc x eff x lumi x xsec
"""

import sys

9### Input                                                                                                                                                                                           
try:
    grid = sys.argv[1] # '3L3' or '3L3-updated'                                                                                                                                       
    mC1  = sys.argv[2] 
    mN1  = sys.argv[3]
    MET  = sys.argv[4] # 'high' or 'low'
    jets = sys.argv[5] # 0 or n
    bin  = sys.argv[6] # b-g (f1,f2,g1,g2)
except:
    print 'Give command line arguments: grid mC1 mN1 MET jets bin'
    sys.exit(1)  # Abort 
SR = [MET, bin]


### Extract cross sections (NB! unit nb) and filter efficiencies
xsec_feff_file = open('/afs/cern.ch/user/e/erye/private/SimpleAnalysisSara/src/SimpleAnalysis/SimpleAnalysis/scripts/txt_files/3L3_xsec_feff.txt', 'r') 
lines = xsec_feff_file.readlines()
xsec_feff_file.close()
for line in lines:
    line = line.strip().split('\t')
    if (line[0] == 'C1mN2' and str(mC1) == line[2].split('p')[0] and str(mN1) == line[3].split('p')[0]):
        #filtEff_m = line[5]
        xsec_m = float(line[4]) * 1000.  # NB! Contains BF. Original unit nb, converted to pb
        xsec_m_woBF = str(float(xsec_m) / 0.0327) # Without BF
    elif (line[0] == 'C1pN2' and str(mC1) == line[2].split('p')[0] and str(mN1) == line[3].split('p')[0]):
        #filtEff_p = line[5]
        xsec_p = float(line[4]) * 1000. # NB! Contains BF. Original unit nb, converted to pb
        xsec_p_woBF = str(float(xsec_p) / 0.0327) # Without BF
# Combine plus and minus contributions in the correct way
xsec = float(xsec_p_woBF) + float(xsec_m_woBF)
#filtEff = (float(filtEff_p) * float(xsec_p_woBF) + float(filtEff_m) * float(xsec_m_woBF) ) / xsec

### Extract acceptance
acc_file = open('/afs/cern.ch/user/e/erye/private/SimpleAnalysisSara/src/SimpleAnalysis/SimpleAnalysis/scripts/acc/'+grid+'/SR'+SR[0]+'_'+jets+'j'+SR[1]+'_acceptance_'+grid+'.dat', 'r') 
lines = acc_file.readlines()
acc_file.close()
for line in lines:
    line = line.strip().split(' ')
    if (str(mC1) == line[0] and str(mN1) == line[1]):
        acc = line[2]

### Extract efficiency
eff_file = open('/afs/cern.ch/user/e/erye/private/SimpleAnalysisSara/src/SimpleAnalysis/SimpleAnalysis/scripts/eff/'+grid+'/SR'+SR[0]+'_'+jets+'J'+SR[1]+'_efficiency_'+grid+'.dat', 'r') 
lines = eff_file.readlines()
eff_file.close()
for line in lines:
    line = line.strip().split(' ')
    if (str(mC1) == line[0] and str(mN1) == line[1]):
        eff = line[2]
        # Remove percent
        eff = float(eff) / 100. 

### Luminosity normalised to 1 pb-1
lumi = 139000

### Combine
result = float(lumi) * xsec * float(acc) * float(eff)


### Cutflow
cf_file_path = './output/SR'+SR[0]+'_'+jets+'J'+SR[1]+'_MGPy8EG_A14N23LO_C1N2_WZ_'+str(mC1)+'_'+str(mN1)+'_3L_3L3_NoSys_cutflow.txt'
cf_file = open(cf_file_path, 'r')
cf_lines = cf_file.readlines()
cf_file.close()
last_line = cf_lines[-1]
value = last_line.split(' ')[2]


print 'xsec          = ' + str(xsec)
print 'xsec x lumi   = ' + str(float(lumi)*xsec)
print ' '
print 'grid          = ' + grid
print 'mC1           = ' + str(mC1)
print 'mN1           = ' + str(mN1)
print 'jets          = ' + jets
print 'signal region = ' + str(SR)
print '------------------------------'
print 'final value of cutflow  [reco]  = ' + value
print 'lumi x xsec x acc x eff [truth] = {:.2f}'.format(result)

