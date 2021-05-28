# Putting cuts as lists allows easy construction of N-1

from cuts_v2_3_lists import cutsDict_plotting

#____________________________________________________________________________
def configure_cuts(var, add_cuts, sig_reg, isData18, allNCuts, print_cuts=True):

    # From cut lists, build cut string doing N-1 appropriately
    l_cuts = cutsDict_plotting[sig_reg]

    # In case we do NOT want to skip cut on variable to be plotted
    if allNCuts:
        print('allNCuts = True')
        l_cuts_nMinus1 = l_cuts
        
    else:
        print('allNCuts = False')
        # (N-1) if variable to be plotted is in cut, do not cut on it 
        l_cuts_nMinus1 = [cut for cut in l_cuts if var not in cut]
    
    #print l_cuts_nMinus1
        
    # join cuts with && (AND) operator
    cuts = ' && '.join(l_cuts_nMinus1)
    added_cuts = cuts + ' && ' + add_cuts

    if print_cuts:
        print('===============================================')
        print('Cuts applied:')
        for x in l_cuts_nMinus1:
          print x
       # print('-----------------------------------------------')
       # print 'Unweighted final cut-string:', added_cuts
        print('===============================================\n')

    return added_cuts
