# Putting cuts as lists allows easy construction of N-1

from cuts_v2_3_lists_cutflow import cutsDict # Ordered

#____________________________________________________________________________
def configure_cuts(sig_reg, print_cuts=True):

    # Fetch list of (all) cuts
    l_cuts = cutsDict[sig_reg]

    # Initialize nested list
    nested_list = []

    # Fill nested list
    for i in range(len(l_cuts)):
        partial_list = []
        for j in range(0,i+1):
            partial_list.append(l_cuts[j])
        # Join cuts with && (AND) operator
        partial_list_string = ' && '.join(partial_list)
        nested_list.append(partial_list_string)

    # if print_cuts:
    #     print('===============================================')
    #     print('Cuts applied:')
    #     print('-----------------------'
    #     for x in l_cuts:
    #       print x
    #     print('===============================================\n')

    return l_cuts, nested_list


# Testing
#if __name__ == "__main__":
#    configure_cuts('SRhigh_0Jb')        
