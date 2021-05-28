#!/bin/bash

#----- List (some of) the available regions and variables

### Regions

# Low MET
SRlow_0J='SRlow_0Jb SRlow_0Jc SRlow_0Jd SRlow_0Je SRlow_0Jf1 SRlow_0Jf2 SRlow_0Jg1 SRlow_0Jg2 '
SRlow_nJ='SRlow_nJb SRlow_nJc SRlow_nJd SRlow_nJe SRlow_nJf1 SRlow_nJf2 SRlow_nJg1 SRlow_nJg2 '

# High MET
SRhigh_0J='SRhigh_0Jb SRhigh_0Jc SRhigh_0Jd SRhigh_0Je SRhigh_0Jf1 SRhigh_0Jf2 SRhigh_0Jg1 SRhigh_0Jg2 '
SRhigh_nJ='SRhigh_nJa SRhigh_nJb SRhigh_nJc SRhigh_nJd SRhigh_nJe SRhigh_nJf SRhigh_nJg '

reg=$SRlow_0J$SRlow_nJ$SRhigh_0J$SRhigh_nJ
#reg=$SRhigh_0J

### Variables

var='met_Et'


#----- Choose which of the above period(s), region(s) and variable(s) to plot

regions=$reg
variables=$var


#----- Make cutflows

cutflow () {
  ./cutflow.py -s $region -v $variable
}

for region in $regions
do
  for variable in $variables
  do
    cutflow $region $variable
    echo $region
    echo $variable
  done
done
