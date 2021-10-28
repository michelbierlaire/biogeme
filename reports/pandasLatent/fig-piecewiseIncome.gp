set terminal epslatex mono
set output "fig-piecewiseIncome.eps" 

max(x,y) = (x >= y) ? x : y
min(x,y) = (x <= y) ? x : y

C1(x)  =  min( x , 4)
C2(x)  =  max(0,min( x - 4, 2))
C3(x)  =  max(0,min( x - 6, 2))
C4(x)  =  max(0,min( x - 8, 2))
C5(x)  =  max(0,x - 10)

set noborder
set xrange [0:12]
set xtics axis 0,1 
set noytics
set zeroaxis
set x2label "Income"
set ylabel "$x^*$"
set key outside bottom center

#set label "\\tiny{$\\hat{\\beta}_4 = -0.802$}" at 1.1,-1.2 right
#set label "\\tiny{$\\hat{\\beta}_{5} = -0.268$}" at 3,-2 right
#set label "\\tiny{$\\hat{\\beta}_{6} = -0.231$}" at 6, -2.8 right
#set label "\\tiny{$\\hat{\\beta}_{7} = -0.962$}" at 9, -4.1 right

#set label "\\tiny{Nonlinear specification: $23.6$ op\\_cost $- 3.42$ op\\_cost$^2$}" at 4,16 right
#set arrow from 3, 17 to 2,32


#set label "\\tiny{Linear specification: $10.8$  op\\_cost}" at 1, 5 left

#set arrow from 1.5, 6 to 1, 10

#Indicators only - linear regression
beta1_1 = 0.103
beta1_2 = -0.25
beta1_3 = 0.297
beta1_4 = -0.6217
beta1_5 = 0.102

f1(x) =  beta1_1 * C1(x) + beta1_2 * C2(x) +beta1_3 * C3(x) +beta1_4 * C4(x)+beta1_5 * C5(x)

# Indicators only - ordered probit
beta2_1 = 0.0897
beta2_2 = -0.221
beta2_3 = 0.259
beta2_4 = -0.523
beta2_5 = 0.0843

f2(x) =  beta2_1 * C1(x) + beta2_2 * C2(x) +beta2_3 * C3(x) +beta2_4 * C4(x)+beta2_5 * C5(x)

# Choice only 
beta3_1 = -0.0903
beta3_2 = 0.0851
beta3_3 = -0.23
beta3_4 = 0.357
beta3_5 = -0.104

f3(x) =  beta3_1 * C1(x) + beta3_2 * C2(x) +beta3_3 * C3(x) +beta3_4 * C4(x)+beta3_5 * C5(x)

# Indicators and Choice 
beta4_1 = 0.151
beta4_2 = -0.29
beta4_3 = 0.34
beta4_4 = -0.684
beta4_5 = 0.12

f4(x) =  beta4_1 * C1(x) + beta4_2 * C2(x) +beta4_3 * C3(x) +beta4_4 * C4(x)+beta4_5 * C5(x)

# Indicators and Choice + agent effect
beta5_1 = 0.147
beta5_2 = -0.314
beta5_3 = 0.396
beta5_4 = -0.687
beta5_5 = 0.128

f5(x) =  beta5_1 * C1(x) + beta5_2 * C2(x) +beta5_3 * C3(x) +beta5_4 * C4(x)+beta5_5 * C5(x)




plot f1(x) t "Indicators only -- Regression", f2(x) t "Indicators only -- Ordered probit", f3(x) t "Choice only", f4(x) t "Indicators and choice" , f5(x) t "Indicators, choice and agent effect"


