set terminal epslatex  mono
set output "fig-ordinal-1.eps" 
set xrange [-2:2]
set sample 200
set noborder
set xzeroaxis
set yzeroaxis
delta1 = 0.3
delta2 = 0.9
tau1 = -delta1 - delta2
tau2 = -delta1
tau3 = delta1
tau4 = delta1 + delta2
f(x) = exp(-x*x)
f1(x) = (x <= tau3) ? 0 : f(x)
f2(x) = (x >= tau4) ? 0 : f1(x)
ub = f(0) * 1.05
set label "$\\operatorname{Pr}(I=4)=\\operatorname{Pr}(\\tau_{3} \\leq z \\leq  \\tau_4)$" at 1,0.6
set xlabel "$z^*$" offset 20
#set label at 0.02,ub "$f_{z^*}$"
#set arrow from 1,0.6 to mid,0.25
set yrange [0:ub]
set arrow from -2,0 to 3,0
#set arrow from 0,0 to 0,ub
set label at tau1,-0.02 center "$\\tau_1$"
set label at tau2,-0.02 center "$\\tau_2$"
set label at tau3,-0.02 center "$\\tau_3$"
set label at tau4,-0.02 center "$\\tau_4$"

set arrow from tau1,f(tau1) to tau1,-0.01 nohead
set arrow from tau2,f(tau2) to tau2,-0.01 nohead
set arrow from tau3,f(tau3) to tau3,-0.01 nohead
set arrow from tau4,f(tau4) to tau4,-0.01 nohead
unset xtics
unset ytics
plot f(x) t "", f2(x) w impulses t ""
