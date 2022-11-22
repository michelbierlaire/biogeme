#include "biogeme.h"
#include "bioDebug.h"

int main() {

  std::vector<bioString> expr ;
  expr.push_back("<Variable>{4478384840}\"CHOICE\"[33]") ;
  expr.push_back("<Beta>{4478385568}\"ASC_TRAIN\"[1]") ;
  expr.push_back("<Beta>{4478385680}\"B_TIME\"[3]") ;
  expr.push_back("<Beta>{4478385736}\"B_TIME_S\"[4]") ;
  expr.push_back("<bioDraws>{4478385848}\"B_TIME_RND\"[0]") ;
  expr.push_back("<Times>{4478385904}(2),4478385736,4478385848") ;
  expr.push_back("<Plus>{4478385960}(2),4478385680,4478385904") ;
  expr.push_back("<DefineVariable>{4411400544}\"TRAIN_TT_SCALED\"[35]") ;
  expr.push_back("<Times>{4411401608}(2),4478385960,4411400544") ;
  expr.push_back("<Plus>{4411401272}(2),4478385568,4411401608") ;
  expr.push_back("<Beta>{4478385792}\"B_COST\"[2]") ;
  expr.push_back("<DefineVariable>{4411400824}\"TRAIN_COST_SCALED\"[36]") ;
  expr.push_back("<Times>{4411401720}(2),4478385792,4411400824") ;
  expr.push_back("<Plus>{4411400880}(2),4411401272,4411401720") ;
  expr.push_back("<Beta>{4478385400}\"ASC_SM\"[5]") ;
  expr.push_back("<Beta>{4478385680}\"B_TIME\"[3]") ;
  expr.push_back("<Beta>{4478385736}\"B_TIME_S\"[4]") ;
  expr.push_back("<bioDraws>{4478385848}\"B_TIME_RND\"[0]") ;
  expr.push_back("<Times>{4478385904}(2),4478385736,4478385848") ;
  expr.push_back("<Plus>{4478385960}(2),4478385680,4478385904") ;
  expr.push_back("<DefineVariable>{4411400992}\"SM_TT_SCALED\"[37]") ;
  expr.push_back("<Times>{4411401776}(2),4478385960,4411400992") ;
  expr.push_back("<Plus>{4411401832}(2),4478385400,4411401776") ;
  expr.push_back("<Beta>{4478385792}\"B_COST\"[2]") ;
  expr.push_back("<DefineVariable>{4411401216}\"SM_COST_SCALED\"[38]") ;
  expr.push_back("<Times>{4411401888}(2),4478385792,4411401216") ;
  expr.push_back("<Plus>{4411401944}(2),4411401832,4411401888") ;
  expr.push_back("<Beta>{4478385624}\"ASC_CAR\"[0]") ;
  expr.push_back("<Beta>{4478385680}\"B_TIME\"[3]") ;
  expr.push_back("<Beta>{4478385736}\"B_TIME_S\"[4]") ;
  expr.push_back("<bioDraws>{4478385848}\"B_TIME_RND\"[0]") ;
  expr.push_back("<Times>{4478385904}(2),4478385736,4478385848") ;
  expr.push_back("<Plus>{4478385960}(2),4478385680,4478385904") ;
  expr.push_back("<DefineVariable>{4411401384}\"CAR_TT_SCALED\"[39]") ;
  expr.push_back("<Times>{4411402000}(2),4478385960,4411401384") ;
  expr.push_back("<Plus>{4411402056}(2),4478385624,4411402000") ;
  expr.push_back("<Beta>{4478385792}\"B_COST\"[2]") ;
  expr.push_back("<DefineVariable>{4411401552}\"CAR_CO_SCALED\"[40]") ;
  expr.push_back("<Times>{4411402112}(2),4478385792,4411401552") ;
  expr.push_back("<Plus>{4411402168}(2),4411402056,4411402112") ;
  expr.push_back("<DefineVariable>{4411400768}\"TRAIN_AV_SP\"[42]") ;
  expr.push_back("<Variable>{4478384224}\"SM_AV\"[23]") ;
  expr.push_back("<DefineVariable>{4411402392}\"CAR_AV_SP\"[41]") ;
  expr.push_back("<bioLogLogit>{4411402728}(3),4478384840,1,4411400880,4411400768,2,4411401944,4478384224,3,4411402168,4411402392") ;
  expr.push_back("<exp>{4411402336}(1),4411402728") ;
  expr.push_back("<MonteCarlo>{4411402784}(1),4411402336") ;
  expr.push_back("<log>{4411402224}(1),4411402784") ;
  
  biogeme bio ;
  bio.setExpressions(expr,std::vector<bioString>(),10);
  std::vector<bioReal> b ;
  b.push_back(1.0) ;
  std::vector<bioReal> fb ;
  std::vector<bioReal> aRow(36,1.0) ;
  std::vector< std::vector<bioReal> > data(6768,aRow);
  bioReal r = bio.calculateLikelihood(b,fb,data) ;
}
