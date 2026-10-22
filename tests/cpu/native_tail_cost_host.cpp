#include "experimental/native_tail/cost_model.h"
#include <iostream>
int main(){using namespace weave::native_tail;int count,total;std::cin>>count>>total;
  TailPlan plans[kCostMaxPlans];int anchors[4*kCostMaxPlans],chosen[4];double features[20];
  double table[4*kCostMaxPlans*kCostAnchors*kCostRow];TailCurve cache[4*kCostMaxPlans];
  for(int p=0;p<count;++p)std::cin>>plans[p].comm>>plans[p].chunks>>plans[p].steal>>plans[p].serial;
  for(int r=0;r<4*count;++r)std::cin>>anchors[r];for(double&f:features)std::cin>>f;
  for(int i=0;i<4*count*kCostAnchors*kCostRow;++i)std::cin>>table[i];
  double cost=select_tail_plan(table,anchors,features,plans,count,total,cache,chosen);
  std::cout.precision(17);std::cout<<cost;for(int r:chosen)std::cout<<' '<<r;std::cout<<'\n';}
