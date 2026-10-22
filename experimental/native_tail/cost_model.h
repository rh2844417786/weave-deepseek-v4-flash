#pragma once
#include <cmath>
#include <cstdint>
#ifdef __CUDACC__
#define NT_HD __host__ __device__
#else
#define NT_HD
#endif
namespace weave::native_tail {
constexpr int kCostMaxPlans=22,kCostAnchors=4,kCostRow=26;
NT_HD inline double nt_add(double a,double b){
#ifdef __CUDA_ARCH__
  return __dadd_rn(a,b);
#else
  return a+b;
#endif
}
NT_HD inline double nt_mul(double a,double b){
#ifdef __CUDA_ARCH__
  return __dmul_rn(a,b);
#else
  return a*b;
#endif
}
NT_HD inline double nt_div(double a,double b){
#ifdef __CUDA_ARCH__
  return __ddiv_rn(a,b);
#else
  return a/b;
#endif
}

struct TailPlan {int comm,chunks,steal,serial;};
struct TailCurve {double scalar[5];double stage[4][4];};
NT_HD inline double curve_local(const TailCurve& c,int chunks) {
  double t=0;for(int i=0;i<5;++i)t+=c.scalar[i];
  for(int s=0;s<4;++s)for(int j=0;j<chunks;++j)t+=c.stage[s][j];return t;
}
NT_HD inline double nt_max(double a,double b){return a>b?a:b;}
NT_HD inline double feature_distance(const double* a,const double* b,int total) {
  const double scale[5]={double(total),64.,double(total),double(total)*total,double(total)};
  double d=0;for(int j=0;j<5;++j)d+=std::abs(a[j]-b[j])/scale[j];return d;
}
NT_HD inline void interpolate_curve(const double* table,int anchors,const double* f,int total,TailCurve& out) {
  int a=0,b=1;double da=1.e300,db=1.e300;
  for(int i=0;i<anchors;++i){double d=feature_distance(table+i*kCostRow,f,total);
    if(d<da){b=a;db=da;a=i;da=d;}else if(d<db){b=i;db=d;}}
  for(int j=0;j<21;++j){double x=table[a*kCostRow+5+j],y=table[b*kCostRow+5+j];
    double v=db==0?nt_mul(nt_add(x,y),.5):da==0?x:nt_div(nt_add(nt_mul(x,db),nt_mul(y,da)),nt_add(da,db));
    if(j<5)out.scalar[j]=v;else out.stage[(j-5)/4][(j-5)%4]=v;}
}
// Estimated common-start dependency DAG; never subtract absolute GPU timestamps.
NT_HD inline double dag_cost(const TailCurve* curves,int count,const int* ids,int chunks) {
  double prefix=0,ready[4],comm[4];
  for(int r=0;r<4;++r){const auto&c=curves[r*count+ids[r]];prefix=nt_max(prefix,c.scalar[0]+c.scalar[2]+c.scalar[1]);}
  for(int r=0;r<4;++r){ready[r]=prefix;comm[r]=prefix;}
  for(int j=0;j<chunks;++j){double published=0,owner[4],all=0;
    for(int r=0;r<4;++r){const auto&c=curves[r*count+ids[r]];ready[r]=nt_add(nt_add(ready[r],c.stage[0][j]),c.stage[1][j]);published=nt_max(published,ready[r]);}
    for(int r=0;r<4;++r){const auto&c=curves[r*count+ids[r]];owner[r]=nt_max(published,comm[r])+c.stage[2][j]+(j==0?c.scalar[4]:0);all=nt_max(all,owner[r]);}
    for(int r=0;r<4;++r)comm[r]=nt_max(all,owner[r])+curves[r*count+ids[r]].stage[3][j];
  }
  double end=0;for(int r=0;r<4;++r)end=nt_max(end,comm[r]+curves[r*count+ids[r]].scalar[3]);return end;
}
NT_HD inline bool lex_less(const int* a,const int* b){for(int r=0;r<4;++r){if(a[r]!=b[r])return a[r]<b[r];}return false;}
NT_HD inline void consider_plan(const TailCurve*curves,const TailPlan*plans,int count,const int*ids,int*chosen,double&best){
  double cost=dag_cost(curves,count,ids,plans[ids[0]].chunks);
  if(cost<best||(cost==best&&lex_less(ids,chosen))){best=cost;for(int r=0;r<4;++r)chosen[r]=ids[r];}}
// One bounded coordinator. Cache is preallocated global scratch, avoiding an
// 18 KiB per-thread local array or a fake dynamic shared-memory geometry change.
NT_HD inline double select_tail_plan(const double*table,const int*anchors,const double*features,
    const TailPlan*plans,int count,int total,TailCurve*cache,int*chosen){
  for(int r=0;r<4;++r)for(int p=0;p<count;++p)
    interpolate_curve(table+((r*count+p)*kCostAnchors*kCostRow),anchors[r*count+p],features+r*5,total,cache[r*count+p]);
  double best=1.e300;int ids[4]={0,0,0,0};for(int r=0;r<4;++r)chosen[r]=0;
  for(int p=0;p<count;++p){for(int r=0;r<4;++r)ids[r]=p;consider_plan(cache,plans,count,ids,chosen,best);}
  for(int chunks=1;chunks<=4;chunks*=2){int shortlist[4][2],length[4];
    for(int r=0;r<4;++r){double a=1.e300,b=1.e300;shortlist[r][0]=shortlist[r][1]=-1;length[r]=0;
      for(int p=0;p<count;++p)if(plans[p].chunks==chunks&&!plans[p].serial){
        double cost=curve_local(cache[r*count+p],chunks);
        if(cost<a){b=a;shortlist[r][1]=shortlist[r][0];a=cost;shortlist[r][0]=p;}
        else if(cost<b){b=cost;shortlist[r][1]=p;}}
      length[r]=(shortlist[r][0]>=0)+(shortlist[r][1]>=0);}
    if(!length[0]||!length[1]||!length[2]||!length[3])continue;
    for(int a=0;a<length[0];++a)for(int b=0;b<length[1];++b)for(int c=0;c<length[2];++c)for(int d=0;d<length[3];++d){
      ids[0]=shortlist[0][a];ids[1]=shortlist[1][b];ids[2]=shortlist[2][c];ids[3]=shortlist[3][d];consider_plan(cache,plans,count,ids,chosen,best);}
  }return best;
}
} // namespace
