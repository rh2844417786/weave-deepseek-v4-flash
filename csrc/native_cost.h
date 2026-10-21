// Native Marlin measured full-MoE CTA-pair planner. Apache-2.0.
#pragma once
#include <cstdint>
#include <cmath>
#if defined(__CUDACC__)
#define WEAVE_COST_HD __host__ __device__
#else
#define WEAVE_COST_HD
#endif

struct WeaveCostPrediction { int index; double us; double distance; };
struct WeaveCostAnchor { double us; double distance; bool valid; };

WEAVE_COST_HD inline bool weave_cost_finite(double v) { return v==v && v<=1.7976931348623157e308 && v>=-1.7976931348623157e308; }
// Keep cost interpolation's rounding order consistent with the CPU oracle.
// Do not change compiler flags or arithmetic for any native Marlin operation.
WEAVE_COST_HD inline double weave_cost_multiply(double a,double b) {
#if defined(__CUDA_ARCH__)
  return __dmul_rn(a,b);
#else
  volatile double product=a*b;return product;
#endif
}
WEAVE_COST_HD inline double weave_cost_add(double a,double b) {
#if defined(__CUDA_ARCH__)
  return __dadd_rn(a,b);
#else
  volatile double sum=a+b;return sum;
#endif
}
WEAVE_COST_HD inline int weave_cost_bucket(int index, int role) {
  const int position = role==0 ? index/3 : index%3;
  return position==0 ? 8 : position==1 ? 4 : 2;
}
WEAVE_COST_HD inline int weave_cost_workers(int grid,int bucket,int blocks,int n_tiles,int k_tiles) {
  if (blocks==0) return 0;
  int workers=(grid*bucket+7)/8;
  const int64_t work=int64_t(blocks)*n_tiles*k_tiles;
  return work<workers ? int(work) : workers;
}
WEAVE_COST_HD inline WeaveCostAnchor weave_cost_candidate(const float* table,const float* features,int total) {
  const double scales[5]={double(total),64.0,double(total),double(total*total),double(total)};
  double d0=1e300,d1=1e300,c0=0,c1=0;
  for (int anchor=0;anchor<6;++anchor) {
    const float* row=table+anchor*8;
    double distance=0;
    for (int f=0;f<5;++f) {
      if (!weave_cost_finite(row[f]) || row[f]<0) return {0,0,false};
      distance+=fabs(double(features[f])-double(row[f]))/scales[f];
    }
    if (!weave_cost_finite(row[5]) || !weave_cost_finite(row[6]) || !weave_cost_finite(row[7]) ||
        row[5]<=0 || row[6]<=0 || row[7]<=0) return {0,0,false};
    const double cost=double(row[5])-double(row[6])+double(row[7]);
    if (!weave_cost_finite(cost) || cost<=0) return {0,0,false};
    if (distance<d0) {d1=d0;c1=c0;d0=distance;c0=cost;}
    else if (distance<d1) {d1=distance;c1=cost;}
  }
  const double value=d1==0 ? (c0+c1)*0.5 : d0==0 ? c0 :
      weave_cost_add(weave_cost_multiply(c0,d1),weave_cost_multiply(c1,d0))/(d0+d1);
  return {value,d0,weave_cost_finite(value) && value>0};
}
WEAVE_COST_HD inline WeaveCostPrediction weave_cost_predict(const float* table,const float* features,int total) {
  WeaveCostPrediction best={-1,0,0};
  if (total<1 || total>192) return best;
  for (int i=0;i<9;++i) {
    const auto candidate=weave_cost_candidate(table+i*48,features,total);
    if (!candidate.valid) return {-1,0,0};
    if (best.index==-1 || candidate.us<best.us) best={i,candidate.us,candidate.distance};
  }
  return best;
}

#if defined(__CUDACC__)
// One CTA per MoE call. Shared per-expert rows are bounded by EP4's 64 experts.
// Existing sorted grouping is read once, before the native gate/up and down GEMMs.
__global__ void weave_cost_plan_kernel(const int32_t* sorted,const int32_t* experts,
    const int32_t* padded_ptr,int sorted_capacity,int expert_capacity,int block_size,int total,
    int64_t* plan,const float* table,const int32_t* control) {
  __shared__ int rows[64];
  __shared__ int local_blocks;
  __shared__ int invalid;
  __shared__ float features[5];
  __shared__ WeaveCostAnchor predictions[9];
  const int tid=int(threadIdx.x);
  if (tid<64) rows[tid]=0;
  if (tid==0) {local_blocks=0;invalid=0;}
  __syncthreads();
  const int padded=padded_ptr[0];
  // Bound corrupted grouping before any array access. No data-dependent
  // loop can exceed B32/topk6 plus 256 experts' native block padding.
  const bool bad=padded<0 || padded>sorted_capacity || padded>total+256*(block_size-1)
      || padded%block_size || padded/block_size>expert_capacity;
  if (bad) {if(tid==0) invalid=1;}
  else {
    for (int i=tid;i<padded;i+=128) {
      const int expert=experts[i/block_size];
      const int token=sorted[i];
      if (expert < -1 || expert>=64 || token<0 || token>total) atomicExch(&invalid,1);
      else {
        // Native Marlin processes the valid-count prefix of each block. A real
        // token after padding would make that prefix include the sentinel and
        // could address A/C out of bounds, even when the total count is valid.
        if (token<total && i%block_size!=0 && sorted[i-1]>=total) atomicExch(&invalid,1);
        if (expert>=0) {
          if (i%block_size==0) atomicAdd(&local_blocks,1);
          if (token<total) atomicAdd(rows+expert,1);
        }
      }
    }
  }
  __syncthreads();
  if (tid==0) {
    int count=0,active=0,squares=0,maximum=0;
    for(int i=0;i<64;++i) {
      const int value=rows[i];count+=value;active+=(value>0);squares+=value*value;
      if(value>maximum) maximum=value;
    }
    features[0]=float(count);features[1]=float(active);features[2]=float(local_blocks);
    features[3]=float(squares);features[4]=float(maximum);
    if(count>total || (count==0)!=(local_blocks==0)) invalid=1;
    plan[2]=bad ? 0 : padded/block_size;plan[3]=local_blocks;plan[4]=count;plan[5]=total;
    plan[6]=active;plan[7]=squares;plan[8]=maximum;plan[10]+=1;
    plan[9]=0;plan[21]=-1;plan[22]=0;plan[23]=0;
    if(control[0]!=0 && control[0]!=1) invalid=1;
  }
  __syncthreads();
  if (!invalid && control[0]==0 && tid<9)
    predictions[tid]=weave_cost_candidate(table+tid*48,features,total);
  __syncthreads();
  if (tid==0) {
    int index=-1;
    if(!invalid && control[0]==1) {
      for(int i=0;i<9;++i)
        if(weave_cost_bucket(i,0)==control[1] && weave_cost_bucket(i,1)==control[2]) index=i;
      if(index<0) invalid=1;
    } else if(!invalid) {
      for(int i=0;i<9;++i) {
        if(!predictions[i].valid) {invalid=1;break;}
        if(index<0 || predictions[i].us<predictions[index].us) index=i;
      }
      if(!invalid) {
        plan[9]=int64_t(predictions[index].us*1000.0+0.5);
        plan[22]=int64_t(predictions[index].distance*1000000.0+0.5);
      }
    }
    if(invalid) {plan[0]=0;plan[1]=0;plan[20]+=1;plan[23]=1;}
    else {
      plan[0]=weave_cost_bucket(index,0);plan[1]=weave_cost_bucket(index,1);
      plan[21]=index;plan[11+index]+=1;
    }
  }
}
#endif
