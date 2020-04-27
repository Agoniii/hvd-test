#!/bin/bash

Node_list=node1:5,node2:5,node3:5,node4:5
Process=20

nohup mpirun --allow-run-as-root -np ${Process} -H ${Node_list} -map-by slot \
--mca btl_openib_want_cuda_gdr 1 -mca coll_fca_enable 0 --mca btl_openib_if_include mlx5_2:1 \
--report-bindings --display-map --mca btl_openib_rroce_enable 1 --mca pml ob1 --mca btl ^openib \
--mca btl_openib_cpc_include rdmacm  --mca coll_hcoll_enable 0  --mca plm_rsh_no_tree_spawn 1 \
-x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=ALL \
-x NCCL_IB_GID_INDEX=3 -x NCCL_IB_HCA=mlx5_2:1 -x NCCL_IB_SL=3 -x NCCL_NET_GDR_READ=1 \
-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
-x HOROVOD_NUM_NCCL_STREAMS=1 \
-x HOROVOD_FUSION_THRESHOLD=0 \
-x NCCL_IB_DISABLE=0 \
-x LD_LIBRARY_PATH \
python3 -u train.py > gtraceglob.log 2>&1 & 
