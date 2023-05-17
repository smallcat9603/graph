#!/bin/sh

txt=$1
nparts=$2

base="${txt%.txt}" # 去除最后的 ".txt"
connected="${base}.connected.txt"
metis="${base}.connected.metis.txt"
part="${base}.connected.metis.txt.part.${nparts}"

echo "--> python graph_to_connected.py ${txt}"
python graph_to_connected.py ${txt} &&
echo "--> python edgelist_to_metis.py ${connected}"
python edgelist_to_metis.py ${connected} &&
echo "--> gpmetis -contig ${metis} ${nparts}"
gpmetis -contig ${metis} ${nparts} &&
echo "--> python metis_to_subgraphs.py ${connected} ${part}"
python metis_to_subgraphs.py ${connected} ${part}
