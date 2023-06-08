#!/bin/sh

graphbase=$1 # facebook git twitch livejounal
nservers=$2 # 1 2 4 6 8
server=$3 # 0
nthreads=$4 # 2 8
chunksize=$5 # 100

filebase=""
if [ "${nservers}" == "1" ]; then
    if [ "${graphbase}" == "facebook" ]; then
        filebase=../data/facebook_combined_undirected_connected
    elif [ "${graphbase}" == "git" ]; then
        filebase=../data/musae_git_edges_undirected.connected
    elif [ "${graphbase}" == "twitch" ]; then 
        filebase=../data/large_twitch_edges_undirected.connected
    elif [ "${graphbase}" == "livejounal" ]; then
        filebase=../data/soc-LiveJournal1_directed.undirected.connected
    else
        echo "graph not exist"
    fi
else
    if [ "${graphbase}" == "facebook" ]; then
        filebase=../data/${nservers}/facebook_combined_undirected_connected
    elif [ "${graphbase}" == "git" ]; then
        filebase=../data/${nservers}/musae_git_edges_undirected.connected
    elif [ "${graphbase}" == "twitch" ]; then 
        filebase=../data/${nservers}/large_twitch_edges_undirected.connected
    elif [ "${graphbase}" == "livejounal" ]; then
        filebase=../data/${nservers}/soc-LiveJournal1_directed.undirected.connected
    else
        echo "graph not exist"
    fi    
fi

echo "python3 server.py -g ${filebase} -t ${nthreads} -c ${chunksize} ${nservers} ${server}"
python3 server.py -g ${filebase} -t ${nthreads} -c ${chunksize} ${nservers} ${server}
