import os
import sys
import math
import dpkt
from datetime import datetime
import collections
import numpy
import gzip
import traceback
import socket
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool
import multiprocessing


def inet_to_str(inet):
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def entropy(ctr):
    a=ctr.values()
    b=sum(a)
    al=numpy.asarray(a)
    al=al/float(b)
    return -sum(numpy.log(al)*al)

def label_packets(fin_path):
    if fin_path.split(".")[-1] != "pcap":
        return None

    with open(fin_path) as fin:
        pcap = dpkt.pcap.Reader(fin)

        for ts, buf in pcap:
            # Got time.
            time = datetime.fromtimestamp(ts)
            strtime = time.strftime('%Y-%m-%d %H:%M:%S')

            # Got IP, now match with binetflows
            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data
            str_ip = inet_to_str(ip.src)


def feature_extraction(fin_path):
    try:
        feature_array = []
        feature_names = [
            'start_time',
            "packet_count",
            "bytecount",
            "cfrag",
            "entropy_ctt1",
            "entropy_cprot",
            "cport_items",
            "csip",
            "cdip",
            "cmsbip",
            "c2msbip",
            "ctcpcon_cudpcon",
            "cpnum",
            "cpnum_items",
            "cb1",
            "cbh",
            "cws",
            "ctcphs",
            "ctcphs_items",
            "ctcpflags",
            "ctcpflags_items",
            "ip_src", # Added by David.
            "ip_dest"  # Added by David.
        ]

        slicetime = float(sys.argv[2])
        print(".... extracting features", fin_path)

        if fin_path.split(".")[-1] == "pcap":
            print 'started pcap'
            fin = open(fin_path)
            print 'using dpkt'
            pcap = dpkt.pcap.Reader(fin)
            print 'opening'
            count = 0 #packet count
            cfrag=0 #number of fragmented packets
            bytecount=0;
            cbl=0
            cbh=0
            cttl=collections.Counter()
            cprot=collections.Counter()
            csip=collections.Counter()
            cdip=collections.Counter()
            cmsbip=collections.Counter()
            c2msbip=collections.Counter()
            cpnum=collections.Counter()
            cws=collections.Counter()
            ctcphs=collections.Counter()
            ctcpflags=collections.Counter()
            ctcpcon=collections.Counter()
            cudpcon=collections.Counter()
            for ts, buf in pcap:
                if count==0:
                    stime=ts
                b=bytearray()
                b.extend(buf)
                b=b[14:]
                blen=len(b)

                if blen>=20:
                    if ts > (stime+slicetime):
                        #print "{0:.10f}".format(stime), "{0:.10f}".format(ts)
                        strtime = datetime.fromtimestamp(stime).strftime(
                                '%Y-%m-%d %H:%M:%S')
                        # feature_array.append("{0:.10f}".format(stime))
                        feature_array.append(strtime)
                        feature_array.append(count)
                        feature_array.append(bytecount)
                        feature_array.append(cfrag)
                        feature_array.append(str(len(cttl))+" "+str(entropy(cttl)))
                        feature_array.append(str(len(cprot))+" "+str(entropy(cprot)))
                        feature_array.append(cprot.items())
                        feature_array.append(str(len(csip))+" "+str(entropy(csip)))
                        feature_array.append(str(len(cdip))+" "+str(entropy(cdip)))
                        feature_array.append(str(len(cmsbip))+" "+str(entropy(cmsbip)))
                        feature_array.append(str(len(c2msbip))+" "+str(entropy(c2msbip)))
                        feature_array.append(str(len(ctcpcon))+" "+str(len(cudpcon)))
                        feature_array.append(str(len(cpnum))+" "+str(entropy(cpnum)))
                        feature_array.append(cpnum.items())
                        feature_array.append(cbl)
                        feature_array.append(cbh)
                        feature_array.append(str(entropy(cws))+" "+str(cws[0]))
                        feature_array.append(str(len(ctcphs))+" "+str(entropy(ctcphs)))
                        feature_array.append(str(ctcphs.items()))
                        feature_array.append(str(len(ctcpflags))+" "+str(entropy(ctcpflags)))
                        feature_array.append(str(ctcpflags.items()))

                        # Add Ip addresss
                        eth = dpkt.ethernet.Ethernet(buf)
                        ip = eth.data
                        feature_array.append(inet_to_str(ip.src))
                        feature_array.append(inet_to_str(ip.dst))

                        stime += slicetime
                        count=0
                        cfrag=0
                        bytecount=0
                        cttl.clear()
                        cprot.clear()
                        csip.clear()
                        cdip.clear()
                        cmsbip.clear()
                        c2msbip.clear()
                        cpnum.clear()
                        cws.clear()
                        ctcphs.clear()
                        ctcpflags.clear()
                        ctcpcon.clear()
                        cudpcon.clear()
                    count = count+1;
                    pl=(b[2]<<8)+b[3] #packet length
                    bytecount=bytecount+pl
                    if b[7]>0:  #fragment offset
                        cfrag=cfrag+1;
                    cttl[b[8]]+=1  #TTL (for entropy of TTL)
                    cprot[b[9]]+=1 #protocol of IP payload
                    sip=(b[12]<<24)+(b[13]<<16)+(b[14]<<8)+b[15]
                    dip=(b[16]<<24)+(b[17]<<16)+(b[18]<<8)+b[19]
                    csip[sip]+=1
                    cdip[dip]+=1
                    cmsbip[b[12]] += 1 #msb of IP
                    cmsbip[b[16]] += 1 #msb of IP
                    c2msbip[(b[12]<<8)+b[13]]+=1 #msb2 of IP
                    c2msbip[(b[16]<<8)+b[13]]+=1 #msb2 of IP
                    if blen>27: #has a transport payload
                        if ((b[9]==6) or (b[9]==17)):
                            pns=(b[20]<<8)+b[21]
                            pnd=(b[22]<<8)+b[23]
                            if (b[9]==6):
                                ctcpcon[(sip,dip,pns,pnd)]+=1
                                ctcpcon[(dip,sip,pnd,pns)]+=1
                            if (b[9]==17):
                                cudpcon[(sip,dip,pns,pnd)]+=1
                                cudpcon[(dip,sip,pnd,pns)]+=1
                            if pns>1023:
                                cpnum[1024]+=1
                            if pns<1024:
                                cpnum[pns]+=1
                            if pnd>1023:
                                cpnum[1024]+=1
                            if pnd<1024:
                                cpnum[pnd]+=1
                            if pnd<1024 and pns<1024:
                                cbl+=1
                            if pns>1023 and pnd>1023:
                                cbh+=1;
                            if ((blen>39) and (b[9]==6)):
                                hs=b[32]>>4
                                ctcphs[hs]+=1
                                ws=(b[34]<<8)+b[35]
                                cws[ws]+=1
                                ctcpflags[b[33]]+=1
            fin.close()
            #output file
            #out_filename = ".".join(fin_path.split(".")[0:-4])+"_ft"+sys.argv[2].strip(".")+".txt"
            # out_filename = fin_path.replace('.','_')+"_ft"+sys.argv[2].strip(".")+".txt"
            out_filename = fin_path.split('.')[0]  + "_ft"+sys.argv[2].strip(".")+".txt"
            print "writing file: %s" % out_filename
            with open(out_filename, 'w') as output_file:
                for i, feature in enumerate(feature_array):
                    output_file.write(
                            feature_names[i % len(feature_names)]
                            + ": "+ str(feature)+"\n")
            output_file.close()
            return {fin_path:feature_array}
        else:
            return None
    except Exception as e:
        print "Error"
        print str(e), '<- error'
        return None

def generate_file_paths(start_dir):
    file_paths = []
    for (path, dirs, files) in os.walk(start_dir):
        for file_name in files:
            file_path = join(path, file_name)
            if isfile(file_path):
                file_paths.append(os.path.join(path, file_name))
    return file_paths

# sys.argv[1] - directory path, sys.argv[2] - time-slice
# example: python /work/mohanty/code/multiprocess.py /work/mohanty/data/data.caida.org/passive-2009/ .125

files = generate_file_paths(sys.argv[1])

num_processes = multiprocessing.cpu_count()
processPool = Pool(num_processes)
try:
    results = processPool.map(feature_extraction, files)
except Exception as e:
    print "issue"
    print e

print(len(results))
