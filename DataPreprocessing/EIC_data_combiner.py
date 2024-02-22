import json


#converts table with the selected subset of peak data to:
#data: {"label of what type of data list contains" : [], "160923_53_Blank.mzML Peak m/z" : [480.4254518, 483.3658964]}
#and key: {"index 0-618" : "keys to data", "7" : "160923_53_Blank.mzML Peak m/z"}
print("converting subset table to dictionaries...")
with open("Data/Table_2_Holtemme_Stichtag2015_water_ESIpos_MRP2.7_gap-filled_clean.csv", "r") as f:
    first_line = f.readline().split(",")
    data = {}
    key = {}
    for i in range(len(first_line)):
        data[first_line[i]] = []
        key[i] = first_line[i]
    
    next_line = f.readline()
    while next_line != "":
        next_line = next_line.split(",")
        for j in range(len(next_line)):
            x = next_line[j]
            if x.isdecimal():
                data[key[j]].append(float(x))
            elif x.isnumeric():
                data[key[j]].append(int(x))
            else:
                data[key[j]].append(x)
        next_line = f.readline()
    f.close()


#finds the peak m/z for every data row, skipping to a column that isnt 0 if the first instance is 0
print("extracting mzlist from subset table...")
mzlist = []
for i in range(4989):
    for j in range(51):
        if float(data[key[j*12 + 7]][i]) > 0:
            mzlist.append(float(data[key[j*12 + 7]][i]))
            break


#extracts a similar m/z list from EIC_data json to match against mzlist
print("extracting mzlist from EIC data...")
with open("Data/EIC14_JSON.json", "r") as f:
    eic = json.load(f)
    eic_mzlist = []
    eic_mzrtlist = []
    # for i in list(eic["EIC_list_14"]["160923_53_Blank.mzML"].keys()):
    for i in list(eic["160923_53_Blank.mzML"].keys()):
        eic_mzlist.append(float(i.split("@")[0]))
        eic_mzrtlist.append(i)
    f.close()


#runs through both m/z lists seeing if theyre within .01 (the same value) and when it hits an index with 
#values that are different breaks loop and prints out the section of both lists to compare which one has the extra
#the extra is then removed by manually adding the index to whichever list it belongs in
print("matching mzlists to each other...")

ex_list = [101, 578, 1146, 2356, 2408, 2548, 2622, 3308, 4121, 4340, 4361]
mx_list = []

# ex_list = [101, 578, 1146, 2355, 2407, 2547, 2621, 3307, 4120, 4339, 4360]
# mx_list = [1684, 4786]

#list for trimming down eic object by keys
mzrt_exclusion_list = []

for ex in ex_list:
    mzrt_exclusion_list.append(eic_mzrtlist.pop(ex))
    eic_mzlist.pop(ex)
for mx in mx_list:
    mzlist.pop(mx)

    #pop the unmatched row from each column in 'data'
    for d in data.keys():
        data[d].pop(mx)



for i in range(len(mzlist)):
    if abs(eic_mzlist[i] - mzlist[i]) > .01:
        print(i)
        print(eic_mzrtlist[i-2:i+2])
        print(eic_mzlist[i-2:i+2])
        print(mzlist[i-2:i+2])
        break



#extracts a dictionary for use in matching with classification data, which only has "rowid" to identify itself
#id_to_mz = {"row id" : "mz level", 1 : 100.978302}
print("extracting rowid -> mz level dictionary from full table...")
with open("Data/Table_1_Holtemme_Stichtag2015_water_ESIpos_MRP2.7_gap-filled.csv", "r") as f:
    first_line = f.readline()
    id_to_mz = {}
    next_line = f.readline()
    while next_line != "":
        next_line = next_line.split(",")
        id_to_mz[int(next_line[0])] = float(next_line[1])
        next_line = f.readline()
    f.close()




#converts classification table to new csv without the rows not matched to mzlist
print("writing new classification csv matched to mzlist...")
with open("Data/Classification_after_cleanup.csv", "r") as fin:
    with open("Data/Classification_table_UPDATED.csv", "w") as fout:
        first_line = fin.readline()
        fout.write(first_line)

        #manually added exclusion list
        clfx_list = [14477, 11484, 13394]
        i = 0
        next_line = fin.readline()
        while next_line != "" and i < 4987:
            rid = int(next_line.split(",")[0])
            if (rid in id_to_mz.keys()) and (rid not in clfx_list):
                if abs(id_to_mz[rid] - mzlist[i]) < .01:
                    fout.write(next_line)
                else:
                    print(i)
                    print(id_to_mz[rid], rid)
                    print(mzlist[i-2:i+2])
                    break
                i += 1

            next_line = fin.readline()


        fin.close()
        fout.close()


#writes updated classification data to:
#data_c: {"mzml name" : [classification data], "161010_0_Blank.mzML" : ["", 1, 0, 0,...]}
#and key_c: {"index 0-51" : "keys to data", "1" : "161010_0_Blank.mzML"}
print("new classification csv to dictionaries...")
with open("Data/Classification_table_UPDATED.csv", "r") as f:
    first_line = f.readline().split(",")
    data_c = {}
    key_c = {}
    for i in range(len(first_line)):
        data_c[first_line[i]] = []
        key_c[i] = first_line[i]
    
    next_line = f.readline()
    while next_line != "":
        next_line = next_line.strip().split(",")
        for j in range(len(next_line)):
            x = next_line[j]
            if x.isnumeric():
                data_c[key_c[j]].append(int(x))
            else:
                data_c[key_c[j]].append(x)
        next_line = f.readline()

    #fixing newline issue
    data_c["160923_53_Blank.mzML"] = data_c["160923_53_Blank.mzML\n"]
    del data_c["160923_53_Blank.mzML\n"]
    f.close()


#trims down eic to get eic14, then adds information from classification to get final object with only valid peaks.
print("creating final eic14 object with only valid peaks...")
eic14 = {}
for mzml_key in eic.keys():
    eic14[mzml_key] = {}
    for i in range(4987):
        if (data_c[mzml_key][i] != "") and (float(data['"' + mzml_key + ' Peak RT end"'][i]) != 0):
            eic14[mzml_key][eic_mzrtlist[i]] = {"intensity array" : [float(x) for x in eic[mzml_key][eic_mzrtlist[i]][0]],
                                                "rt array": [float(y) for y in eic[mzml_key][eic_mzrtlist[i]][1]],
                                                "rt start" : float(data['"' + mzml_key + ' Peak RT start"'][i])*60,
                                                "rt end" : float(data['"' + mzml_key + ' Peak RT end"'][i])*60,
                                                "mz array" : [float(data['"' + mzml_key + ' Peak m/z"'][i]) for _ in eic[mzml_key][eic_mzrtlist[i]][1]],
                                                "truth" : data_c[mzml_key][i]
                                                }
            eic14[mzml_key][eic_mzrtlist[i]]["truth array"] = []
            for rt in eic14[mzml_key][eic_mzrtlist[i]]["rt array"]:
                if eic14[mzml_key][eic_mzrtlist[i]]["rt start"] < rt < eic14[mzml_key][eic_mzrtlist[i]]["rt end"]:
                    eic14[mzml_key][eic_mzrtlist[i]]["truth array"].append(data_c[mzml_key][i])
                else:
                    eic14[mzml_key][eic_mzrtlist[i]]["truth array"].append(0)

print("dumping completed data to json...")
with open("Data/EIC14_data.json", "w") as f:
    json.dump(eic14, f, indent=1)
    f.close()

totaltrue = 0
totalpeaks = 0
durations = []
startsbefore = 0
endsafter = 0
allpeak = 0
lengths = []
peaktruescans = 0
peakfalsescans = 0
for fkey in eic14.keys():
    print(len(eic14[fkey]), fkey)
    totalpeaks += len(eic14[fkey])
    for gkey in eic14[fkey].keys():
        lengths.append(len(eic14[fkey][gkey]["rt array"]))
        if len(eic14[fkey][gkey]["rt array"]) != len(eic14[fkey][gkey]["intensity array"]):
            print("wtf", fkey, gkey)
        if eic14[fkey][gkey]["truth"] == 1:
            d = eic14[fkey][gkey]["rt end"] - eic14[fkey][gkey]["rt start"]
            durations.append(d)
            totaltrue += 1
            if eic14[fkey][gkey]["rt end"] > eic14[fkey][gkey]["rt array"][-1]:
                if eic14[fkey][gkey]["rt start"] < eic14[fkey][gkey]["rt array"][0]:
                    allpeak += 1
                else:
                    endsafter += 1
            elif eic14[fkey][gkey]["rt start"] < eic14[fkey][gkey]["rt array"][0]:
                startsbefore += 1
            
            for i in eic14[fkey][gkey]["rt array"]:
                if eic14[fkey][gkey]["rt start"] < i <eic14[fkey][gkey]["rt end"]:
                    peaktruescans += 1
                else:
                    peakfalsescans += 1

print("totaltrue", totaltrue)
print("totalpeaks", totalpeaks)
print("startsbefore", startsbefore)
print("endsafter", endsafter)
print("allpeak", allpeak)
print("maxd", max(durations))
print("mind", min(durations))
print("meand", sum(durations)/len(durations))
durations.sort()
print("1/4, 1/2, 3/4 duration", durations[11590], durations[23180], durations[34771])
lengths.sort()
print("lengths", min(lengths), max(lengths))
print("1/4, 1/2, 3/4 lengths", lengths[11590], lengths[23180], lengths[34771])
print("peaktruescans", peaktruescans)
print("peakfalsescans", peakfalsescans)
print("peaktrue%", peaktruescans/(peaktruescans+peakfalsescans))