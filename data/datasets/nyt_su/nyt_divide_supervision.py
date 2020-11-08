import json
import random

random.seed(42)


def format_transform(old_line,rel2id):
    new_line = {}
    new_line['relation'] =old_line['relation']
    new_line['relid'] = rel2id[old_line['relation']]
    new_line['sentence'] = old_line['sentence']
    new_line['head'] = old_line['head']
    new_line['tail'] = old_line['tail']

    return new_line

def count_relation_instance(all_data:dict(),rel2id = None):
    len_list = []
    rel_list = []
    ans = dict()
    id2rel = None
    if rel2id is not None:
        id2rel = dict([v,k] for k,v in rel2id.items())
    for tt in list(all_data.keys()):
        print("rel:{} instance num:{} ".format(tt,len(all_data[tt])))
        len_list.append(len(all_data[tt]))
        rel_list.append(tt)
        if id2rel is not None:
            print("the label is :",id2rel[tt])
    return len_list


with open("./nyt_ori.json", "r") as r:
    all_data = json.load(r)

chose_begin = 20
chose_end = 2000
# build up the tran,dev,test set (6:2:2) and the test and dev set contain the instance num in [1,1000]
train = []
test = []
dev = []
train_rel = []
dev_rel = []
test_rel = []

rel_cnt = 2
instance_relid = dict()
relid_instance = dict()

length = len(all_data)
test_len = int(length * 0.4)
train_len = length - test_len

rel_instance = dict()  # relid -> instance_id
rel_ins_num = dict()
for i, item in enumerate(all_data):
    relid = item['relid']
    # if rel_ins_num.get(relid,-1) == -1:
    #     rel_ins_num
    rel_ins_num[relid] = rel_ins_num.get(relid, 0) + 1
    if rel_instance.get(relid, -1) == -1:
        rel_instance[relid] = [i]
    else:
        rel_instance[relid].append(i)

# ============
# chose from 20 - 2000
chose_rel = []
chose_num =0
for rel,num in rel_ins_num.items():
    if num in range(chose_begin,chose_end+1):
        chose_rel.append(rel)
        chose_num+=rel_ins_num[rel]

print("chose for test:",len(chose_rel))
print(chose_num/length)

# =============
# devide equally (dev : test =  1 : 1)

test_len = int(chose_num * 0.5)
dev_len = chose_num - test_len

# random.shuffle(test_rel)  # chose whether to shuffle

for rel in chose_rel:
    random.shuffle(rel_instance[rel])
    for cnt,ins_id in enumerate(rel_instance[rel]):
        if cnt < len(rel_instance[rel])*0.5:
            dev_rel.append(rel)
            dev.append(ins_id)
        else:
            test_rel.append(rel)
            test.append(ins_id)

for rel,ins in rel_instance.items():
    if rel in chose_rel:
        continue
    train_rel.append(rel)
    train = train + ins

print("train set num :", len(train))
print("train relation num :", len(train_rel))

print("dev set num :", len(dev))
print("dev relation num :", len(set(dev_rel)))
# print([rel_ins_num[id2rel(i)] for i in dev_rel])

print("test set num :", len(test))
print("test relation num :", len(set(test_rel)))

# convert instance id to item
ans_train = []
ans_dev = []
ans_test = []

for id in train:
    ans_train.append(all_data[id])

for id in dev:
    ans_dev.append(all_data[id])

for id in test:
    ans_test.append(all_data[id])

with open('./nyt_supervision_train.json','w') as r:
    json.dump(ans_train,r)

with open('./nyt_supervision_dev.json','w') as r:
    json.dump(ans_dev,r)

with open('./nyt_supervision_test.json','w') as r:
    json.dump(ans_test,r)

